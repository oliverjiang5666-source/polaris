"""
Ablation Experiments: Test each hypothesis independently.

Each experiment changes ONE variable against V3 baseline.
Walk-forward validated, Shandong only.

Experiments:
  A: Add day-ahead price as feature + DP baseline
  B: Cross-day SOC (no daily reset to 50%)
  C: Clustering dimensions (3d / 10d-PCA / 24d / 96d)
  D: Rolling window (180d / 365d / all)

Usage:
    PYTHONPATH=. python3 scripts/23_ablation_experiments.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

from config import BatteryConfig
from oracle.lp_oracle import solve_day
from forecast.mpc_controller import _step_battery

PROCESSED_DIR = Path("data/china/processed")
N_REGIMES = 12
N_SOC = 20
POWER_STEP = 4
SOC_MIN, SOC_MAX = 0.05, 0.95


def solve_dp_from_prices(prices_96, battery, power_levels, soc_levels, init_soc=0.5,
                         end_soc_penalty=0.0, target_end_soc=0.5):
    """DP on a single price curve. Returns policy and schedule from init_soc."""
    n_soc = len(soc_levels)
    n_power = len(power_levels)
    ih = battery.interval_hours
    ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    cap = battery.capacity_mwh

    V = np.zeros((97, n_soc))
    # End-of-day SOC penalty
    if end_soc_penalty > 0:
        for s in range(n_soc):
            V[96][s] = -end_soc_penalty * abs(soc_levels[s] - target_end_soc) * cap

    best_p = np.zeros((96, n_soc), dtype=np.int32)

    for h in range(95, -1, -1):
        price = prices_96[h]
        for s_idx in range(n_soc):
            soc = soc_levels[s_idx]
            best_val, best_idx = -1e18, n_power // 2
            for p_idx in range(n_power):
                pw = power_levels[p_idx]
                e = pw * ih
                if e > 0: sc = -e / cap / ed
                elif e < 0: sc = -e * ec / cap
                else: sc = 0
                ns = soc + sc
                if ns < SOC_MIN - 0.001 or ns > SOC_MAX + 0.001: continue
                ns = np.clip(ns, SOC_MIN, SOC_MAX)
                ns_idx = int(np.clip(round((ns - SOC_MIN) / (SOC_MAX - SOC_MIN) * (n_soc - 1)), 0, n_soc - 1))
                reward = e * price - abs(e) * 2.0
                total = reward + V[h + 1][ns_idx]
                if total > best_val: best_val, best_idx = total, p_idx
            V[h][s_idx] = best_val
            best_p[h][s_idx] = best_idx

    # Generate schedule from init_soc
    schedule = np.zeros(96)
    soc = init_soc
    final_soc = init_soc
    for h in range(96):
        s_idx = int(np.clip(round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (n_soc - 1)), 0, n_soc - 1))
        pw = power_levels[best_p[h][s_idx]]
        schedule[h] = pw
        e = pw * ih
        if e > 0: soc += -e / cap / ed
        elif e < 0: soc += -e * ec / cap
        soc = np.clip(soc, SOC_MIN, SOC_MAX)
        final_soc = soc

    return best_p, schedule, final_soc


def simulate_day(prices_96, schedule, battery, init_soc=0.5):
    """Simulate one day, return (revenue, final_soc)."""
    soc = init_soc
    rev = 0.0
    for t in range(96):
        soc, nr, _ = _step_battery(float(schedule[t]), prices_96[t], soc, battery)
        rev += nr
    return rev, soc


def build_features(pm, df, d, labels, include_da=False):
    """Build day-ahead features. If include_da, add da_price features."""
    f = {}
    t = pm[d]
    f["price_mean"], f["price_std"] = t.mean(), t.std()
    f["price_range"] = t.max() - t.min()
    f["price_min"], f["price_max"] = t.min(), t.max()
    f["price_skew"] = float(pd.Series(t).skew())

    for i, nm in enumerate(["night", "morn", "mid", "aftn", "eve", "late"]):
        f[f"{nm}_mean"] = t[i * 16:(i + 1) * 16].mean()
    f["morn_vs_eve"] = t[16:32].mean() - t[64:80].mean()

    if d >= 2:
        y = pm[d - 1]
        f["y_mean"], f["y_std"], f["y_range"] = y.mean(), y.std(), y.max() - y.min()
        f["dod_change"] = t.mean() - y.mean()
    else:
        f["y_mean"], f["y_std"], f["y_range"], f["dod_change"] = t.mean(), t.std(), 0, 0

    if d >= 7:
        w = pm[d - 6:d + 1]
        f["wk_mean"], f["wk_std"], f["wk_trend"] = w.mean(), w.std(), pm[d].mean() - pm[d - 6].mean()
    else:
        f["wk_mean"], f["wk_std"], f["wk_trend"] = t.mean(), t.std(), 0

    if labels is not None:
        if d < len(labels): f["today_reg"] = labels[d]
        if d >= 1: f["yest_reg"] = labels[d - 1]

    for col in ["load_norm", "renewable_penetration", "wind_ratio",
                 "solar_ratio", "net_load_norm", "temperature_norm"]:
        if col in df.columns:
            v = df[col].fillna(0).values[d * 96:(d + 1) * 96]
            if len(v) > 0: f[f"{col}_m"], f[f"{col}_x"] = v.mean(), v.max()

    for col in ["temperature_norm", "wind_speed_norm", "solar_radiation_norm"]:
        if col in df.columns:
            ts, te = (d + 1) * 96, (d + 2) * 96
            if te <= len(df):
                v = df[col].fillna(0).values[ts:te]
                if len(v) > 0: f[f"tw_{col}_m"], f[f"tw_{col}_x"] = v.mean(), v.max()

    ti = (d + 1) * 96
    if ti < len(df):
        dt = df.index[ti]
        f["tw_wd"], f["tw_mo"] = dt.weekday(), dt.month
        f["tw_we"] = 1.0 if dt.weekday() >= 5 else 0.0

    # Experiment A: day-ahead price features
    if include_da and "da_price" in df.columns:
        da = df["da_price"].fillna(0).values[d * 96:(d + 1) * 96]
        if len(da) == 96:
            f["da_mean"] = da.mean()
            f["da_std"] = da.std()
            f["da_range"] = da.max() - da.min()
            f["da_rt_spread_mean"] = (t - da).mean()
            for i, nm in enumerate(["da_night", "da_morn", "da_mid", "da_aftn", "da_eve", "da_late"]):
                f[f"{nm}_mean"] = da[i * 16:(i + 1) * 16].mean()

    return f


def run_v3_baseline(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels):
    """V3 baseline: full-data clustering, 96-dim, SOC reset, no da_price."""
    quarters = []
    qs = test_start
    for q in range(4):
        qe = test_start + (q + 1) * 91 if q < 3 else n_days
        quarters.append((qs, qe))
        qs = qe

    total_rev = 0.0
    for qs, qe in quarters:
        train_m = pm[:qs]
        tr_mn = train_m.mean(axis=1, keepdims=True)
        tr_st = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        km = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=42)
        km.fit((train_m - tr_mn) / tr_st)

        all_sh = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm[d].mean(), max(pm[d].std(), 1.0)
            all_sh[d] = (pm[d] - m) / s
        all_lb = km.predict(all_sh)

        profiles = np.zeros((N_REGIMES, 96))
        for c in range(N_REGIMES):
            mask = all_lb[:qs] == c
            profiles[c] = train_m[mask].mean(axis=0) if mask.sum() > 0 else train_m.mean(axis=0)

        Xt, yt = [], []
        for d in range(7, qs - 1):
            Xt.append(build_features(pm, df, d, all_lb, include_da=False))
            yt.append(all_lb[d + 1])
        Xdf = pd.DataFrame(Xt); ya = np.array(yt)
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        for d in range(qs, qe):
            feat = build_features(pm, df, d - 1, all_lb, include_da=False)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns: Xp[col] = 0
            Xp = Xp[Xdf.columns]
            probs = clf.predict_proba(Xp)[0]

            blended = sum(p * profiles[clf.classes_[i]] for i, p in enumerate(probs) if p > 0.02)
            _, schedule, _ = solve_dp_from_prices(blended, battery, power_levels, soc_levels)
            rev, _ = simulate_day(pm[d], schedule, battery)
            total_rev += rev

    return total_rev


def run_experiment_a(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels):
    """Experiment A: Add da_price as feature + blend with regime profile."""
    quarters = [(test_start + q * 91, test_start + (q + 1) * 91 if q < 3 else n_days) for q in range(4)]
    total_rev = 0.0

    for qs, qe in quarters:
        train_m = pm[:qs]
        tr_mn = train_m.mean(axis=1, keepdims=True)
        tr_st = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        km = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=42)
        km.fit((train_m - tr_mn) / tr_st)

        all_sh = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm[d].mean(), max(pm[d].std(), 1.0)
            all_sh[d] = (pm[d] - m) / s
        all_lb = km.predict(all_sh)

        profiles = np.zeros((N_REGIMES, 96))
        for c in range(N_REGIMES):
            mask = all_lb[:qs] == c
            profiles[c] = train_m[mask].mean(axis=0) if mask.sum() > 0 else train_m.mean(axis=0)

        Xt, yt = [], []
        for d in range(7, qs - 1):
            Xt.append(build_features(pm, df, d, all_lb, include_da=True))
            yt.append(all_lb[d + 1])
        Xdf = pd.DataFrame(Xt); ya = np.array(yt)
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        da_matrix = df["da_price"].fillna(0).values[:n_days * 96].reshape(n_days, 96) if "da_price" in df.columns else None

        for d in range(qs, qe):
            feat = build_features(pm, df, d - 1, all_lb, include_da=True)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns: Xp[col] = 0
            Xp = Xp[Xdf.columns]
            probs = clf.predict_proba(Xp)[0]

            regime_blend = sum(p * profiles[clf.classes_[i]] for i, p in enumerate(probs) if p > 0.02)

            # Blend with today's da_price as proxy for tomorrow's level
            if da_matrix is not None and d - 1 < len(da_matrix):
                today_da = da_matrix[d - 1]
                # Scale today's da_price shape to regime_blend level
                da_scaled = today_da * (regime_blend.mean() / (today_da.mean() + 1e-8))
                curve = 0.4 * da_scaled + 0.6 * regime_blend
            else:
                curve = regime_blend

            _, schedule, _ = solve_dp_from_prices(curve, battery, power_levels, soc_levels)
            rev, _ = simulate_day(pm[d], schedule, battery)
            total_rev += rev

    return total_rev


def run_experiment_b(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels):
    """Experiment B: Cross-day SOC (no reset to 50%)."""
    quarters = [(test_start + q * 91, test_start + (q + 1) * 91 if q < 3 else n_days) for q in range(4)]
    total_rev = 0.0

    for qs, qe in quarters:
        train_m = pm[:qs]
        tr_mn = train_m.mean(axis=1, keepdims=True)
        tr_st = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        km = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=42)
        km.fit((train_m - tr_mn) / tr_st)

        all_sh = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm[d].mean(), max(pm[d].std(), 1.0)
            all_sh[d] = (pm[d] - m) / s
        all_lb = km.predict(all_sh)

        profiles = np.zeros((N_REGIMES, 96))
        for c in range(N_REGIMES):
            mask = all_lb[:qs] == c
            profiles[c] = train_m[mask].mean(axis=0) if mask.sum() > 0 else train_m.mean(axis=0)

        Xt, yt = [], []
        for d in range(7, qs - 1):
            Xt.append(build_features(pm, df, d, all_lb))
            yt.append(all_lb[d + 1])
        Xdf = pd.DataFrame(Xt); ya = np.array(yt)
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        # Cross-day SOC: carry over between days
        current_soc = 0.5
        for d in range(qs, qe):
            feat = build_features(pm, df, d - 1, all_lb)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns: Xp[col] = 0
            Xp = Xp[Xdf.columns]
            probs = clf.predict_proba(Xp)[0]
            blended = sum(p * profiles[clf.classes_[i]] for i, p in enumerate(probs) if p > 0.02)

            # DP from current_soc with soft end-of-day penalty
            _, schedule, final_soc = solve_dp_from_prices(
                blended, battery, power_levels, soc_levels,
                init_soc=current_soc, end_soc_penalty=50.0, target_end_soc=0.5
            )
            rev, actual_final_soc = simulate_day(pm[d], schedule, battery, init_soc=current_soc)
            total_rev += rev
            current_soc = actual_final_soc  # carry over!

    return total_rev


def run_experiment_c(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels, dim_mode="96d"):
    """Experiment C: Different clustering dimensions."""
    quarters = [(test_start + q * 91, test_start + (q + 1) * 91 if q < 3 else n_days) for q in range(4)]
    total_rev = 0.0

    for qs, qe in quarters:
        train_m = pm[:qs]
        tr_mn = train_m.mean(axis=1, keepdims=True)
        tr_st = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        train_norm = (train_m - tr_mn) / tr_st

        # Transform to target dimensionality
        if dim_mode == "3d":
            def to_3d(curves):
                spread = curves.max(axis=1) - curves.min(axis=1)
                peak_h = curves.argmax(axis=1) / 4.0
                valley_h = curves.argmin(axis=1) / 4.0
                return np.column_stack([spread, peak_h, valley_h])
            train_feat = to_3d(train_norm)
        elif dim_mode == "10d":
            pca = PCA(n_components=10, random_state=42)
            train_feat = pca.fit_transform(train_norm)
        elif dim_mode == "24d":
            train_feat = train_norm.reshape(-1, 24, 4).mean(axis=2)  # 96→24
        else:  # 96d
            train_feat = train_norm

        km = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=42)
        km.fit(train_feat)

        # Assign labels to all days
        all_norm = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm[d].mean(), max(pm[d].std(), 1.0)
            all_norm[d] = (pm[d] - m) / s

        if dim_mode == "3d":
            all_feat = to_3d(all_norm)
        elif dim_mode == "10d":
            all_feat = pca.transform(all_norm)
        elif dim_mode == "24d":
            all_feat = all_norm.reshape(-1, 24, 4).mean(axis=2)
        else:
            all_feat = all_norm

        all_lb = km.predict(all_feat)

        profiles = np.zeros((N_REGIMES, 96))
        for c in range(N_REGIMES):
            mask = all_lb[:qs] == c
            profiles[c] = train_m[mask].mean(axis=0) if mask.sum() > 0 else train_m.mean(axis=0)

        Xt, yt = [], []
        for d in range(7, qs - 1):
            Xt.append(build_features(pm, df, d, all_lb))
            yt.append(all_lb[d + 1])
        Xdf = pd.DataFrame(Xt); ya = np.array(yt)
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        for d in range(qs, qe):
            feat = build_features(pm, df, d - 1, all_lb)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns: Xp[col] = 0
            Xp = Xp[Xdf.columns]
            probs = clf.predict_proba(Xp)[0]
            blended = sum(p * profiles[clf.classes_[i]] for i, p in enumerate(probs) if p > 0.02)
            _, schedule, _ = solve_dp_from_prices(blended, battery, power_levels, soc_levels)
            rev, _ = simulate_day(pm[d], schedule, battery)
            total_rev += rev

    return total_rev


def run_experiment_d(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels, window_days=None):
    """Experiment D: Rolling window clustering."""
    quarters = [(test_start + q * 91, test_start + (q + 1) * 91 if q < 3 else n_days) for q in range(4)]
    total_rev = 0.0

    for qs, qe in quarters:
        if window_days:
            w_start = max(0, qs - window_days)
            train_m = pm[w_start:qs]
        else:
            train_m = pm[:qs]

        tr_mn = train_m.mean(axis=1, keepdims=True)
        tr_st = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        km = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=42)
        km.fit((train_m - tr_mn) / tr_st)

        all_sh = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm[d].mean(), max(pm[d].std(), 1.0)
            all_sh[d] = (pm[d] - m) / s
        all_lb = km.predict(all_sh)

        profiles = np.zeros((N_REGIMES, 96))
        train_lb = all_lb[max(0, qs - (window_days or qs)):qs] if window_days else all_lb[:qs]
        train_m_full = pm[max(0, qs - (window_days or qs)):qs] if window_days else pm[:qs]
        for c in range(N_REGIMES):
            mask = train_lb == c
            profiles[c] = train_m_full[mask].mean(axis=0) if mask.sum() > 0 else train_m_full.mean(axis=0)

        Xt, yt = [], []
        feat_start = max(7, qs - (window_days or qs)) if window_days else 7
        for d in range(feat_start, qs - 1):
            Xt.append(build_features(pm, df, d, all_lb))
            yt.append(all_lb[d + 1])
        Xdf = pd.DataFrame(Xt); ya = np.array(yt)
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        for d in range(qs, qe):
            feat = build_features(pm, df, d - 1, all_lb)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns: Xp[col] = 0
            Xp = Xp[Xdf.columns]
            probs = clf.predict_proba(Xp)[0]
            blended = sum(p * profiles[clf.classes_[i]] for i, p in enumerate(probs) if p > 0.02)
            _, schedule, _ = solve_dp_from_prices(blended, battery, power_levels, soc_levels)
            rev, _ = simulate_day(pm[d], schedule, battery)
            total_rev += rev

    return total_rev


def main():
    battery = BatteryConfig()
    power_levels = np.arange(-200, 204, POWER_STEP, dtype=np.float64)
    soc_levels = np.linspace(SOC_MIN, SOC_MAX, N_SOC)

    logger.info("=" * 70)
    logger.info("  ABLATION EXPERIMENTS — Shandong")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "shandong_oracle.parquet")
    prices = df["rt_price"].fillna(0).values.astype(np.float64)
    n_days = len(df) // 96
    pm = prices[:n_days * 96].reshape(n_days, 96)
    test_start = n_days - 365

    oracle_revs = np.zeros(n_days)
    for d in range(test_start, n_days):
        r = solve_day(pm[d], battery, init_soc=0.5)
        oracle_revs[d] = r["revenue"]
    total_oracle = oracle_revs[test_start:].sum()
    logger.info(f"Oracle: {total_oracle:,.0f}")

    results = {}

    # Baseline
    logger.info("\n--- V3 Baseline (control) ---")
    rev = run_v3_baseline(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels)
    results["V3 Baseline"] = rev
    logger.info(f"  Revenue: {rev:,.0f}  Capture: {rev/total_oracle*100:.1f}%")

    # Experiment A
    logger.info("\n--- Experiment A: Day-ahead price ---")
    rev = run_experiment_a(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels)
    results["A: DA Price"] = rev
    logger.info(f"  Revenue: {rev:,.0f}  Capture: {rev/total_oracle*100:.1f}%")

    # Experiment B
    logger.info("\n--- Experiment B: Cross-day SOC ---")
    rev = run_experiment_b(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels)
    results["B: Cross-day SOC"] = rev
    logger.info(f"  Revenue: {rev:,.0f}  Capture: {rev/total_oracle*100:.1f}%")

    # Experiment C
    for dim in ["3d", "10d", "24d"]:
        logger.info(f"\n--- Experiment C: Cluster {dim} ---")
        rev = run_experiment_c(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels, dim)
        results[f"C: {dim}"] = rev
        logger.info(f"  Revenue: {rev:,.0f}  Capture: {rev/total_oracle*100:.1f}%")

    # Experiment D
    for window in [180, 365]:
        logger.info(f"\n--- Experiment D: Window {window}d ---")
        rev = run_experiment_d(pm, df, oracle_revs, test_start, n_days, battery, power_levels, soc_levels, window)
        results[f"D: {window}d"] = rev
        logger.info(f"  Revenue: {rev:,.0f}  Capture: {rev/total_oracle*100:.1f}%")

    # Summary
    baseline = results["V3 Baseline"]
    logger.info(f"\n{'=' * 70}")
    logger.info(f"  ABLATION SUMMARY — Shandong (Walk-Forward)")
    logger.info(f"{'=' * 70}")
    logger.info(f"  {'Experiment':<25} {'Revenue':>14} {'Capture':>8} {'vs V3':>8}")
    logger.info(f"  {'-'*25} {'-'*14} {'-'*8} {'-'*8}")

    for name, rev in sorted(results.items(), key=lambda x: -x[1]):
        cap = rev / total_oracle * 100
        vs = (rev - baseline) / abs(baseline) * 100
        marker = " ***" if vs > 2 else " *" if vs > 0.5 else ""
        logger.info(f"  {name:<25} {rev:>14,.0f} {cap:>7.1f}% {vs:>+7.1f}%{marker}")

    logger.info(f"\n  Oracle:                   {total_oracle:>14,.0f}  100.0%")
    logger.info(f"  LightGBM (reference):     {'44,159,515':>14}   53.2%")


if __name__ == "__main__":
    main()
