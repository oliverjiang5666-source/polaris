"""
Regime V3: All 4 improvements STACKED + All 4 provinces.

Stacking: Soft classification probabilities → weight stochastic DP scenarios.
  1. K=12 clustering (finer regimes)
  2. Tomorrow's weather features
  3. Classifier outputs probabilities (soft classification)
  4. Probabilities weight regime PROFILE scenarios → Stochastic DP

Walk-forward validated (no cheating): cluster & train on TRAIN only.

Usage:
    PYTHONPATH=. python3 scripts/22_regime_v3_allprov.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier

from config import BatteryConfig
from oracle.lp_oracle import solve_day
from forecast.mpc_controller import _step_battery

PROCESSED_DIR = Path("data/china/processed")
N_REGIMES = 12
N_SOC = 20
POWER_STEP = 4
SOC_MIN, SOC_MAX = 0.05, 0.95


def solve_stochastic_dp(scenarios, weights, battery, power_levels, soc_levels):
    """Stochastic DP on weighted price scenarios."""
    n_soc = len(soc_levels)
    n_power = len(power_levels)
    ih = battery.interval_hours
    ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    cap = battery.capacity_mwh

    # Pre-compute expected prices per timestep (for speed)
    expected_prices = np.zeros(96)
    for i, w in enumerate(weights):
        expected_prices += w * scenarios[i]

    V = np.zeros((97, n_soc))
    best_p = np.zeros((96, n_soc), dtype=np.int32)

    for h in range(95, -1, -1):
        # Weighted price at this hour
        wp = expected_prices[h]

        for s_idx in range(n_soc):
            soc = soc_levels[s_idx]
            best_val, best_idx = -1e18, n_power // 2

            for p_idx in range(n_power):
                pw = power_levels[p_idx]
                e = pw * ih
                if e > 0:
                    sc = -e / cap / ed
                elif e < 0:
                    sc = -e * ec / cap
                else:
                    sc = 0
                ns = soc + sc
                if ns < SOC_MIN - 0.001 or ns > SOC_MAX + 0.001:
                    continue
                ns = np.clip(ns, SOC_MIN, SOC_MAX)
                ns_idx = int(np.clip(round((ns - SOC_MIN) / (SOC_MAX - SOC_MIN) * (n_soc - 1)), 0, n_soc - 1))

                reward = e * wp - abs(e) * 2.0
                total = reward + V[h + 1][ns_idx]
                if total > best_val:
                    best_val, best_idx = total, p_idx

            V[h][s_idx] = best_val
            best_p[h][s_idx] = best_idx

    return best_p


def build_features(pm, df, d, labels):
    """Day-ahead features: predict D+1 from D's data."""
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
        if d < len(labels):
            f["today_reg"] = labels[d]
        if d >= 1:
            f["yest_reg"] = labels[d - 1]

    for col in ["load_norm", "renewable_penetration", "wind_ratio",
                 "solar_ratio", "net_load_norm", "temperature_norm"]:
        if col in df.columns:
            v = df[col].fillna(0).values[d * 96:(d + 1) * 96]
            if len(v) > 0:
                f[f"{col}_m"], f[f"{col}_x"] = v.mean(), v.max()

    for col in ["temperature_norm", "wind_speed_norm", "solar_radiation_norm"]:
        if col in df.columns:
            ts, te = (d + 1) * 96, (d + 2) * 96
            if te <= len(df):
                v = df[col].fillna(0).values[ts:te]
                if len(v) > 0:
                    f[f"tw_{col}_m"], f[f"tw_{col}_x"] = v.mean(), v.max()

    ti = (d + 1) * 96
    if ti < len(df):
        dt = df.index[ti]
        f["tw_wd"], f["tw_mo"] = dt.weekday(), dt.month
        f["tw_we"] = 1.0 if dt.weekday() >= 5 else 0.0

    return f


def run_province(province):
    battery = BatteryConfig()
    power_levels = np.arange(-200, 204, POWER_STEP, dtype=np.float64)
    soc_levels = np.linspace(SOC_MIN, SOC_MAX, N_SOC)

    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    prices = df["rt_price"].fillna(0).values.astype(np.float64)
    n_days = len(df) // 96
    pm = prices[:n_days * 96].reshape(n_days, 96)

    test_days = 365
    test_start = n_days - test_days
    if test_start < 400:
        test_start = n_days // 2
        test_days = n_days - test_start

    quarter_size = test_days // 4
    quarters = []
    for q in range(4):
        qs = test_start + q * quarter_size
        qe = test_start + (q + 1) * quarter_size if q < 3 else n_days
        quarters.append((qs, qe))

    logger.info(f"\n{'#' * 60}")
    logger.info(f"  {province.upper()} — {n_days} days, test={test_days}d")
    logger.info(f"{'#' * 60}")

    # Oracle
    oracle_revs = np.zeros(n_days)
    for d in range(test_start, n_days):
        r = solve_day(pm[d], battery, init_soc=0.5)
        oracle_revs[d] = r["revenue"]

    total_soft = 0.0
    total_oracle_test = 0.0

    for qi, (qs, qe) in enumerate(quarters):
        # Cluster on train only
        train_m = pm[:qs]
        tr_means = train_m.mean(axis=1, keepdims=True)
        tr_stds = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        km = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
        km.fit((train_m - tr_means) / tr_stds)

        all_shapes = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm[d].mean(), max(pm[d].std(), 1.0)
            all_shapes[d] = (pm[d] - m) / s
        all_labels = km.predict(all_shapes)
        train_labels = all_labels[:qs]

        # Regime profiles from train
        profiles = np.zeros((N_REGIMES, 96))
        for c in range(N_REGIMES):
            mask = train_labels == c
            profiles[c] = train_m[mask].mean(axis=0) if mask.sum() > 0 else train_m.mean(axis=0)

        # Train classifier
        Xt, yt = [], []
        for d in range(7, qs - 1):
            Xt.append(build_features(pm, df, d, all_labels))
            yt.append(all_labels[d + 1])
        Xdf = pd.DataFrame(Xt)
        ya = np.array(yt)

        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        # Evaluate: soft classification + stochastic DP (STACKED)
        q_soft = 0.0
        q_oracle = 0.0

        for d in range(qs, qe):
            feat = build_features(pm, df, d - 1, all_labels)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns:
                    Xp[col] = 0
            Xp = Xp[Xdf.columns]

            probs = clf.predict_proba(Xp)[0]

            # STACKED: soft classification probabilities weight the regime profiles
            # Then stochastic DP optimizes over this probability-weighted set
            scenarios = []
            weights = []
            for ci, p in enumerate(probs):
                if p < 0.02:
                    continue
                rc = clf.classes_[ci]
                scenarios.append(profiles[rc])
                weights.append(p)
            scenarios = np.array(scenarios)
            weights = np.array(weights)
            weights /= weights.sum()

            policy = solve_stochastic_dp(scenarios, weights, battery, power_levels, soc_levels)

            soc = 0.5
            rev = 0.0
            for t in range(96):
                si = int(np.clip(round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC - 1)), 0, N_SOC - 1))
                pw = power_levels[policy[t][si]]
                soc, nr, _ = _step_battery(pw, pm[d, t], soc, battery)
                rev += nr

            q_soft += rev
            q_oracle += oracle_revs[d]

        total_soft += q_soft
        total_oracle_test += q_oracle
        logger.info(f"  Q{qi+1}: Soft+StochDP={q_soft:>12,.0f}  Oracle={q_oracle:>12,.0f}  Capture={q_soft/q_oracle*100:.1f}%")

    capture = total_soft / total_oracle_test * 100
    logger.info(f"  TOTAL: Soft+StochDP={total_soft:>12,.0f}  Oracle={total_oracle_test:>12,.0f}  Capture={capture:.1f}%")

    return {"province": province, "soft_stoch": total_soft, "oracle": total_oracle_test, "capture": capture}


def main():
    provinces = ["shandong", "shanxi", "guangdong", "gansu"]
    results = []

    for prov in provinces:
        r = run_province(prov)
        results.append(r)

    # Reference: LightGBM and V5 ensemble from previous experiments
    lgbm_refs = {"shandong": 44_159_515, "shanxi": 54_650_247, "guangdong": 22_336_483, "gansu": 31_034_126}
    v5_refs = {"shandong": 44_682_220, "shanxi": 55_482_645, "guangdong": 23_796_518, "gansu": 31_953_412}

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  FINAL — All Provinces (Walk-Forward, Stacked)")
    logger.info(f"{'=' * 70}")
    logger.info(f"  {'Province':<12} {'Regime V3':>14} {'V5 Ensemble':>14} {'LightGBM':>14} {'vs LGBM':>8} {'Capture':>8}")
    logger.info(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*14} {'-'*8} {'-'*8}")

    total_v3, total_v5, total_lgbm = 0, 0, 0
    for r in results:
        p = r["province"]
        v3 = r["soft_stoch"]
        v5 = v5_refs.get(p, 0)
        lgbm = lgbm_refs.get(p, 0)
        vs = (v3 - lgbm) / abs(lgbm) * 100 if lgbm else 0
        total_v3 += v3
        total_v5 += v5
        total_lgbm += lgbm
        logger.info(f"  {p:<12} {v3:>14,.0f} {v5:>14,.0f} {lgbm:>14,.0f} {vs:>+7.1f}% {r['capture']:>7.1f}%")

    vs_total = (total_v3 - total_lgbm) / abs(total_lgbm) * 100
    logger.info(f"  {'TOTAL':<12} {total_v3:>14,.0f} {total_v5:>14,.0f} {total_lgbm:>14,.0f} {vs_total:>+7.1f}%")
    logger.info(f"\n  Total incremental revenue vs LGBM: {total_v3 - total_lgbm:>+14,.0f}")
    logger.info(f"  Total incremental revenue vs V5:   {total_v3 - total_v5:>+14,.0f}")


if __name__ == "__main__":
    main()
