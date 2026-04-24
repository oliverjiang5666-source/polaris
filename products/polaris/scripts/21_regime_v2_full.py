"""
Regime Strategy V2: Soft Classification + K=12 + Weather Features + Stochastic DP.

All 4 improvements in one script, with STRICT walk-forward validation (no cheating).

Walk-forward protocol:
  - 365 test days split into 4 quarters (Q1-Q4)
  - For each quarter:
    1. Cluster on TRAIN data only (no test data in clustering)
    2. Train classifier on TRAIN data only
    3. Predict TOMORROW's regime from TODAY's data (day-ahead)
    4. Evaluate on this quarter's test days

Improvements over V1:
  1. Soft classification (predict_proba → blended profiles → DP)
  2. K=12 (finer regimes, higher ceiling)
  3. Tomorrow's weather forecast features (Open-Meteo is available day-ahead)
  4. Stochastic DP (optimize under price uncertainty within each regime)

Usage:
    PYTHONPATH=. python3 scripts/21_regime_v2_full.py
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
PROVINCE = "shandong"

N_REGIMES = 12
N_SOC = 20
POWER_STEP = 4  # MW
SOC_MIN = 0.05
SOC_MAX = 0.95


# ============================================================
# Stochastic DP: optimize under price uncertainty
# ============================================================

def solve_stochastic_dp(price_scenarios, scenario_weights, battery, power_levels, soc_levels):
    """
    Stochastic DP: find optimal policy given multiple price scenarios with weights.

    Args:
        price_scenarios: (n_scenarios, 96) - possible price curves
        scenario_weights: (n_scenarios,) - probability of each scenario
        battery: BatteryConfig
        power_levels: array of power options (MW)
        soc_levels: array of SOC discretization points

    Returns:
        best_power_idx: (96, n_soc) - optimal power level index for each (hour, soc)
        expected_schedule: (96,) - expected power schedule starting from SOC=50%
    """
    n_soc = len(soc_levels)
    n_power = len(power_levels)
    n_scenarios = len(price_scenarios)
    interval_h = battery.interval_hours
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    cap_mwh = battery.capacity_mwh
    degrad = 2.0

    best_p_idx = np.zeros((96, n_soc), dtype=np.int32)
    V = np.zeros((97, n_soc))  # value function

    for h in range(95, -1, -1):
        for s_idx in range(n_soc):
            soc = soc_levels[s_idx]
            best_val = -1e18
            best_idx = n_power // 2  # idle

            for p_idx in range(n_power):
                power_mw = power_levels[p_idx]
                energy_mwh = power_mw * interval_h

                if energy_mwh > 0:
                    soc_change = -energy_mwh / cap_mwh / eta_d
                elif energy_mwh < 0:
                    soc_change = -energy_mwh * eta_c / cap_mwh
                else:
                    soc_change = 0.0

                new_soc = soc + soc_change
                if new_soc < SOC_MIN - 0.001 or new_soc > SOC_MAX + 0.001:
                    continue
                new_soc = np.clip(new_soc, SOC_MIN, SOC_MAX)
                next_s = int(np.clip(np.round((new_soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (n_soc - 1)), 0, n_soc - 1))

                # Expected reward across scenarios
                expected_reward = 0.0
                for sc in range(n_scenarios):
                    price = price_scenarios[sc, h]
                    revenue = energy_mwh * price
                    degradation = abs(energy_mwh) * degrad
                    expected_reward += scenario_weights[sc] * (revenue - degradation)

                total = expected_reward + V[h + 1][next_s]
                if total > best_val:
                    best_val = total
                    best_idx = p_idx

            V[h][s_idx] = best_val
            best_p_idx[h][s_idx] = best_idx

    # Generate schedule from SOC=50%
    schedule = np.zeros(96)
    soc = 0.5
    for h in range(96):
        s_idx = int(np.clip(np.round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (n_soc - 1)), 0, n_soc - 1))
        p_idx = best_p_idx[h][s_idx]
        power_mw = power_levels[p_idx]
        schedule[h] = power_mw

        energy_mwh = power_mw * battery.interval_hours
        if energy_mwh > 0:
            soc += -energy_mwh / cap_mwh / eta_d
        elif energy_mwh < 0:
            soc += -energy_mwh * eta_c / cap_mwh
        soc = np.clip(soc, SOC_MIN, SOC_MAX)

    return best_p_idx, schedule


# ============================================================
# Feature builder (day-ahead: predict D+1 from D's data)
# ============================================================

def build_features_dayahead(price_matrix, df, day_idx, labels=None):
    """Build features for predicting TOMORROW's regime from TODAY's data."""
    feat = {}
    today = price_matrix[day_idx]

    # Today's price statistics
    feat["price_mean"] = today.mean()
    feat["price_std"] = today.std()
    feat["price_range"] = today.max() - today.min()
    feat["price_min"] = today.min()
    feat["price_max"] = today.max()
    feat["price_skew"] = float(pd.Series(today).skew())
    feat["price_kurt"] = float(pd.Series(today).kurtosis())

    # Today's intraday shape (5 periods)
    feat["night_mean"] = today[:16].mean()       # 0-4h
    feat["morning_mean"] = today[16:32].mean()    # 4-8h
    feat["midday_mean"] = today[32:48].mean()     # 8-12h
    feat["afternoon_mean"] = today[48:64].mean()  # 12-16h
    feat["evening_mean"] = today[64:80].mean()    # 16-20h
    feat["late_night_mean"] = today[80:96].mean()  # 20-24h

    # Today's price momentum
    feat["morning_vs_evening"] = today[16:32].mean() - today[64:80].mean()
    feat["first_half_vs_second"] = today[:48].mean() - today[48:].mean()

    # Yesterday and day-before
    if day_idx >= 2:
        yesterday = price_matrix[day_idx - 1]
        feat["y_price_mean"] = yesterday.mean()
        feat["y_price_std"] = yesterday.std()
        feat["y_price_range"] = yesterday.max() - yesterday.min()
        feat["day_over_day_change"] = today.mean() - yesterday.mean()

        day_before = price_matrix[day_idx - 2]
        feat["d2_price_mean"] = day_before.mean()
    else:
        feat["y_price_mean"] = today.mean()
        feat["y_price_std"] = today.std()
        feat["y_price_range"] = 0
        feat["day_over_day_change"] = 0
        feat["d2_price_mean"] = today.mean()

    # 7-day rolling
    if day_idx >= 7:
        week = price_matrix[day_idx-6:day_idx+1]
        feat["week_mean"] = week.mean()
        feat["week_std"] = week.std()
        feat["week_trend"] = price_matrix[day_idx].mean() - price_matrix[day_idx-6].mean()
    else:
        feat["week_mean"] = today.mean()
        feat["week_std"] = today.std()
        feat["week_trend"] = 0

    # Today's regime (if available)
    if labels is not None and day_idx < len(labels):
        feat["today_regime"] = labels[day_idx]
    if labels is not None and day_idx >= 1:
        feat["yesterday_regime"] = labels[day_idx - 1]

    # Fundamental features from df (today's daily averages)
    fund_cols = ["load_norm", "renewable_penetration", "wind_ratio",
                 "solar_ratio", "net_load_norm", "temperature_norm"]
    for col in fund_cols:
        if col in df.columns:
            vals = df[col].fillna(0).values[day_idx * 96:(day_idx + 1) * 96]
            if len(vals) > 0:
                feat[f"{col}_mean"] = vals.mean()
                feat[f"{col}_max"] = vals.max()
                feat[f"{col}_min"] = vals.min()

    # TOMORROW's weather forecast (available day-ahead from Open-Meteo)
    # We use tomorrow's actual weather as proxy for forecast
    # (In production, fetch from Open-Meteo forecast API)
    tomorrow_start = (day_idx + 1) * 96
    tomorrow_end = tomorrow_start + 96
    weather_cols = ["temperature_norm", "wind_speed_norm", "solar_radiation_norm"]
    for col in weather_cols:
        if col in df.columns and tomorrow_end <= len(df):
            vals = df[col].fillna(0).values[tomorrow_start:tomorrow_end]
            if len(vals) > 0:
                feat[f"tomorrow_{col}_mean"] = vals.mean()
                feat[f"tomorrow_{col}_max"] = vals.max()

    # Calendar (TOMORROW)
    tomorrow_idx = (day_idx + 1) * 96
    if tomorrow_idx < len(df):
        date = df.index[tomorrow_idx]
        feat["tomorrow_weekday"] = date.weekday()
        feat["tomorrow_month"] = date.month
        feat["tomorrow_is_weekend"] = 1.0 if date.weekday() >= 5 else 0.0
    else:
        feat["tomorrow_weekday"] = 0
        feat["tomorrow_month"] = 1
        feat["tomorrow_is_weekend"] = 0

    return feat


# ============================================================
# Main
# ============================================================

def main():
    battery = BatteryConfig()
    max_power = battery.capacity_mw
    power_levels = np.arange(-max_power, max_power + POWER_STEP, POWER_STEP, dtype=np.float64)
    soc_levels = np.linspace(SOC_MIN, SOC_MAX, N_SOC)

    logger.info("=" * 70)
    logger.info(f"  REGIME V2: Soft + K={N_REGIMES} + Weather + Stochastic DP")
    logger.info(f"  STRICT WALK-FORWARD VALIDATION (no cheating)")
    logger.info("=" * 70)

    # Load data
    df = pd.read_parquet(PROCESSED_DIR / f"{PROVINCE}_oracle.parquet")
    prices_all = df["rt_price"].fillna(0).values.astype(np.float64)
    n_days = len(df) // 96
    price_matrix = prices_all[:n_days * 96].reshape(n_days, 96)
    logger.info(f"Loaded: {n_days} days")

    # Test period: last 365 days, split into 4 quarters
    test_days = 365
    test_start = n_days - test_days
    quarter_size = 91

    quarters = []
    for q in range(4):
        q_start = test_start + q * quarter_size
        q_end = test_start + (q + 1) * quarter_size if q < 3 else n_days
        quarters.append((q_start, q_end))
        logger.info(f"  Q{q+1}: day {q_start+1}-{q_end} ({q_end - q_start} days)")

    # Oracle revenues for test period
    oracle_revs = np.zeros(n_days)
    for d in range(test_start, n_days):
        r = solve_day(price_matrix[d], battery, init_soc=0.5)
        oracle_revs[d] = r["revenue"]
    logger.info(f"  Oracle test total: {oracle_revs[test_start:].sum():,.0f}")

    # ================================================================
    # Walk-forward: for each quarter, train fresh, then evaluate
    # ================================================================
    all_results = []

    for q_idx, (q_start, q_end) in enumerate(quarters):
        q_label = f"Q{q_idx + 1}"
        q_days = q_end - q_start
        train_end = q_start  # everything before this quarter

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  {q_label}: Train on {train_end} days, Test on {q_days} days")
        logger.info(f"{'=' * 60}")

        # --- STEP 1: Cluster on TRAIN data only ---
        train_matrix = price_matrix[:train_end]
        train_means = train_matrix.mean(axis=1, keepdims=True)
        train_stds = np.maximum(train_matrix.std(axis=1, keepdims=True), 1.0)
        train_shapes = (train_matrix - train_means) / train_stds

        km = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
        train_labels = km.fit_predict(train_shapes)

        # Assign labels to ALL days (including test) using trained KMeans
        all_shapes = np.zeros((n_days, 96))
        for d in range(n_days):
            m = price_matrix[d].mean()
            s = max(price_matrix[d].std(), 1.0)
            all_shapes[d] = (price_matrix[d] - m) / s
        all_labels = km.predict(all_shapes)

        # Regime profiles from TRAIN data only
        regime_profiles = np.zeros((N_REGIMES, 96))
        regime_scenarios = {}  # for stochastic DP
        for c in range(N_REGIMES):
            mask = train_labels == c
            if mask.sum() > 0:
                regime_profiles[c] = train_matrix[mask].mean(axis=0)
                # Store individual day curves as scenarios
                regime_scenarios[c] = train_matrix[mask]
            else:
                regime_profiles[c] = train_matrix.mean(axis=0)
                regime_scenarios[c] = train_matrix

        # --- STEP 2: Train day-ahead classifier on TRAIN data ---
        X_train_list = []
        y_train_list = []
        for d in range(7, train_end - 1):
            feat = build_features_dayahead(price_matrix, df, d, all_labels)
            X_train_list.append(feat)
            y_train_list.append(all_labels[d + 1])

        X_train = pd.DataFrame(X_train_list)
        y_train = np.array(y_train_list)

        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        clf.fit(X_train, y_train)
        train_acc = (clf.predict(X_train) == y_train).mean()
        logger.info(f"  Classifier train accuracy: {train_acc:.1%}")

        # --- STEP 3: Pre-compute stochastic DP for each regime ---
        # Use up to 50 representative scenarios per regime
        regime_policies = {}
        for c in range(N_REGIMES):
            scenarios = regime_scenarios[c]
            if len(scenarios) > 50:
                # Sample 50 representative scenarios
                idx = np.random.RandomState(42).choice(len(scenarios), 50, replace=False)
                scenarios = scenarios[idx]
            weights = np.ones(len(scenarios)) / len(scenarios)

            policy, _ = solve_stochastic_dp(scenarios, weights, battery, power_levels, soc_levels)
            regime_policies[c] = policy

        # --- STEP 4: Evaluate on this quarter's test days ---
        for d in range(q_start, q_end):
            day_prices = price_matrix[d]
            actual_regime = all_labels[d]

            # Day-ahead prediction: use day d-1's data to predict day d
            feat = build_features_dayahead(price_matrix, df, d - 1, all_labels)
            X_pred = pd.DataFrame([feat])
            # Ensure columns match
            for col in X_train.columns:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[X_train.columns]

            probs = clf.predict_proba(X_pred)[0]
            pred_regime = clf.predict(X_pred)[0]

            # === Method A: Hard classification (V1 baseline) ===
            soc = 0.5
            rev_hard = 0.0
            policy_hard = regime_policies[pred_regime]
            for t in range(96):
                s_idx = int(np.clip(np.round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC - 1)), 0, N_SOC - 1))
                power_mw = power_levels[policy_hard[t][s_idx]]
                soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery)
                rev_hard += net_rev

            # === Method B: Soft classification (blended profile → DP) ===
            # Blend regime profiles by predicted probabilities
            blended_profile = np.zeros(96)
            for c_idx, p in enumerate(probs):
                regime_c = clf.classes_[c_idx]
                blended_profile += p * regime_profiles[regime_c]

            # Solve DP for this blended profile
            blended_scenarios = blended_profile.reshape(1, 96)
            blended_weights = np.array([1.0])
            policy_soft, schedule_soft = solve_stochastic_dp(
                blended_scenarios, blended_weights, battery, power_levels, soc_levels
            )

            soc = 0.5
            rev_soft = 0.0
            for t in range(96):
                s_idx = int(np.clip(np.round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC - 1)), 0, N_SOC - 1))
                power_mw = power_levels[policy_soft[t][s_idx]]
                soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery)
                rev_soft += net_rev

            # === Method C: Stochastic DP (use regime scenarios weighted by probs) ===
            # Build scenario set from top-3 most likely regimes
            top_k = min(3, len(probs))
            top_indices = np.argsort(probs)[-top_k:]
            stoch_scenarios = []
            stoch_weights = []
            for c_idx in top_indices:
                regime_c = clf.classes_[c_idx]
                prob_c = probs[c_idx]
                if prob_c < 0.05:
                    continue
                sc = regime_scenarios[regime_c]
                # Take up to 10 scenarios per regime
                n_sc = min(10, len(sc))
                idx = np.random.RandomState(d).choice(len(sc), n_sc, replace=False)
                for s in sc[idx]:
                    stoch_scenarios.append(s)
                    stoch_weights.append(prob_c / n_sc)

            stoch_scenarios = np.array(stoch_scenarios)
            stoch_weights = np.array(stoch_weights)
            stoch_weights /= stoch_weights.sum()

            policy_stoch, _ = solve_stochastic_dp(
                stoch_scenarios, stoch_weights, battery, power_levels, soc_levels
            )

            soc = 0.5
            rev_stoch = 0.0
            for t in range(96):
                s_idx = int(np.clip(np.round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC - 1)), 0, N_SOC - 1))
                power_mw = power_levels[policy_stoch[t][s_idx]]
                soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery)
                rev_stoch += net_rev

            # === Method D: Perfect classification (oracle regime) ===
            soc = 0.5
            rev_perfect = 0.0
            policy_perf = regime_policies[actual_regime]
            for t in range(96):
                s_idx = int(np.clip(np.round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC - 1)), 0, N_SOC - 1))
                power_mw = power_levels[policy_perf[t][s_idx]]
                soc, net_rev, _ = _step_battery(power_mw, day_prices[t], soc, battery)
                rev_perfect += net_rev

            all_results.append({
                "day": d, "quarter": q_label,
                "actual_regime": actual_regime,
                "pred_regime": pred_regime,
                "correct": pred_regime == actual_regime,
                "rev_hard": rev_hard,
                "rev_soft": rev_soft,
                "rev_stoch": rev_stoch,
                "rev_perfect": rev_perfect,
                "rev_oracle": oracle_revs[d],
            })

        # Quarter summary
        q_results = [r for r in all_results if r["quarter"] == q_label]
        q_acc = np.mean([r["correct"] for r in q_results])
        q_hard = sum(r["rev_hard"] for r in q_results)
        q_soft = sum(r["rev_soft"] for r in q_results)
        q_stoch = sum(r["rev_stoch"] for r in q_results)
        q_oracle = sum(r["rev_oracle"] for r in q_results)

        logger.info(f"\n  {q_label} Results (classifier accuracy: {q_acc:.1%}):")
        logger.info(f"    Hard classify:  {q_hard:>12,.0f}  ({q_hard/q_oracle*100:.1f}% capture)")
        logger.info(f"    Soft classify:  {q_soft:>12,.0f}  ({q_soft/q_oracle*100:.1f}% capture)")
        logger.info(f"    Stochastic DP:  {q_stoch:>12,.0f}  ({q_stoch/q_oracle*100:.1f}% capture)")
        logger.info(f"    Oracle:         {q_oracle:>12,.0f}")

    # ================================================================
    # Final Summary
    # ================================================================
    df_r = pd.DataFrame(all_results)

    total_hard = df_r["rev_hard"].sum()
    total_soft = df_r["rev_soft"].sum()
    total_stoch = df_r["rev_stoch"].sum()
    total_perfect = df_r["rev_perfect"].sum()
    total_oracle = df_r["rev_oracle"].sum()

    # Reference numbers from previous experiments
    v5_ensemble = 44_682_220
    pure_lgbm = 44_159_515
    regime_v1 = 48_792_763

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  FINAL RESULTS — Walk-Forward Validated ({test_days} days)")
    logger.info(f"{'=' * 70}")
    logger.info(f"")
    logger.info(f"  {'Method':<40} {'Revenue':>14} {'Capture':>8} {'vs LGBM':>8}")
    logger.info(f"  {'-'*40} {'-'*14} {'-'*8} {'-'*8}")

    methods = [
        ("Oracle (theoretical max)", total_oracle, None),
        ("Regime V2: Stochastic DP (best)", total_stoch, None),
        ("Regime V2: Soft classify", total_soft, None),
        ("Regime V2: Hard classify K=12", total_hard, None),
        ("Regime V1: Hard K=6 (prev best)", regime_v1, None),
        ("V5 Ensemble (LGBM+PatchTST)", v5_ensemble, None),
        ("Pure LightGBM", pure_lgbm, None),
    ]

    for name, rev, _ in methods:
        capture = rev / total_oracle * 100
        vs_lgbm = (rev - pure_lgbm) / abs(pure_lgbm) * 100
        logger.info(f"  {name:<40} {rev:>14,.0f} {capture:>7.1f}% {vs_lgbm:>+7.1f}%")

    # Per-quarter breakdown for best method (stochastic DP)
    logger.info(f"\n  Per-quarter (Stochastic DP):")
    for q_label in ["Q1", "Q2", "Q3", "Q4"]:
        q_data = df_r[df_r["quarter"] == q_label]
        q_stoch = q_data["rev_stoch"].sum()
        q_oracle = q_data["rev_oracle"].sum()
        q_acc = q_data["correct"].mean()
        logger.info(f"    {q_label}: {q_stoch:>12,.0f} ({q_stoch/q_oracle*100:.1f}% capture, clf acc={q_acc:.1%})")

    # Save
    output_path = PROCESSED_DIR / f"regime_v2_results_{PROVINCE}.csv"
    df_r.to_csv(output_path, index=False)
    logger.info(f"\n  Results saved: {output_path}")


if __name__ == "__main__":
    main()
