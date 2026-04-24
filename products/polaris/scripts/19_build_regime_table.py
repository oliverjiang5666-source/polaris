"""
Build Regime-Based Trading Strategy: Clustering + DP + Classifier.

1. Cluster historical daily price curves → 6 regimes
2. For each regime, compute avg price profile
3. DP solve optimal policy: power(regime, hour, SOC) → best action
4. Train day-ahead regime classifier (predict TOMORROW's regime from TODAY's data)
5. Save lookup table + classifier

Usage:
    PYTHONPATH=. python3 scripts/19_build_regime_table.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from config import BatteryConfig

PROCESSED_DIR = Path("data/china/processed")
MODEL_DIR = Path("models")
PROVINCE = "shandong"

# ============================================================
# Config
# ============================================================
N_REGIMES = 6
N_SOC_LEVELS = 20       # 5% each: 5%, 10%, ..., 95%
POWER_STEP_MW = 4       # 4MW per level (50 charge + 50 discharge + 1 idle = 101 levels)
SOC_MIN = 0.05
SOC_MAX = 0.95


def main():
    battery = BatteryConfig()
    max_power = battery.capacity_mw  # 200 MW

    # Power levels: -200, -196, ..., 0, ..., 196, 200
    power_levels = np.arange(-max_power, max_power + POWER_STEP_MW, POWER_STEP_MW, dtype=np.float64)
    n_power = len(power_levels)
    logger.info(f"Power levels: {n_power} ({power_levels[0]:.0f} to {power_levels[-1]:.0f} MW, step={POWER_STEP_MW}MW)")

    # SOC levels
    soc_levels = np.linspace(SOC_MIN, SOC_MAX, N_SOC_LEVELS)
    logger.info(f"SOC levels: {N_SOC_LEVELS} ({soc_levels[0]:.0%} to {soc_levels[-1]:.0%})")

    # ================================================================
    # Load data
    # ================================================================
    df = pd.read_parquet(PROCESSED_DIR / f"{PROVINCE}_oracle.parquet")
    prices_all = df["rt_price"].fillna(0).values.astype(np.float64)
    n_days = len(df) // 96
    price_matrix = prices_all[:n_days * 96].reshape(n_days, 96)
    logger.info(f"Loaded {PROVINCE}: {n_days} days")

    # ================================================================
    # STEP 1: Clustering
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP 1: Clustering into {N_REGIMES} regimes")
    logger.info(f"{'=' * 60}")

    # Normalize shapes (relative to daily mean/std)
    daily_means = price_matrix.mean(axis=1, keepdims=True)
    daily_stds = np.maximum(price_matrix.std(axis=1, keepdims=True), 1.0)
    price_shapes = (price_matrix - daily_means) / daily_stds

    km = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
    labels = km.fit_predict(price_shapes)

    for c in range(N_REGIMES):
        n = (labels == c).sum()
        avg_price = price_matrix[labels == c].mean()
        avg_range = (price_matrix[labels == c].max(axis=1) - price_matrix[labels == c].min(axis=1)).mean()
        logger.info(f"  Regime {c}: {n:>5} days, avg_price={avg_price:.0f}, avg_range={avg_range:.0f}")

    # ================================================================
    # STEP 2: Price profiles per regime
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP 2: Price profiles per regime")
    logger.info(f"{'=' * 60}")

    regime_profiles = np.zeros((N_REGIMES, 96))
    for c in range(N_REGIMES):
        regime_profiles[c] = price_matrix[labels == c].mean(axis=0)
        daily_rev_potential = regime_profiles[c].max() - regime_profiles[c].min()
        logger.info(f"  Regime {c}: min={regime_profiles[c].min():.0f}, max={regime_profiles[c].max():.0f}, "
                    f"spread={daily_rev_potential:.0f}")

    # ================================================================
    # STEP 3: DP solve optimal policy for each regime
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP 3: DP solving {N_REGIMES} regime policies")
    logger.info(f"{'=' * 60}")

    interval_h = battery.interval_hours  # 0.25h
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    cap_mwh = battery.capacity_mwh
    degrad = 2.0  # CNY/MWh degradation cost

    # Lookup table: best_power[regime][hour][soc_idx] → power_level_idx
    best_power_idx = np.zeros((N_REGIMES, 96, N_SOC_LEVELS), dtype=np.int32)
    # Value function for DP
    # V[soc_idx] at time h = max future revenue from h to 95

    for regime in range(N_REGIMES):
        prices = regime_profiles[regime]

        # DP backward induction
        # V[h][s] = optimal future revenue from step h with SOC level s
        V = np.zeros((97, N_SOC_LEVELS))  # V[96][*] = 0 (end of day)

        for h in range(95, -1, -1):
            price = prices[h]

            for s_idx in range(N_SOC_LEVELS):
                soc = soc_levels[s_idx]
                best_val = -1e18
                best_p_idx = n_power // 2  # default: idle (0 MW)

                for p_idx in range(n_power):
                    power_mw = power_levels[p_idx]
                    energy_mwh = power_mw * interval_h

                    # Compute SOC change
                    if energy_mwh > 0:  # discharge
                        soc_change = -energy_mwh / cap_mwh / eta_d
                    elif energy_mwh < 0:  # charge
                        soc_change = -energy_mwh * eta_c / cap_mwh
                    else:
                        soc_change = 0.0

                    new_soc = soc + soc_change

                    # Check SOC bounds
                    if new_soc < SOC_MIN - 0.001 or new_soc > SOC_MAX + 0.001:
                        continue

                    new_soc = np.clip(new_soc, SOC_MIN, SOC_MAX)

                    # Immediate reward
                    revenue = energy_mwh * price
                    degradation = abs(energy_mwh) * degrad
                    reward = revenue - degradation

                    # Find nearest SOC level for next state
                    next_s_idx = int(np.clip(np.round((new_soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC_LEVELS - 1)), 0, N_SOC_LEVELS - 1))

                    total = reward + V[h + 1][next_s_idx]

                    if total > best_val:
                        best_val = total
                        best_p_idx = p_idx

                V[h][s_idx] = best_val
                best_power_idx[regime][h][s_idx] = best_p_idx

        # Report expected daily revenue for this regime
        mid_soc_idx = N_SOC_LEVELS // 2
        expected_rev = V[0][mid_soc_idx]
        logger.info(f"  Regime {regime}: DP expected daily revenue = {expected_rev:,.0f} (starting SOC=50%)")

    # ================================================================
    # STEP 4: Train day-ahead regime classifier
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP 4: Training day-ahead regime classifier")
    logger.info(f"{'=' * 60}")
    logger.info(f"  (Predict TOMORROW's regime from TODAY's data)")

    # Build features: today's data → tomorrow's label
    feature_cols_available = [c for c in [
        "load_norm", "renewable_penetration", "wind_ratio", "solar_ratio",
        "net_load_norm", "temperature_norm",
    ] if c in df.columns]

    X_list = []
    y_list = []

    for d in range(7, n_days - 1):  # predict d+1 from d
        feat = {}

        # Today's price stats
        today = price_matrix[d]
        feat["price_mean"] = today.mean()
        feat["price_std"] = today.std()
        feat["price_range"] = today.max() - today.min()
        feat["price_min"] = today.min()
        feat["price_max"] = today.max()
        feat["price_skew"] = float(pd.Series(today).skew())

        # Today's shape features (key hours)
        feat["morning_mean"] = today[:16].mean()    # 0-4h
        feat["midday_mean"] = today[32:48].mean()   # 8-12h
        feat["afternoon_mean"] = today[48:64].mean() # 12-16h
        feat["evening_mean"] = today[64:80].mean()  # 16-20h
        feat["night_mean"] = today[80:96].mean()    # 20-24h

        # Today's regime (as feature — yesterday's pattern predicts tomorrow's)
        feat["today_regime"] = labels[d]

        # Yesterday's regime
        feat["yesterday_regime"] = labels[d - 1]

        # 7-day rolling stats
        week = price_matrix[d-6:d+1]
        feat["week_mean"] = week.mean()
        feat["week_std"] = week.std()

        # Fundamental features (today's daily averages)
        for col in feature_cols_available:
            vals = df[col].fillna(0).values[d * 96:(d + 1) * 96]
            feat[f"{col}_mean"] = vals.mean()
            feat[f"{col}_max"] = vals.max()

        # Calendar
        date = df.index[d * 96]
        feat["weekday_tomorrow"] = (date.weekday() + 1) % 7  # tomorrow
        feat["month"] = date.month
        feat["is_weekend_tomorrow"] = 1.0 if feat["weekday_tomorrow"] >= 5 else 0.0

        X_list.append(feat)
        y_list.append(labels[d + 1])  # TOMORROW's regime

    X_df = pd.DataFrame(X_list)
    y = np.array(y_list)

    # Train/test: last 365 days as test
    split = len(X_df) - 365
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    logger.info(f"  Train accuracy: {train_acc:.1%} ({(y_pred_train == y_train).sum()}/{len(y_train)})")
    logger.info(f"  Test accuracy:  {test_acc:.1%} ({(y_pred_test == y_test).sum()}/{len(y_test)})")

    # Feature importances
    logger.info(f"\n  Top 10 features:")
    importances = sorted(zip(X_df.columns, clf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:10]:
        logger.info(f"    {feat:<30} {imp:.3f}")

    # Per-regime accuracy
    logger.info(f"\n  Per-regime test accuracy:")
    for c in range(N_REGIMES):
        mask = y_test == c
        if mask.sum() == 0:
            continue
        acc = (y_pred_test[mask] == c).sum() / mask.sum()
        logger.info(f"    Regime {c}: {acc:.1%} ({(y_pred_test[mask] == c).sum()}/{mask.sum()})")

    # ================================================================
    # STEP 5: Save everything
    # ================================================================
    logger.info(f"\n{'=' * 60}")
    logger.info(f"STEP 5: Saving models")
    logger.info(f"{'=' * 60}")

    MODEL_DIR.mkdir(exist_ok=True)

    # Save regime table
    table_path = MODEL_DIR / f"regime_table_{PROVINCE}.npz"
    np.savez(
        table_path,
        best_power_idx=best_power_idx,      # (6, 96, 20)
        power_levels=power_levels,            # (101,)
        soc_levels=soc_levels,                # (20,)
        regime_profiles=regime_profiles,      # (6, 96)
        cluster_centers=km.cluster_centers_,  # (6, 96)
        labels=labels,                        # (n_days,)
        price_matrix=price_matrix,            # for backtest
    )
    logger.info(f"  Saved regime table: {table_path}")

    # Save classifier
    clf_path = MODEL_DIR / f"regime_classifier_{PROVINCE}.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump({"classifier": clf, "feature_names": list(X_df.columns), "km": km}, f)
    logger.info(f"  Saved classifier: {clf_path}")

    # Save test predictions for backtest
    pred_path = MODEL_DIR / f"regime_predictions_{PROVINCE}.npz"
    np.savez(pred_path, y_test=y_test, y_pred=y_pred_test, test_start_day=n_days - 365)
    logger.info(f"  Saved predictions: {pred_path}")

    logger.info(f"\n  Done! Next: run scripts/20_regime_backtest.py")


if __name__ == "__main__":
    main()
