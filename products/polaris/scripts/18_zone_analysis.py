"""
Zone-Based Strategy Analysis: Shandong electricity price exploration.

1. Price distribution → natural zone boundaries
2. Daily curve clustering → typical day patterns (regimes)
3. Weather/load correlation with regimes
4. Zone-based strategy backtest vs exact-price strategy

Usage:
    PYTHONPATH=. python3 scripts/18_zone_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter

PROCESSED_DIR = Path("data/china/processed")


def main():
    # ================================================================
    # Load data
    # ================================================================
    df = pd.read_parquet(PROCESSED_DIR / "shandong_oracle.parquet")
    prices = df["rt_price"].values
    n_days = len(df) // 96

    logger.info(f"Loaded shandong: {len(df):,} rows, {n_days} days")
    logger.info(f"Price range: {prices.min():.0f} ~ {prices.max():.0f} 元/MWh")
    logger.info(f"Price mean: {prices.mean():.0f}, std: {prices.std():.0f}")

    # ================================================================
    # PART 1: Price Zone Analysis
    # ================================================================
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# PART 1: PRICE ZONE ANALYSIS")
    logger.info(f"{'#' * 70}")

    # Use percentile-based zones (more natural than fixed width)
    n_zones = 20
    percentiles = np.linspace(0, 100, n_zones + 1)
    boundaries = np.percentile(prices, percentiles)

    logger.info(f"\n{n_zones} zones based on percentiles:")
    logger.info(f"  {'Zone':>6} {'Range':>25} {'Count':>8} {'Pct':>6}")
    logger.info(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*6}")

    zone_labels = np.digitize(prices, boundaries[1:-1])  # 0 to n_zones-1
    for z in range(n_zones):
        mask = zone_labels == z
        count = mask.sum()
        pct = count / len(prices) * 100
        lo, hi = boundaries[z], boundaries[z + 1]
        logger.info(f"  {z:>6} {lo:>10.0f} ~ {hi:>10.0f} {count:>8,} {pct:>5.1f}%")

    # Key decision boundaries
    logger.info(f"\n  Key boundaries for trading:")
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        v = np.percentile(prices, p)
        logger.info(f"    P{p:02d} = {v:>8.1f} 元/MWh")

    # ================================================================
    # PART 2: Daily Curve Clustering
    # ================================================================
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# PART 2: DAILY PRICE CURVE CLUSTERING")
    logger.info(f"{'#' * 70}")

    # Reshape into daily curves (n_days, 96)
    price_matrix = prices[:n_days * 96].reshape(n_days, 96)

    # Normalize each day: relative to daily mean (shape-focused, not level-focused)
    daily_means = price_matrix.mean(axis=1, keepdims=True)
    daily_stds = price_matrix.std(axis=1, keepdims=True)
    daily_stds = np.maximum(daily_stds, 1.0)  # avoid division by zero
    price_shapes = (price_matrix - daily_means) / daily_stds

    # Also compute daily stats as features
    daily_stats = pd.DataFrame({
        "mean": price_matrix.mean(axis=1),
        "std": price_matrix.std(axis=1),
        "max": price_matrix.max(axis=1),
        "min": price_matrix.min(axis=1),
        "range": price_matrix.max(axis=1) - price_matrix.min(axis=1),
        "skew": pd.DataFrame(price_matrix.T).skew().values,
    })

    # Try K=4,5,6,7,8 clusters, pick best by inertia drop
    logger.info(f"\nClustering {n_days} daily curves (shape-normalized):")
    inertias = []
    for k in range(3, 10):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(price_shapes)
        inertias.append((k, km.inertia_))
        logger.info(f"  K={k}: inertia={km.inertia_:,.0f}")

    # Use K=6 as a reasonable default (can tune later)
    n_clusters = 6
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(price_shapes)

    logger.info(f"\nUsing K={n_clusters} clusters:")
    logger.info(f"  {'Cluster':>8} {'Days':>6} {'Pct':>6} {'Avg Price':>10} {'Avg Range':>10} {'Avg Std':>8} {'Label'}")
    logger.info(f"  {'-'*8} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*20}")

    # Characterize each cluster
    cluster_info = []
    for c in range(n_clusters):
        mask = labels == c
        n = mask.sum()
        avg_price = price_matrix[mask].mean()
        avg_range = (price_matrix[mask].max(axis=1) - price_matrix[mask].min(axis=1)).mean()
        avg_std = price_matrix[mask].std(axis=1).mean()

        # Find the shape: where are peaks and valleys?
        centroid = km.cluster_centers_[c]
        peak_hour = centroid.argmax() / 4  # convert to hours
        valley_hour = centroid.argmin() / 4

        # Label the pattern
        if avg_std < daily_stats["std"].quantile(0.25):
            pattern = "FLAT (low volatility)"
        elif avg_range > daily_stats["range"].quantile(0.75):
            if centroid[32:48].mean() < centroid[:16].mean():  # midday dip
                pattern = "DUCK CURVE (solar dip)"
            else:
                pattern = "HIGH VOLATILITY"
        elif peak_hour > 16:
            pattern = "EVENING PEAK"
        elif peak_hour < 10:
            pattern = "MORNING PEAK"
        elif valley_hour > 20 or valley_hour < 6:
            pattern = "NIGHT VALLEY"
        else:
            pattern = "MIXED"

        cluster_info.append({
            "cluster": c, "n_days": n, "avg_price": avg_price,
            "avg_range": avg_range, "avg_std": avg_std,
            "peak_hour": peak_hour, "valley_hour": valley_hour,
            "pattern": pattern,
        })
        logger.info(f"  {c:>8} {n:>6} {n/n_days*100:>5.1f}% {avg_price:>10.0f} {avg_range:>10.0f} {avg_std:>8.0f}  {pattern}")

    # Show centroid shapes (text sparkline)
    logger.info(f"\n  Cluster centroid shapes (normalized, hourly avg):")
    for c in range(n_clusters):
        centroid = km.cluster_centers_[c]
        # Downsample 96 → 24 hourly
        hourly = centroid.reshape(24, 4).mean(axis=1)
        # Text sparkline
        spark_chars = "▁▂▃▄▅▆▇█"
        mn, mx = hourly.min(), hourly.max()
        if mx - mn < 0.01:
            sparkline = "▄" * 24
        else:
            sparkline = "".join(spark_chars[min(int((v - mn) / (mx - mn) * 7.99), 7)] for v in hourly)
        n = cluster_info[c]["n_days"]
        logger.info(f"  C{c} ({n:>4}d): {sparkline}  {cluster_info[c]['pattern']}")

    # ================================================================
    # PART 3: Weather/Load vs Regime Correlation
    # ================================================================
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# PART 3: WHAT DRIVES EACH REGIME?")
    logger.info(f"{'#' * 70}")

    # Build daily feature matrix
    feature_cols = [
        "load_norm", "renewable_penetration", "wind_ratio", "solar_ratio",
        "net_load_norm", "temperature_norm",
    ]
    avail_cols = [c for c in feature_cols if c in df.columns]

    daily_features = {}
    for col in avail_cols:
        vals = df[col].fillna(0).values[:n_days * 96].reshape(n_days, 96)
        daily_features[f"{col}_mean"] = vals.mean(axis=1)
        daily_features[f"{col}_max"] = vals.max(axis=1)
        daily_features[f"{col}_std"] = vals.std(axis=1)

    # Add time features
    day_indices = np.arange(n_days)
    dates = df.index[:n_days * 96:96]
    if hasattr(dates, 'weekday'):
        daily_features["weekday"] = dates.weekday.values
        daily_features["month"] = dates.month.values
        daily_features["is_weekend"] = (dates.weekday.values >= 5).astype(float)

    df_daily = pd.DataFrame(daily_features, index=range(n_days))
    df_daily["cluster"] = labels
    df_daily["price_mean"] = price_matrix.mean(axis=1)
    df_daily["price_range"] = price_matrix.max(axis=1) - price_matrix.min(axis=1)

    # Show avg features per cluster
    logger.info(f"\n  Average features by cluster:")
    display_cols = [c for c in [
        "price_mean", "price_range",
        "load_norm_mean", "renewable_penetration_mean",
        "wind_ratio_mean", "solar_ratio_mean",
        "temperature_norm_mean", "is_weekend",
    ] if c in df_daily.columns]

    header = f"  {'Cluster':>8}"
    for col in display_cols:
        short = col.replace("_mean", "").replace("_norm", "")[:10]
        header += f" {short:>10}"
    logger.info(header)
    logger.info(f"  {'-' * (8 + 11 * len(display_cols))}")

    for c in range(n_clusters):
        row = f"  C{c:>6}"
        for col in display_cols:
            val = df_daily[df_daily["cluster"] == c][col].mean()
            row += f" {val:>10.2f}"
        row += f"  ({cluster_info[c]['pattern']})"
        logger.info(row)

    # ================================================================
    # PART 4: Zone-Based Ideal Revenue (Upper Bound)
    # ================================================================
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# PART 4: ZONE-BASED STRATEGY POTENTIAL")
    logger.info(f"{'#' * 70}")

    # For each cluster, compute the "ideal zone strategy":
    # Within each cluster, what's the optimal fixed schedule?
    from oracle.lp_oracle import solve_day
    from config import BatteryConfig

    battery = BatteryConfig()

    # Oracle revenue (per-day optimal)
    oracle_revs = []
    for d in range(n_days):
        day_prices = price_matrix[d]
        if np.isnan(day_prices).any():
            oracle_revs.append(0)
            continue
        r = solve_day(day_prices, battery, init_soc=0.5)
        oracle_revs.append(r["revenue"])
    oracle_revs = np.array(oracle_revs)

    # Cluster-template strategy: for each cluster, use the AVERAGE price curve
    # to generate a fixed schedule, then apply it to all days in that cluster
    template_revs = np.zeros(n_days)
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() == 0:
            continue

        # Average price curve for this cluster (unnormalized)
        avg_curve = price_matrix[mask].mean(axis=0)
        # Solve optimal schedule for the average curve
        template_result = solve_day(avg_curve, battery, init_soc=0.5)
        template_schedule = template_result["net_power"][:96]

        # Apply this fixed schedule to all days in this cluster
        for d in np.where(mask)[0]:
            day_prices = price_matrix[d]
            # Simulate with template schedule
            from forecast.mpc_controller import _step_battery
            soc = 0.5
            rev = 0.0
            for t in range(96):
                power = float(template_schedule[t])
                soc, net_rev, _ = _step_battery(power, day_prices[t], soc, battery)
                rev += net_rev
            template_revs[d] = rev

    # Compare
    total_oracle = oracle_revs.sum()
    total_template = template_revs.sum()
    capture_rate = total_template / total_oracle * 100

    logger.info(f"\n  If we PERFECTLY classify the day pattern, then use cluster-average schedule:")
    logger.info(f"    Oracle (per-day optimal):    {total_oracle:>14,.0f} ({total_oracle/n_days:>8,.0f}/day)")
    logger.info(f"    Template (cluster-average):  {total_template:>14,.0f} ({total_template/n_days:>8,.0f}/day)")
    logger.info(f"    Capture rate:                {capture_rate:.1f}%")
    logger.info(f"")

    # Per-cluster breakdown
    logger.info(f"  Per-cluster capture rates:")
    logger.info(f"  {'Cluster':>8} {'Days':>6} {'Oracle/day':>12} {'Template/day':>12} {'Capture':>8}")
    for c in range(n_clusters):
        mask = labels == c
        n = mask.sum()
        if n == 0:
            continue
        o_avg = oracle_revs[mask].mean()
        t_avg = template_revs[mask].mean()
        cap = t_avg / o_avg * 100 if o_avg > 0 else 0
        logger.info(f"  C{c:>6} {n:>6} {o_avg:>12,.0f} {t_avg:>12,.0f} {cap:>7.1f}%")

    # ================================================================
    # PART 5: How Hard Is Pattern Classification?
    # ================================================================
    logger.info(f"\n{'#' * 70}")
    logger.info(f"# PART 5: PATTERN CLASSIFICATION DIFFICULTY")
    logger.info(f"{'#' * 70}")

    # Can we predict today's cluster from yesterday's features + morning data?
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    # Features: yesterday's stats + today's morning data (first 16 steps = 4 hours)
    X_list = []
    y_list = []

    for d in range(2, n_days):
        feat = {}
        # Yesterday's features
        feat["y_price_mean"] = price_matrix[d-1].mean()
        feat["y_price_std"] = price_matrix[d-1].std()
        feat["y_price_range"] = price_matrix[d-1].max() - price_matrix[d-1].min()
        feat["y_cluster"] = labels[d-1]

        # Today's early morning (first 4 hours = 16 steps, available by 4am)
        morning = price_matrix[d, :16]
        feat["morning_mean"] = morning.mean()
        feat["morning_std"] = morning.std()
        feat["morning_trend"] = morning[-1] - morning[0]

        # Day-level features
        for col in avail_cols:
            if f"{col}_mean" in df_daily.columns:
                feat[f"y_{col}_mean"] = df_daily.loc[d-1, f"{col}_mean"]

        if "weekday" in df_daily.columns:
            feat["weekday"] = df_daily.loc[d, "weekday"]
            feat["is_weekend"] = df_daily.loc[d, "is_weekend"]
        if "month" in df_daily.columns:
            feat["month"] = df_daily.loc[d, "month"]

        X_list.append(feat)
        y_list.append(labels[d])

    X_df = pd.DataFrame(X_list)
    y = np.array(y_list)

    # Train/test split: last 365 days as test
    split = len(X_df) - 365
    X_train, X_test = X_df.iloc[:split], X_df.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"\n  Pattern classification (predict today's cluster from yesterday + morning):")
    logger.info(f"    Train: {split} days, Test: {len(y_test)} days")
    logger.info(f"    Accuracy: {acc:.1%} ({(y_pred == y_test).sum()}/{len(y_test)})")
    logger.info(f"")

    # Confusion: how often does misclassification hurt?
    # If predicted cluster is "close" to actual, the strategy still works ok
    correct_rev = 0.0
    predicted_rev = 0.0
    test_start_day = n_days - 365

    for i in range(len(y_test)):
        d = test_start_day + i
        actual_c = y_test[i]
        pred_c = y_pred[i]

        # Revenue with correct template
        correct_rev += template_revs[d]  # already computed above

        # Revenue with predicted template
        avg_curve = price_matrix[labels == pred_c].mean(axis=0)
        pred_result = solve_day(avg_curve, battery, init_soc=0.5)
        pred_schedule = pred_result["net_power"][:96]

        from forecast.mpc_controller import _step_battery
        soc = 0.5
        rev = 0.0
        for t in range(96):
            soc, net_rev, _ = _step_battery(float(pred_schedule[t]), price_matrix[d, t], soc, battery)
            rev += net_rev
        predicted_rev += rev

    test_oracle = oracle_revs[test_start_day:].sum()
    logger.info(f"  Revenue impact on test set (365 days):")
    logger.info(f"    Oracle (per-day optimal): {test_oracle:>14,.0f}")
    logger.info(f"    Perfect classification:   {correct_rev:>14,.0f}  ({correct_rev/test_oracle*100:.1f}% capture)")
    logger.info(f"    Predicted classification: {predicted_rev:>14,.0f}  ({predicted_rev/test_oracle*100:.1f}% capture)")
    logger.info(f"    Classification cost:      {(correct_rev-predicted_rev):>14,.0f}  ({(correct_rev-predicted_rev)/correct_rev*100:.1f}% loss from misclassification)")

    # Feature importances
    logger.info(f"\n  Top features for pattern classification:")
    importances = sorted(zip(X_df.columns, clf.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importances[:10]:
        logger.info(f"    {feat:<30} {imp:.3f}")


if __name__ == "__main__":
    main()
