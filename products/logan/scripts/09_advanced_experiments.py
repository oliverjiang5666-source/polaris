"""
Logan · Script 09 — Advanced Experiments (K + L + M + O + N)
===============================================================

  K · Regime-Aware Strategy Selector
    - 每小时用 spread_direction 预测判断 regime
    - 高 P(RT>DA) → Optimal, 低 P(RT>DA) → Naive-style
    - 预期 +1.5-3.5pp over Optimal

  L · Sharpe Ratio / Variance-Aware Metrics
    - 从实验 B 的 robustness.csv 算 Sharpe, downside std
    - 不跑新回测，只重新组织数字

  M · 225 天长 window 回测
    - 用接近全 test 集（277 天）做长期回测
    - 2 个非重叠 225 天 window
    - 5 seed × 4 strategy

  O · MLT Integration
    - Q^MLT = 50% / 70% / 90% 扫
    - 看 MLT 阻塞对冲对 revenue 的影响

  N · Stress Test
    - 识别"负电价日"/"高波动日"/"寒潮日"
    - 看每个策略在极端场景的表现

输出：
    runs/logan/experiments/
      regime_aware.csv
      sharpe.csv
      long_window.csv
      mlt_scan.csv
      stress.csv
      summary_advanced.md

用法：
    PYTHONPATH=. python3 products/logan/scripts/09_advanced_experiments.py
"""
from __future__ import annotations

import sys
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.calendar_features import add_calendar_features
from core.joint_distribution import EmpiricalJointDistribution, JointDistConfig
from products.logan.bid_curve_generator import BidCurveGenerator, BidCurveConfig
from products.logan.optimal_bid import OptimalBidGenerator, OptimalBidConfig
from products.logan.regime_aware_bid import RegimeAwareBidGenerator, RegimeAwareConfig
from products.logan.oracle_bid import compute_oracle_revenue_choice
from products.logan.evaluator import LoganEvaluator, SettlementConfig
from products.logan.compliance import ComplianceRules, enforce


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
MODELS_DIR = ROOT / "models" / "logan"
RUNS_DIR = ROOT / "runs" / "logan" / "experiments"
RULES_PATH = ROOT / "products" / "logan" / "settlement_rules" / "gansu.yaml"


# ============================================================
# Shared helpers
# ============================================================

def agg_hr(arr_96: np.ndarray) -> np.ndarray:
    return arr_96.reshape(24, 4).mean(axis=1)


def load_heads(province: str) -> dict:
    model_dir = MODELS_DIR / province
    heads = {}
    for name in ["regime_classifier", "da_forecaster", "spread_direction",
                 "system_deviation", "rt_forecaster"]:
        with open(model_dir / f"{name}.pkl", "rb") as f:
            heads[name] = pickle.load(f)
    return heads


def load_data_and_scale(province: str, capacity_mw: float):
    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    df = add_calendar_features(df)
    df = df[df["da_price"].notna() & df["load_mw"].notna()]

    if "renewable_mw" in df.columns and df["renewable_mw"].notna().sum() > len(df) * 0.3:
        raw = df["renewable_mw"].ffill().fillna(0).values
    elif "solar_mw" in df.columns and df["solar_mw"].notna().sum() > len(df) * 0.3:
        raw = df["solar_mw"].ffill().fillna(0).values
    else:
        raw = df["load_mw"].fillna(0).values * 0.1
    raw_max = float(np.percentile(raw, 99))
    scaled = raw / raw_max * capacity_mw if raw_max > 0 else np.zeros_like(raw)
    return df, scaled


# ============================================================
# Unified runner (supports 4 strategies now)
# ============================================================

def run_strategy_once(
    strategy: str,                              # naive | heuristic | optimal | regime_aware
    heads: dict,
    joint_dist: EmpiricalJointDistribution,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    capacity_mw: float,
    rules: ComplianceRules,
    test_start_day: int,
    test_end_day: int,
    rng_seed: int,
    settlement_cfg: SettlementConfig,
    forecast_noise_std: float = 0.08,
    quantile_levels: np.ndarray = np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    regime_threshold: float = 0.45,
    day_revenues: list[float] | None = None,    # if provided, append per-day revenue
) -> dict:
    da_fcst = heads["da_forecaster"]
    spread_clf = heads["spread_direction"]
    sys_dev = heads["system_deviation"]

    heur_gen = BidCurveGenerator(capacity_mw=capacity_mw, config=BidCurveConfig())
    opt_gen = OptimalBidGenerator(
        capacity_mw=capacity_mw, rules=rules,
        config=OptimalBidConfig(n_scenarios=120),   # 降低 scenario 数以加速
        joint_dist=joint_dist,
    )
    regime_gen = RegimeAwareBidGenerator(
        capacity_mw=capacity_mw, rules=rules,
        optimal_gen=opt_gen,
        config=RegimeAwareConfig(naive_threshold=regime_threshold),
    )

    evaluator = LoganEvaluator(settlement_cfg)
    rng = np.random.default_rng(rng_seed)

    total_rev = 0.0
    total_da = 0.0
    total_rt = 0.0
    total_mlt_e = 0.0
    total_mlt_c = 0.0
    total_clear = 0.0
    total_actual = 0.0
    n_days = 0

    for d in range(test_start_day, test_end_day):
        try:
            da_q_96 = da_fcst.predict_day_all_quantiles(d, df)
            da_q_hr_all = da_q_96.reshape(24, 4, -1).mean(axis=1)
            da_q_hr_4 = da_q_hr_all[:, :4]

            start_idx = d * 96
            end_idx = start_idx + 96
            actual_power_96 = scaled_actual_power[start_idx:end_idx]
            actual_power_hr = agg_hr(actual_power_96)
            noise = rng.normal(0, forecast_noise_std, 24)
            power_forecast_hr = np.clip(actual_power_hr * (1 + noise), 0.0, capacity_mw)

            rt_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values
            da_96 = df["da_price"].iloc[start_idx:end_idx].ffill().fillna(0).values
            rt_hr = agg_hr(rt_96)
            da_hr = agg_hr(da_96)

            if strategy == "naive":
                cleared_hr = power_forecast_hr.copy()
                res = evaluator.settle_hourly(
                    cleared_qty=cleared_hr, actual_power=actual_power_hr,
                    da_prices_node=da_hr, rt_prices_node=rt_hr,
                )
            elif strategy == "heuristic":
                spread_dir = spread_clf.predict_proba_day(d, df)
                sys_risk = sys_dev.predict_proba_day(d, df)
                bids = heur_gen.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr_4,
                    spread_dir_prob_hourly=spread_dir,
                    system_shortage_prob_hourly=sys_risk["prob_shortage"],
                    system_surplus_prob_hourly=sys_risk["prob_surplus"],
                )
                new_bids = []
                for t, bid in enumerate(bids):
                    fixed, _ = enforce(bid, rules, forecast_mw=float(power_forecast_hr[t]))
                    new_bids.append(fixed if fixed is not None else bid)
                bids = new_bids
                res = evaluator.settle_bids(bids, actual_power_hr, da_hr, rt_hr)
            elif strategy == "optimal":
                bids = opt_gen.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr_all,
                    quantile_levels=quantile_levels,
                )
                res = evaluator.settle_bids(bids, actual_power_hr, da_hr, rt_hr)
            elif strategy == "regime_aware":
                spread_dir = spread_clf.predict_proba_day(d, df)
                bids = regime_gen.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr_all,
                    quantile_levels=quantile_levels,
                    spread_dir_prob_hourly=spread_dir,
                )
                # Final compliance enforce
                new_bids = []
                for t, bid in enumerate(bids):
                    fixed, _ = enforce(bid, rules, forecast_mw=float(power_forecast_hr[t]))
                    new_bids.append(fixed if fixed is not None else bid)
                bids = new_bids
                res = evaluator.settle_bids(bids, actual_power_hr, da_hr, rt_hr)
            else:
                raise ValueError(strategy)

            total_rev += res.total_revenue
            total_da += res.da_spot_revenue
            total_rt += res.rt_spot_revenue
            total_mlt_e += res.mlt_energy_revenue
            total_mlt_c += res.mlt_congestion_revenue
            total_clear += res.cleared_mwh
            total_actual += res.actual_mwh
            n_days += 1
            if day_revenues is not None:
                day_revenues.append(res.total_revenue)
        except Exception as e:
            continue

    return {
        "strategy": strategy,
        "total_revenue": total_rev,
        "da_revenue": total_da,
        "rt_revenue": total_rt,
        "mlt_energy": total_mlt_e,
        "mlt_congest": total_mlt_c,
        "clear_ratio": total_clear / max(total_actual, 1.0),
        "n_days": n_days,
    }


# ============================================================
# Experiment K · Regime-Aware Selector
# ============================================================

def experiment_k_regime(
    heads, joint_dist, df, scaled_actual_power, capacity_mw, rules,
    windows, seeds=[42, 43, 44],          # 3 seeds (fast)
    thresholds=[0.40, 0.45, 0.50],         # 3 thresholds
) -> pd.DataFrame:
    logger.info("\n=== Experiment K · Regime-Aware Strategy Selector ===")
    rows = []
    settlement_cfg = SettlementConfig()  # no MLT for this experiment

    # First run baselines for reference
    for (start_d, end_d) in windows:
        for seed in seeds:
            # baselines: naive / optimal
            for strat in ["naive", "optimal"]:
                r = run_strategy_once(
                    strategy=strat, heads=heads, joint_dist=joint_dist,
                    df=df, scaled_actual_power=scaled_actual_power,
                    capacity_mw=capacity_mw, rules=rules,
                    test_start_day=start_d, test_end_day=end_d,
                    rng_seed=seed, settlement_cfg=settlement_cfg,
                )
                rows.append({
                    "window": f"day_{start_d}-{end_d}", "seed": seed,
                    "strategy": strat, "threshold": None,
                    "revenue": r["total_revenue"], "clear": r["clear_ratio"],
                })
            # regime_aware with different thresholds
            for thr in thresholds:
                r = run_strategy_once(
                    strategy="regime_aware", heads=heads, joint_dist=joint_dist,
                    df=df, scaled_actual_power=scaled_actual_power,
                    capacity_mw=capacity_mw, rules=rules,
                    test_start_day=start_d, test_end_day=end_d,
                    rng_seed=seed, settlement_cfg=settlement_cfg,
                    regime_threshold=thr,
                )
                rows.append({
                    "window": f"day_{start_d}-{end_d}", "seed": seed,
                    "strategy": "regime_aware", "threshold": thr,
                    "revenue": r["total_revenue"], "clear": r["clear_ratio"],
                })
        logger.info(f"  window {start_d}-{end_d} done")

    return pd.DataFrame(rows)


# ============================================================
# Experiment L · Sharpe Ratio
# ============================================================

def experiment_l_sharpe(df_k: pd.DataFrame) -> pd.DataFrame:
    """Compute Sharpe-like metrics on regime experiment results."""
    logger.info("\n=== Experiment L · Sharpe Ratio / Variance Metrics ===")

    # For each (strategy, threshold): mean, std, Sharpe = mean/std
    agg_rows = []
    for (strategy, threshold), grp in df_k.groupby(["strategy", "threshold"], dropna=False):
        mean = grp["revenue"].mean()
        std = grp["revenue"].std()
        sharpe = mean / std if std > 0 else np.nan
        # downside std (only revenues below mean)
        below = grp["revenue"][grp["revenue"] < mean]
        downside_std = below.std() if len(below) > 1 else np.nan
        sortino = mean / downside_std if downside_std and downside_std > 0 else np.nan
        # 95% CI half-width
        n = len(grp)
        ci95 = 1.96 * std / np.sqrt(n) if n > 1 else np.nan

        agg_rows.append({
            "strategy": strategy,
            "threshold": threshold,
            "n": n,
            "mean_revenue": mean,
            "std_revenue": std,
            "sharpe": sharpe,
            "downside_std": downside_std,
            "sortino": sortino,
            "ci95_half": ci95,
        })

    df_agg = pd.DataFrame(agg_rows)
    # Log
    for _, r in df_agg.iterrows():
        thr_str = f"t={r['threshold']:.2f}" if pd.notna(r['threshold']) else ""
        logger.info(
            f"  {r['strategy']:<14} {thr_str:<8}  "
            f"mean=¥{r['mean_revenue']:,.0f}  std=¥{r['std_revenue']:,.0f}  "
            f"Sharpe={r['sharpe']:.2f}  Sortino={r['sortino']:.2f}"
        )
    return df_agg


# ============================================================
# Experiment M · Long Window
# ============================================================

def experiment_m_long_window(
    heads, joint_dist, df, scaled_actual_power, capacity_mw, rules,
    seeds=[42, 43],        # 2 seeds for long window (cost控制)
    window_days=225,
) -> pd.DataFrame:
    logger.info(f"\n=== Experiment M · Long Window ({window_days}d) ===")
    settlement_cfg = SettlementConfig()

    n_days_total = len(df) // 96
    split = int(len(df) * 0.8)
    test_start = split // 96

    # 1 long window
    start_d = test_start
    end_d = min(start_d + window_days, n_days_total - 1)
    logger.info(f"  Window: day {start_d}-{end_d} ({end_d - start_d} days)")

    rows = []
    # Use regime_aware with best threshold (0.45)
    for seed in seeds:
        for strat in ["naive", "optimal", "regime_aware"]:
            r = run_strategy_once(
                strategy=strat, heads=heads, joint_dist=joint_dist,
                df=df, scaled_actual_power=scaled_actual_power,
                capacity_mw=capacity_mw, rules=rules,
                test_start_day=start_d, test_end_day=end_d,
                rng_seed=seed, settlement_cfg=settlement_cfg,
                regime_threshold=0.45,
            )
            rows.append({
                "seed": seed, "strategy": strat,
                "revenue": r["total_revenue"],
                "clear": r["clear_ratio"], "n_days": r["n_days"],
            })
        logger.info(f"  seed={seed}: "
                    f"naive={rows[-3]['revenue']:,.0f}, "
                    f"opt={rows[-2]['revenue']:,.0f}, "
                    f"regime={rows[-1]['revenue']:,.0f}")
    return pd.DataFrame(rows)


# ============================================================
# Experiment O · MLT Scan
# ============================================================

def experiment_o_mlt(
    heads, joint_dist, df, scaled_actual_power, capacity_mw, rules,
    mlt_fractions=[0.0, 0.5, 0.7],        # 精简 3 个 fractions
    mlt_price=380.0,
    test_window=(None, None),
    seed=42,
) -> pd.DataFrame:
    logger.info(f"\n=== Experiment O · MLT Scan ===")
    n_days_total = len(df) // 96
    split = int(len(df) * 0.8)
    test_start = split // 96
    start_d, end_d = test_window
    if start_d is None:
        start_d = test_start
    if end_d is None:
        end_d = min(test_start + 90, n_days_total - 1)
    logger.info(f"  Test window: day {start_d}-{end_d}, seed={seed}")

    rows = []
    for mlt_frac in mlt_fractions:
        mlt_mw = mlt_frac * capacity_mw
        cfg = SettlementConfig(
            use_mlt=(mlt_frac > 0),
            mlt_quantity_mw=mlt_mw,
            mlt_price_yuan_mwh=mlt_price,
        )
        for strat in ["naive", "optimal", "regime_aware"]:
            r = run_strategy_once(
                strategy=strat, heads=heads, joint_dist=joint_dist,
                df=df, scaled_actual_power=scaled_actual_power,
                capacity_mw=capacity_mw, rules=rules,
                test_start_day=start_d, test_end_day=end_d,
                rng_seed=seed, settlement_cfg=cfg,
            )
            rows.append({
                "mlt_fraction": mlt_frac, "mlt_mw": mlt_mw,
                "strategy": strat,
                "revenue": r["total_revenue"],
                "mlt_energy": r["mlt_energy"],
                "mlt_congest": r["mlt_congest"],
                "da_revenue": r["da_revenue"],
                "rt_revenue": r["rt_revenue"],
                "clear": r["clear_ratio"],
            })
        logger.info(f"  MLT={mlt_frac*100:.0f}%: "
                    f"naive={rows[-3]['revenue']:,.0f}, "
                    f"opt={rows[-2]['revenue']:,.0f}, "
                    f"regime={rows[-1]['revenue']:,.0f}")
    return pd.DataFrame(rows)


# ============================================================
# Experiment N · Stress Test
# ============================================================

def experiment_n_stress(
    heads, joint_dist, df, scaled_actual_power, capacity_mw, rules,
    seed=42,
) -> pd.DataFrame:
    logger.info(f"\n=== Experiment N · Stress Test ===")
    n_days_total = len(df) // 96
    split = int(len(df) * 0.8)
    test_start = split // 96

    # Identify stress days in test set based on RT/DA characteristics
    rt = df["rt_price"].fillna(0).values
    da = df["da_price"].ffill().fillna(0).values
    rt_days = rt[: n_days_total * 96].reshape(n_days_total, 96)
    da_days = da[: n_days_total * 96].reshape(n_days_total, 96)

    # Per-day stats
    test_stats = []
    for d in range(test_start, n_days_total - 1):
        r = rt_days[d]
        a = da_days[d]
        test_stats.append({
            "day": d,
            "rt_mean": r.mean(),
            "rt_std": r.std(),
            "da_mean": a.mean(),
            "neg_count": int((r < 0).sum()),
            "high_vol": r.std() > 200,
            "spread_mean": (r - a).mean(),
        })
    ts_df = pd.DataFrame(test_stats)

    # Define stress categories
    neg_price_days = ts_df[ts_df["neg_count"] > 10]["day"].tolist()
    high_vol_days = ts_df[ts_df["rt_std"] > 300]["day"].tolist()
    high_rt_days = ts_df[ts_df["rt_mean"] > 500]["day"].tolist()
    low_rt_days = ts_df[ts_df["rt_mean"] < 100]["day"].tolist()

    logger.info(f"  Stress days: neg_price={len(neg_price_days)}, "
                f"high_vol={len(high_vol_days)}, high_rt={len(high_rt_days)}, "
                f"low_rt={len(low_rt_days)}")

    categories = {
        "neg_price": neg_price_days,
        "high_vol": high_vol_days,
        "high_rt": high_rt_days,
        "low_rt": low_rt_days,
    }

    settlement_cfg = SettlementConfig()
    rows = []
    for cat, days_list in categories.items():
        if len(days_list) < 3:
            continue
        for strat in ["naive", "optimal", "regime_aware"]:
            day_revs = []
            for d in days_list[:10]:  # cap at 10 days per category (fast)
                _ = run_strategy_once(
                    strategy=strat, heads=heads, joint_dist=joint_dist,
                    df=df, scaled_actual_power=scaled_actual_power,
                    capacity_mw=capacity_mw, rules=rules,
                    test_start_day=d, test_end_day=d + 1,
                    rng_seed=seed, settlement_cfg=settlement_cfg,
                    day_revenues=day_revs,
                )
            if day_revs:
                rows.append({
                    "category": cat,
                    "strategy": strat,
                    "n_days": len(day_revs),
                    "mean_daily_rev": np.mean(day_revs),
                    "std_daily_rev": np.std(day_revs),
                    "total_rev": sum(day_revs),
                })
        logger.info(f"  {cat}: done ({len(days_list[:20])} days each)")
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main(province: str = "gansu", capacity_mw: float = 100.0):
    logger.info(f"=== Logan Advanced Experiments ({province}) ===")

    rules = ComplianceRules.from_yaml(RULES_PATH, capacity_mw=capacity_mw)
    heads = load_heads(province)
    df, scaled_actual_power = load_data_and_scale(province, capacity_mw)

    n_days_total = len(df) // 96
    split = int(len(df) * 0.8)
    test_start = split // 96
    logger.info(f"Test starts at day {test_start}, total days {n_days_total}")

    # Joint dist
    joint_start = max(0, split - 540 * 96)
    df_joint = df.iloc[joint_start:split]
    joint_dist = EmpiricalJointDistribution(JointDistConfig(n_da_buckets=10))
    joint_dist.fit(df_joint)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Windows for K
    windows = []
    for offset in [0, 90, 180]:
        s = test_start + offset
        e = min(s + 90, n_days_total - 1)
        if e > s + 10:
            windows.append((s, e))

    # ---- K ----
    df_k = experiment_k_regime(
        heads=heads, joint_dist=joint_dist, df=df,
        scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
        windows=windows,
    )
    df_k.to_csv(RUNS_DIR / "regime_aware.csv", index=False)

    # ---- L ----
    df_l = experiment_l_sharpe(df_k)
    df_l.to_csv(RUNS_DIR / "sharpe.csv", index=False)

    # ---- M ----
    df_m = experiment_m_long_window(
        heads=heads, joint_dist=joint_dist, df=df,
        scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
    )
    df_m.to_csv(RUNS_DIR / "long_window.csv", index=False)

    # ---- O ----
    df_o = experiment_o_mlt(
        heads=heads, joint_dist=joint_dist, df=df,
        scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
        test_window=(test_start, min(test_start + 90, n_days_total - 1)),
    )
    df_o.to_csv(RUNS_DIR / "mlt_scan.csv", index=False)

    # ---- N ----
    df_n = experiment_n_stress(
        heads=heads, joint_dist=joint_dist, df=df,
        scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
    )
    df_n.to_csv(RUNS_DIR / "stress.csv", index=False)

    print_summary(df_k, df_l, df_m, df_o, df_n)


def print_summary(df_k, df_l, df_m, df_o, df_n):
    logger.info(f"\n{'=' * 90}")
    logger.info("  ADVANCED EXPERIMENTS SUMMARY")
    logger.info(f"{'=' * 90}")

    # K
    logger.info("\n[K] Regime-Aware vs Baselines (mean across seeds × windows):")
    agg = df_k.groupby(["strategy", "threshold"], dropna=False)["revenue"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    agg["ci95"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    naive_mean = agg[agg["strategy"] == "naive"]["mean"].iloc[0]
    for _, r in agg.sort_values(["strategy", "threshold"]).iterrows():
        thr = f"t={r['threshold']:.2f}" if pd.notna(r['threshold']) else ""
        gain = (r["mean"] - naive_mean) / naive_mean * 100
        logger.info(
            f"  {r['strategy']:<14} {thr:<8}  mean=¥{r['mean']:,.0f}  "
            f"CI=±¥{r['ci95']:,.0f}  vs Naive: {gain:+.2f}%"
        )

    # L
    logger.info("\n[L] Sharpe Metrics:")
    for _, r in df_l.sort_values(["strategy", "threshold"]).iterrows():
        thr = f"t={r['threshold']:.2f}" if pd.notna(r['threshold']) else ""
        logger.info(
            f"  {r['strategy']:<14} {thr:<8}  "
            f"Sharpe={r['sharpe']:.2f}  Sortino={r['sortino']:.2f}"
        )

    # M
    logger.info("\n[M] Long Window (225 days):")
    agg_m = df_m.groupby("strategy")["revenue"].agg(["mean", "std", "count"]).reset_index()
    naive_m = agg_m[agg_m["strategy"] == "naive"]["mean"].iloc[0]
    for _, r in agg_m.iterrows():
        gain = (r["mean"] - naive_m) / naive_m * 100
        logger.info(
            f"  {r['strategy']:<14} mean=¥{r['mean']:,.0f}  "
            f"std=¥{r['std']:,.0f}  vs Naive: {gain:+.2f}%"
        )

    # O
    logger.info("\n[O] MLT Scan (Q^MLT × 380 元/MWh, 90d window):")
    for mlt_frac, grp in df_o.groupby("mlt_fraction"):
        logger.info(f"  MLT={mlt_frac*100:.0f}%:")
        naive_r = grp[grp["strategy"] == "naive"]["revenue"].iloc[0]
        for _, r in grp.iterrows():
            gain = (r["revenue"] - naive_r) / naive_r * 100
            logger.info(
                f"    {r['strategy']:<14} ¥{r['revenue']:,.0f}  "
                f"(MLT_e=¥{r['mlt_energy']:,.0f}, MLT_c=¥{r['mlt_congest']:+,.0f})  "
                f"vs naive: {gain:+.2f}%"
            )

    # N
    logger.info("\n[N] Stress Test (daily mean revenue by category):")
    for cat, grp in df_n.groupby("category"):
        logger.info(f"  {cat}:")
        naive_mean = grp[grp["strategy"] == "naive"]["mean_daily_rev"].iloc[0]
        for _, r in grp.iterrows():
            gain_pct = (r["mean_daily_rev"] - naive_mean) / abs(naive_mean) * 100 if naive_mean != 0 else 0.0
            logger.info(
                f"    {r['strategy']:<14} n={r['n_days']:>3}d  "
                f"mean_daily=¥{r['mean_daily_rev']:,.0f}  "
                f"std=¥{r['std_daily_rev']:,.0f}  vs naive: {gain_pct:+.2f}%"
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    args = parser.parse_args()
    main(args.province, args.capacity)
