"""
Logan · Script 08 — 综合实验 A + B + D
========================================

在甘肃真实规则下：

  实验 A · Oracle Upper Bound
    - Perfect foresight 下的合规最优 revenue
    - 合规版 + 理想版（不受 min_step 约束）

  实验 B · Robustness
    - 5 个 seed × 3 个 window (early / mid / late 90 days)
    - 对 Naive / Heuristic / Optimal 各跑 15 次
    - 输出 mean + std + 95% CI

  实验 D · Head Ablation (Optimal framework)
    - full: SupplyCurve DA quantiles + Empirical Copula
    - no_copula: DA quantiles but iid RT
    - no_supply_curve: 全局 DA 分位数 + Copula
    - both_off: 全局分位数 + iid RT

输出：
    runs/logan/experiments/
      oracle.csv
      robustness.csv
      ablation.csv
      summary.md

用法：
    PYTHONPATH=. python3 products/logan/scripts/08_experiments.py --days 90
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
from products.logan.oracle_bid import compute_oracle_revenue_choice
from products.logan.evaluator import LoganEvaluator, SettlementConfig
from products.logan.compliance import ComplianceRules, validate, enforce


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
# Global quantile fallback (for ablation)
# ============================================================

class GlobalQuantileFallback:
    """简单 fallback：用训练集所有 DA 价的静态分位数，不依赖当前 features"""

    def __init__(self, df_train: pd.DataFrame, levels: tuple = (0.05, 0.25, 0.5, 0.75, 0.95)):
        self.levels = np.array(levels)
        da = df_train["da_price"].ffill().fillna(0).values
        n_days = len(da) // 96
        da_hr = da[: n_days * 96].reshape(n_days, 24, 4).mean(axis=2)  # (n_days, 24)
        # 每小时独立分位数
        self.quantiles_by_hour = np.quantile(da_hr, self.levels, axis=0).T  # (24, n_levels)

    def predict_day_all_quantiles(self, day_idx, df=None):
        """返回 (96, n_levels) —— 对 24 小时分位数 upsample 到 96 步"""
        # 先 (24, n_levels) → (96, n_levels)（每小时 4 次重复）
        hr = self.quantiles_by_hour  # (24, n_levels)
        return np.repeat(hr, 4, axis=0)  # (96, n_levels)


class IIDCopulaFallback:
    """Fallback joint dist：DA 从 empirical，RT 独立采样"""

    def __init__(self, joint_dist: EmpiricalJointDistribution, rt_mag_std: float = 50.0):
        self.joint = joint_dist
        self.rt_mag_std = rt_mag_std

    def sample(self, hour, da_quantiles, quantile_levels, n_scenarios, rng):
        # DA 用 quantile interp
        u = rng.uniform(0, 1, n_scenarios)
        full_levels = np.concatenate([[0.0], quantile_levels, [1.0]])
        full_vals = np.concatenate([[da_quantiles[0] * 0.5], da_quantiles, [da_quantiles[-1] * 1.5]])
        da_samples = np.interp(u, full_levels, full_vals)
        # RT 独立：从 training RT 分布采样（或简化为 da_samples + 噪声）
        signs = np.where(rng.random(n_scenarios) < 0.5, 1.0, -1.0)
        mags = np.abs(rng.normal(0, self.rt_mag_std, n_scenarios))
        rt_samples = da_samples + signs * mags
        return da_samples, rt_samples


# ============================================================
# Unified backtest runner
# ============================================================

def run_strategy_once(
    strategy: str,
    da_forecaster,
    joint_dist,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    capacity_mw: float,
    rules: ComplianceRules,
    test_start_day: int,
    test_end_day: int,
    rng_seed: int,
    forecast_noise_std: float = 0.08,
    quantile_levels: np.ndarray = np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
) -> dict:
    """
    跑一次 backtest，返回 revenue + stats。
    strategy ∈ {"naive", "heuristic", "optimal"}
    """
    heur_gen = BidCurveGenerator(capacity_mw=capacity_mw, config=BidCurveConfig())
    opt_gen = OptimalBidGenerator(
        capacity_mw=capacity_mw,
        rules=rules,
        config=OptimalBidConfig(n_scenarios=300),
        joint_dist=joint_dist,
    )
    evaluator = LoganEvaluator(SettlementConfig())
    rng = np.random.default_rng(rng_seed)

    # Dummy spread/sys for heuristic (use in heuristic strategy only)
    # Instead of loading separate head, fake with 0.5 to make heuristic path run
    # Actually use the real spread_clf if available
    # Keep it simple: heuristic uses default 0.5 / 0.15 probs

    total_rev = 0.0
    total_da = 0.0
    total_rt = 0.0
    total_clear = 0.0
    total_actual = 0.0
    n_days = 0

    for d in range(test_start_day, test_end_day):
        try:
            da_q_96 = da_forecaster.predict_day_all_quantiles(d, df)  # (96, 5)
            da_q_hr_all = da_q_96.reshape(24, 4, -1).mean(axis=1)     # (24, 5)
            da_q_hr_4 = da_q_hr_all[:, :4]                             # [P05,P25,P50,P75]

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
                    cleared_qty=cleared_hr,
                    actual_power=actual_power_hr,
                    da_prices_node=da_hr, rt_prices_node=rt_hr,
                )
            elif strategy == "heuristic":
                # Simplified: 用均匀 0.5 概率（避免依赖 head）
                spread_dir = np.full(24, 0.5)
                sys_short = np.full(24, 0.15)
                sys_surp = np.full(24, 0.15)
                bids = heur_gen.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr_4,
                    spread_dir_prob_hourly=spread_dir,
                    system_shortage_prob_hourly=sys_short,
                    system_surplus_prob_hourly=sys_surp,
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
            else:
                raise ValueError(strategy)

            total_rev += res.total_revenue
            total_da += res.da_spot_revenue
            total_rt += res.rt_spot_revenue
            total_clear += res.cleared_mwh
            total_actual += res.actual_mwh
            n_days += 1
        except Exception as e:
            continue

    return {
        "strategy": strategy,
        "total_revenue": total_rev,
        "da_revenue": total_da,
        "rt_revenue": total_rt,
        "clear_ratio": total_clear / max(total_actual, 1.0),
        "n_days": n_days,
    }


# ============================================================
# Experiment A · Oracle
# ============================================================

def experiment_a_oracle(
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    capacity_mw: float,
    rules: ComplianceRules,
    test_windows: list[tuple[int, int]],
    forecast_noise_std: float = 0.08,
    rng_seed: int = 42,
) -> pd.DataFrame:
    logger.info("\n=== Experiment A · Oracle Upper Bound ===")
    rows = []
    rng = np.random.default_rng(rng_seed)

    for (start_d, end_d) in test_windows:
        # Build full-window arrays
        all_actual = []
        all_forecast = []
        all_da = []
        all_rt = []
        for d in range(start_d, end_d):
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

            all_actual.append(actual_power_hr)
            all_forecast.append(power_forecast_hr)
            all_da.append(da_hr)
            all_rt.append(rt_hr)

        actual = np.concatenate(all_actual)
        forecast = np.concatenate(all_forecast)
        da = np.concatenate(all_da)
        rt = np.concatenate(all_rt)

        res = compute_oracle_revenue_choice(
            actual_power_hr=actual,
            actual_da_hr=da,
            actual_rt_hr=rt,
            forecast_hr=forecast,
            capacity_mw=capacity_mw,
            min_step_fraction=rules.min_step_fraction_of_capacity,
        )
        rows.append({
            "window": f"day_{start_d}-{end_d}",
            "compliance_revenue": res["compliance"].revenue,
            "compliance_clear_ratio": res["compliance"].clear_ratio,
            "unconstrained_revenue": res["unconstrained"].revenue,
            "unconstrained_clear_ratio": res["unconstrained"].clear_ratio,
            "compliance_tax": res["compliance_tax"],
            "compliance_da": res["compliance"].da_revenue,
            "compliance_rt": res["compliance"].rt_revenue,
        })
        logger.info(
            f"  {start_d}-{end_d}: compliance=¥{res['compliance'].revenue:,.0f} "
            f"(clear {res['compliance'].clear_ratio*100:.1f}%), "
            f"unconstrained=¥{res['unconstrained'].revenue:,.0f} "
            f"(clear {res['unconstrained'].clear_ratio*100:.1f}%), "
            f"tax=¥{res['compliance_tax']:,.0f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# Experiment B · Robustness
# ============================================================

def experiment_b_robustness(
    heads: dict,
    joint_dist,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    capacity_mw: float,
    rules: ComplianceRules,
    seeds: list[int],
    windows: list[tuple[int, int]],
) -> pd.DataFrame:
    logger.info("\n=== Experiment B · Robustness ===")
    da_fcst = heads["da_forecaster"]

    rows = []
    for (start_d, end_d) in windows:
        for seed in seeds:
            for strategy in ["naive", "heuristic", "optimal"]:
                t0 = time.time()
                r = run_strategy_once(
                    strategy=strategy,
                    da_forecaster=da_fcst,
                    joint_dist=joint_dist,
                    df=df,
                    scaled_actual_power=scaled_actual_power,
                    capacity_mw=capacity_mw,
                    rules=rules,
                    test_start_day=start_d,
                    test_end_day=end_d,
                    rng_seed=seed,
                )
                elapsed = time.time() - t0
                rows.append({
                    "window": f"day_{start_d}-{end_d}",
                    "seed": seed,
                    "strategy": strategy,
                    "revenue": r["total_revenue"],
                    "clear_ratio": r["clear_ratio"],
                    "n_days": r["n_days"],
                    "elapsed_s": elapsed,
                })
            logger.info(
                f"  window {start_d}-{end_d}, seed={seed}: "
                f"naive={rows[-3]['revenue']:,.0f}, "
                f"heur={rows[-2]['revenue']:,.0f}, "
                f"opt={rows[-1]['revenue']:,.0f}"
            )

    return pd.DataFrame(rows)


# ============================================================
# Experiment D · Head Ablation
# ============================================================

def experiment_d_ablation(
    heads: dict,
    joint_dist,
    iid_fallback,
    da_global_fallback,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    capacity_mw: float,
    rules: ComplianceRules,
    test_start_day: int,
    test_end_day: int,
    rng_seed: int = 42,
) -> pd.DataFrame:
    logger.info("\n=== Experiment D · Head Ablation (Optimal framework) ===")
    da_trained = heads["da_forecaster"]

    variants = [
        ("full",              da_trained,            joint_dist),
        ("no_copula",         da_trained,            iid_fallback),
        ("no_supply_curve",   da_global_fallback,    joint_dist),
        ("both_off",          da_global_fallback,    iid_fallback),
    ]

    rows = []
    for name, da_model, jd_model in variants:
        r = run_strategy_once(
            strategy="optimal",
            da_forecaster=da_model,
            joint_dist=jd_model,
            df=df,
            scaled_actual_power=scaled_actual_power,
            capacity_mw=capacity_mw,
            rules=rules,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
            rng_seed=rng_seed,
        )
        rows.append({
            "variant": name,
            "revenue": r["total_revenue"],
            "clear_ratio": r["clear_ratio"],
            "da_revenue": r["da_revenue"],
            "rt_revenue": r["rt_revenue"],
        })
        logger.info(
            f"  {name:<18} revenue=¥{r['total_revenue']:,.0f}  "
            f"clear={r['clear_ratio']*100:.1f}%  "
            f"DA=¥{r['da_revenue']:,.0f}  RT=¥{r['rt_revenue']:+,.0f}"
        )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main(province: str = "gansu", capacity_mw: float = 100.0, days: int = 90):
    logger.info(f"=== Logan Comprehensive Experiments ({province}, {days} days/window) ===")

    rules = ComplianceRules.from_yaml(RULES_PATH, capacity_mw=capacity_mw)
    logger.info(f"Rules: {rules.min_steps}-{rules.max_steps} 段, 每段 ≥ {rules.min_step_mw:.1f} MW")

    heads = load_heads(province)
    df, scaled_actual_power = load_data_and_scale(province, capacity_mw)
    n_days_total = len(df) // 96
    split = int(len(df) * 0.8)
    test_start = split // 96
    logger.info(f"Total days: {n_days_total}, test starts at day {test_start}")

    # Test windows (3 个 non-overlap window)
    windows = []
    for offset in [0, days, 2 * days]:
        s = test_start + offset
        e = min(s + days, n_days_total - 1)
        if e > s + 10:
            windows.append((s, e))
    logger.info(f"Windows: {windows}")

    # Joint distribution（recent 540d 训练）
    joint_start = max(0, split - 540 * 96)
    df_joint = df.iloc[joint_start:split]
    joint_dist = EmpiricalJointDistribution(JointDistConfig(n_da_buckets=10))
    joint_dist.fit(df_joint)

    # IID copula fallback
    iid_fallback = IIDCopulaFallback(joint_dist, rt_mag_std=100.0)

    # Global quantile DA fallback
    da_global = GlobalQuantileFallback(df_joint)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Run A ----
    df_a = experiment_a_oracle(
        df=df, scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
        test_windows=windows,
    )
    df_a.to_csv(RUNS_DIR / "oracle.csv", index=False)

    # ---- Run B ----
    seeds = [42, 43, 44, 45, 46]
    df_b = experiment_b_robustness(
        heads=heads, joint_dist=joint_dist,
        df=df, scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
        seeds=seeds, windows=windows,
    )
    df_b.to_csv(RUNS_DIR / "robustness.csv", index=False)

    # ---- Run D ----
    # Use first window for ablation (fast)
    start_d, end_d = windows[0]
    df_d = experiment_d_ablation(
        heads=heads, joint_dist=joint_dist,
        iid_fallback=iid_fallback, da_global_fallback=da_global,
        df=df, scaled_actual_power=scaled_actual_power,
        capacity_mw=capacity_mw, rules=rules,
        test_start_day=start_d, test_end_day=end_d,
    )
    df_d.to_csv(RUNS_DIR / "ablation.csv", index=False)

    # ---- Summary ----
    print_summary(df_a, df_b, df_d)


def print_summary(df_a: pd.DataFrame, df_b: pd.DataFrame, df_d: pd.DataFrame):
    logger.info(f"\n{'=' * 80}")
    logger.info("  COMPREHENSIVE SUMMARY")
    logger.info(f"{'=' * 80}")

    # A: Oracle
    logger.info("\n[A] Oracle Upper Bound (per window):")
    for _, row in df_a.iterrows():
        logger.info(
            f"  {row['window']:<18} "
            f"compliance=¥{row['compliance_revenue']:,.0f}  "
            f"unconstrained=¥{row['unconstrained_revenue']:,.0f}  "
            f"合规成本=¥{row['compliance_tax']:,.0f}"
        )

    # B: Robustness
    logger.info("\n[B] Robustness (mean ± std across seeds × windows):")
    agg = df_b.groupby("strategy")["revenue"].agg(["mean", "std", "count"]).reset_index()
    # 95% CI assuming normal
    agg["ci95_half"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    for _, row in agg.iterrows():
        logger.info(
            f"  {row['strategy']:<12} "
            f"mean=¥{row['mean']:,.0f}  "
            f"std=¥{row['std']:,.0f}  "
            f"95% CI=±¥{row['ci95_half']:,.0f}  "
            f"(n={int(row['count'])})"
        )

    # Gain vs naive
    naive_mean = agg[agg["strategy"] == "naive"]["mean"].iloc[0]
    logger.info("\n[B'] Gain vs Naive (mean):")
    for _, row in agg.iterrows():
        if row["strategy"] == "naive":
            continue
        gain = row["mean"] - naive_mean
        pct = gain / naive_mean * 100
        logger.info(
            f"  {row['strategy']:<12} gain=¥{gain:+,.0f} ({pct:+.2f}%)"
        )

    # D: Ablation
    logger.info("\n[D] Head Ablation (Optimal, first window):")
    full_rev = df_d[df_d["variant"] == "full"]["revenue"].iloc[0]
    for _, row in df_d.iterrows():
        delta = row["revenue"] - full_rev
        pct = delta / full_rev * 100 if full_rev else 0.0
        logger.info(
            f"  {row['variant']:<18} "
            f"revenue=¥{row['revenue']:,.0f}  "
            f"vs full: {delta:+,.0f} ({pct:+.2f}%)  "
            f"clear={row['clear_ratio']*100:.1f}%"
        )

    # Anchor: Heuristic vs Oracle
    oracle_comp_mean = df_a["compliance_revenue"].mean()
    heur_mean = agg[agg["strategy"] == "heuristic"]["mean"].iloc[0]
    opt_mean = agg[agg["strategy"] == "optimal"]["mean"].iloc[0]
    logger.info("\n[Anchor] Optimality Gap:")
    logger.info(f"  Oracle (compliance):    ¥{oracle_comp_mean:,.0f}")
    logger.info(f"  Optimal (Logan):        ¥{opt_mean:,.0f}  "
                f"({opt_mean/oracle_comp_mean*100:.1f}% of Oracle)")
    logger.info(f"  Heuristic:              ¥{heur_mean:,.0f}  "
                f"({heur_mean/oracle_comp_mean*100:.1f}% of Oracle)")
    logger.info(f"  Naive:                  ¥{naive_mean:,.0f}  "
                f"({naive_mean/oracle_comp_mean*100:.1f}% of Oracle)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    main(args.province, args.capacity, args.days)
