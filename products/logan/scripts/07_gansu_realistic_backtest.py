"""
Logan · Script 07 — Gansu Realistic Backtest (with compliance)
================================================================

在甘肃真实规则下做端到端回测：
  1. 加载 settlement_rules/gansu.yaml
  2. 训好的四个 head 继续用
  3. 三种策略对比：
     - Naive: 报 0 元保中标（cleared = forecast）
     - Heuristic: BidCurveGenerator + enforce compliance
     - Optimal: OptimalBidGenerator (breakpoint search + LP)
  4. 所有 bid 经过 compliance.enforce 验证
  5. evaluator 按甘肃真实 4 段结算

⚠️ 之前 +9.17% 的数字是在虚构规则下的。本脚本给出真实规则下的诚实数字。

用法：
    PYTHONPATH=. python3 products/logan/scripts/07_gansu_realistic_backtest.py --days 90
"""
from __future__ import annotations

import sys
import pickle
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
from products.logan.evaluator import LoganEvaluator, SettlementConfig
from products.logan.compliance import ComplianceRules, validate, enforce


PROCESSED_DIR = ROOT / "data" / "china" / "processed"
MODELS_DIR = ROOT / "models" / "logan"
RUNS_DIR = ROOT / "runs" / "logan"
RULES_PATH = ROOT / "products" / "logan" / "settlement_rules" / "gansu.yaml"


def aggregate_15min_to_hour(arr_96: np.ndarray) -> np.ndarray:
    return arr_96.reshape(24, 4).mean(axis=1)


def load_heads(province: str) -> dict:
    model_dir = MODELS_DIR / province
    heads = {}
    for name in ["regime_classifier", "da_forecaster", "spread_direction",
                 "system_deviation", "rt_forecaster"]:
        with open(model_dir / f"{name}.pkl", "rb") as f:
            heads[name] = pickle.load(f)
    return heads


def run_strategy(
    strategy: str,                      # "naive" | "heuristic" | "optimal"
    province: str,
    capacity_mw: float,
    test_days: int,
    forecast_noise_std: float,
    rng_seed: int,
    heads: dict,
    df: pd.DataFrame,
    scaled_actual_power: np.ndarray,
    rules: ComplianceRules,
    joint_dist: EmpiricalJointDistribution,
) -> dict:
    """Run one strategy through the backtest."""
    da_fcst = heads["da_forecaster"]
    spread_clf = heads["spread_direction"]
    sys_dev = heads["system_deviation"]

    split = int(len(df) * 0.8)
    test_start_day = split // 96
    n_days_total = len(df) // 96
    test_end_day = min(test_start_day + test_days, n_days_total - 1)

    # Generators
    heur_gen = BidCurveGenerator(capacity_mw=capacity_mw, config=BidCurveConfig())
    opt_gen = OptimalBidGenerator(
        capacity_mw=capacity_mw,
        rules=rules,
        config=OptimalBidConfig(n_scenarios=300),
        joint_dist=joint_dist,
    )

    evaluator = LoganEvaluator(SettlementConfig())
    rng = np.random.default_rng(rng_seed)

    total_rev = 0.0
    total_mlt_energy = 0.0
    total_mlt_congest = 0.0
    total_da = 0.0
    total_rt = 0.0
    total_clear = 0.0
    total_actual = 0.0

    compliance_fails = 0
    compliance_fixes = 0
    n_days = 0

    quantile_levels_4 = np.array([0.05, 0.25, 0.5, 0.75])
    quantile_levels_all = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    for d in range(test_start_day, test_end_day):
        try:
            da_q_96 = da_fcst.predict_day_all_quantiles(d, df)      # (96, 5)
            # 聚合到小时：用平均（简化）
            da_q_hr_all = da_q_96.reshape(24, 4, -1).mean(axis=1)   # (24, 5)
            da_q_hr_4 = da_q_hr_all[:, :4]                           # 去掉 P95

            spread_dir = spread_clf.predict_proba_day(d, df)         # (24,)

            start_idx = d * 96
            end_idx = start_idx + 96
            actual_power_96 = scaled_actual_power[start_idx:end_idx]
            actual_power_hr = aggregate_15min_to_hour(actual_power_96)

            noise = rng.normal(0, forecast_noise_std, 24)
            power_forecast_hr = np.clip(actual_power_hr * (1 + noise), 0.0, capacity_mw)

            rt_96 = df["rt_price"].iloc[start_idx:end_idx].fillna(0).values
            da_96 = df["da_price"].iloc[start_idx:end_idx].ffill().fillna(0).values
            rt_hr = aggregate_15min_to_hour(rt_96)
            da_hr = aggregate_15min_to_hour(da_96)

            if strategy == "naive":
                # 报 0 元保中标：cleared = forecast
                cleared_hr = power_forecast_hr.copy()
                res = evaluator.settle_hourly(
                    cleared_qty=cleared_hr,
                    actual_power=actual_power_hr,
                    da_prices_node=da_hr,
                    rt_prices_node=rt_hr,
                )
            elif strategy == "heuristic":
                sys_risk = sys_dev.predict_proba_day(d, df)
                bids = heur_gen.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr_4,
                    spread_dir_prob_hourly=spread_dir,
                    system_shortage_prob_hourly=sys_risk["prob_shortage"],
                    system_surplus_prob_hourly=sys_risk["prob_surplus"],
                )
                # Compliance enforce
                new_bids = []
                for t, bid in enumerate(bids):
                    original_valid = validate(bid, rules, forecast_mw=float(power_forecast_hr[t]))
                    if not original_valid.ok:
                        compliance_fails += 1
                    fixed, log = enforce(bid, rules, forecast_mw=float(power_forecast_hr[t]))
                    if fixed is not None:
                        new_bids.append(fixed)
                        if log:
                            compliance_fixes += 1
                    else:
                        # 无法修正，按 naive 处理
                        new_bids.append(bid)  # 可能违规，但留着让 settle 自然处理
                bids = new_bids
                res = evaluator.settle_bids(
                    bids=bids,
                    actual_power=actual_power_hr,
                    da_prices_node=da_hr,
                    rt_prices_node=rt_hr,
                )
            elif strategy == "optimal":
                bids = opt_gen.generate(
                    power_forecast_hourly=power_forecast_hr,
                    da_quantiles_hourly=da_q_hr_all,
                    quantile_levels=quantile_levels_all,
                )
                # Check compliance (Optimal 已自带合规，这里只是 double-check)
                for t, bid in enumerate(bids):
                    v = validate(bid, rules, forecast_mw=float(power_forecast_hr[t]))
                    if not v.ok:
                        compliance_fails += 1
                        # Try enforce
                        fixed, _ = enforce(bid, rules, forecast_mw=float(power_forecast_hr[t]))
                        if fixed is not None:
                            bids[t] = fixed
                            compliance_fixes += 1
                res = evaluator.settle_bids(
                    bids=bids,
                    actual_power=actual_power_hr,
                    da_prices_node=da_hr,
                    rt_prices_node=rt_hr,
                )
            else:
                raise ValueError(strategy)

            total_rev += res.total_revenue
            total_mlt_energy += res.mlt_energy_revenue
            total_mlt_congest += res.mlt_congestion_revenue
            total_da += res.da_spot_revenue
            total_rt += res.rt_spot_revenue
            total_clear += res.cleared_mwh
            total_actual += res.actual_mwh
            n_days += 1
        except Exception as e:
            logger.warning(f"Day {d} skipped ({strategy}): {e}")

    return {
        "strategy": strategy,
        "total_revenue": total_rev,
        "mlt_energy": total_mlt_energy,
        "mlt_congest": total_mlt_congest,
        "da_spot": total_da,
        "rt_spot": total_rt,
        "cleared_mwh": total_clear,
        "actual_mwh": total_actual,
        "clear_ratio": total_clear / max(total_actual, 1.0),
        "compliance_fails": compliance_fails,
        "compliance_fixes": compliance_fixes,
        "n_days": n_days,
    }


def main(province: str = "gansu", capacity_mw: float = 100.0,
         test_days: int = 90, forecast_noise_std: float = 0.08,
         rng_seed: int = 42):
    logger.info(f"=== Gansu Realistic Backtest ({province}, {test_days}d, {capacity_mw} MW) ===")

    # Load rules
    rules = ComplianceRules.from_yaml(RULES_PATH, capacity_mw=capacity_mw)
    logger.info(f"Compliance rules loaded: {rules.min_steps}-{rules.max_steps} 段, "
                f"每段 ≥ {rules.min_step_mw:.1f} MW, 价精度 {rules.price_precision} 元/MWh")

    # Data + models
    heads = load_heads(province)
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
    scaled_actual_power = raw / raw_max * capacity_mw if raw_max > 0 else np.zeros_like(raw)

    # Fit joint distribution (recent 540d)
    train_end = int(len(df) * 0.8)
    joint_start = max(0, train_end - 540 * 96)
    df_joint = df.iloc[joint_start:train_end]
    logger.info(f"Fitting joint distribution on {len(df_joint):,} rows (recent 540d)")
    joint_dist = EmpiricalJointDistribution(JointDistConfig(n_da_buckets=10))
    joint_dist.fit(df_joint)
    rhos = joint_dist.compute_rank_correlation()
    logger.info(f"Spearman ρ(DA, RT): mean={rhos.mean():.2f}, min={rhos.min():.2f}, max={rhos.max():.2f}")

    # Run three strategies
    results = []
    for strat in ["naive", "heuristic", "optimal"]:
        logger.info(f"\n--- Running {strat} ---")
        r = run_strategy(
            strategy=strat, province=province, capacity_mw=capacity_mw,
            test_days=test_days, forecast_noise_std=forecast_noise_std,
            rng_seed=rng_seed, heads=heads, df=df,
            scaled_actual_power=scaled_actual_power,
            rules=rules, joint_dist=joint_dist,
        )
        logger.info(
            f"  Revenue: ¥{r['total_revenue']:>12,.0f}  "
            f"DA=¥{r['da_spot']:>10,.0f}  RT=¥{r['rt_spot']:>+10,.0f}  "
            f"Clear={r['clear_ratio']*100:.1f}%  "
            f"Comply fails={r['compliance_fails']}, fixes={r['compliance_fixes']}"
        )
        results.append(r)

    # Comparison
    logger.info(f"\n{'=' * 80}")
    logger.info(f"  Gansu Realistic Settlement Comparison")
    logger.info(f"{'=' * 80}")
    naive = next(r for r in results if r["strategy"] == "naive")
    for r in results:
        gain_abs = r["total_revenue"] - naive["total_revenue"]
        gain_pct = gain_abs / max(abs(naive["total_revenue"]), 1.0) * 100
        logger.info(
            f"{r['strategy']:<12} "
            f"¥{r['total_revenue']:>12,.0f}  "
            f"gain={gain_abs:>+10,.0f} ({gain_pct:+.2f}%)  "
            f"clear={r['clear_ratio']*100:>5.1f}%"
        )

    # Save
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(RUNS_DIR / "realistic_backtest_gansu.csv", index=False)
    logger.info(f"\nSaved: {RUNS_DIR / 'realistic_backtest_gansu.csv'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="gansu")
    parser.add_argument("--capacity", type=float, default=100.0)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--noise", type=float, default=0.08)
    args = parser.parse_args()
    main(args.province, args.capacity, args.days, args.noise)
