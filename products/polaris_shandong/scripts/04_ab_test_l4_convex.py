"""
Polaris 山东 · Script 04 — A/B test L4 bid curve 凸化: quantile vs convex_hull
==============================================================================

测试路径 (真实封存报价流程):
  1. Tensor DP 出 96 点 plan
  2. Bid curve 构造 (2 种方法 A/B)
  3. 按真实 LMP 清算 bid curve → cleared_power_96
  4. Evaluator 按 cleared_power 结算

测试集: 2025 年每月 1 号 (12 天), 能看趋势又不太久.

方法:
  A (quantile):     旧启发式分位数切
  B (convex_hull):  严格 Lee-Sun §II-C upper convex hull + RDP

Usage:
    PYTHONPATH=. python3 products/polaris_shandong/scripts/04_ab_test_l4_convex.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import BatteryConfig
from optimization.milp.data_loader import load_province
from optimization.milp.scenario_generator import RegimeClassifier
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig

from products.polaris_shandong.bid_curve import (
    StorageBidCurve, build_from_tensor_dp_plan,
)
from products.polaris_shandong.compliance import ComplianceRules, validate


def build_scenarios_regime_conditioned(classifier, data, target_day, R_cap=500):
    probs, _ = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()
    train_labels = classifier.train_labels
    n_train = len(train_labels)
    prices_all = data.rt_prices[:n_train]
    n_reg = classifier.n_regimes
    counts = np.array([(train_labels == c).sum() for c in range(n_reg)])
    w = np.zeros(n_train)
    for d in range(n_train):
        if counts[train_labels[d]] > 0:
            w[d] = probs[train_labels[d]] / counts[train_labels[d]]
    w = w / w.sum()
    idx = np.argsort(-w)[:R_cap]
    idx = np.sort(idx)
    return prices_all[idx].T, np.tile((w[idx] / w[idx].sum())[None, :], (96, 1))


def ab_test(capacity_mw=100.0, capacity_mwh=200.0, delta_soc=0.005):
    capacity_mwh_g = capacity_mwh    # closure for inner
    battery = BatteryConfig(capacity_mw=capacity_mw, capacity_mwh=capacity_mwh)
    data = load_province("shandong")
    day_starts = data.df.index[::96].normalize()

    # 选 2025 每月 1 号
    target_dates = [pd.Timestamp(f"2025-{m:02d}-01") for m in range(1, 13)]
    test_days = []
    for td in target_dates:
        matches = (day_starts >= td)
        if matches.any():
            test_days.append(int(matches.argmax()))

    # 每季度训练 classifier (简化: 只用 2024-12-31 cutoff)
    y25_start = int((day_starts >= pd.Timestamp("2025-01-01")).argmax())
    clf = RegimeClassifier(n_regimes=12)
    clf.fit(data, train_day_end=y25_start)

    dp = TensorDP(battery, DPConfig(delta_soc=delta_soc))
    logger.info(f"TensorDP S={dp.S}, P={dp.P}")

    # Rules for compliance check
    rules_path = ROOT / "products" / "polaris_shandong" / "settlement_rules" / "shandong.yaml"
    rules = ComplianceRules.from_yaml(rules_path, capacity_mw, capacity_mw)

    records = []
    for d in test_days:
        # DP
        scen, probs = build_scenarios_regime_conditioned(clf, data, d)
        V = dp.backward_induction(scen, probs)
        sim = dp.forward_simulate(V, data.rt_prices[d], init_soc=0.5)
        power_96 = sim["powers"]
        rt_96 = data.rt_prices[d]

        dp_revenue = float((power_96 * rt_96 * 0.25).sum() - 2.0 * np.abs(power_96).sum() * 0.25)

        results = {}
        for method in ["quantile", "convex_hull"]:
            charge_segs, discharge_segs = build_from_tensor_dp_plan(
                power_96=power_96,
                lmp_96=rt_96,
                rated_charge_power=capacity_mw,
                rated_discharge_power=capacity_mw,
                n_segments_each_side=5,
                method=method,
            )
            bid = StorageBidCurve(
                charge_segments=charge_segs,
                discharge_segments=discharge_segs,
                rated_charge_power_mw=capacity_mw,
                rated_discharge_power_mw=capacity_mw,
                da_charge_upper_96=np.full(96, capacity_mw),
                da_discharge_upper_96=np.full(96, capacity_mw),
            )
            v = validate(bid, rules)

            # Cleared power by bid curve (按 LMP 触发, SoC-aware 保证物理可行)
            cleared_96 = bid.cleared_series_96(
                rt_96, soc_aware=True, capacity_mwh=capacity_mwh,
                initial_soc_pct=50.0, dt_hours=0.25,
            )
            bid_revenue = float((cleared_96 * rt_96 * 0.25).sum() - 2.0 * np.abs(cleared_96).sum() * 0.25)

            # Recovery = bid revenue / DP revenue
            recovery = bid_revenue / dp_revenue if dp_revenue > 1 else 0.0

            # Power mismatch: RMS of (cleared - dp)
            rms_diff = float(np.sqrt(((cleared_96 - power_96) ** 2).mean()))

            results[method] = {
                "compliance_ok": v.ok,
                "bid_revenue": bid_revenue,
                "recovery_vs_dp": recovery,
                "rms_power_diff": rms_diff,
                "n_chg_segs": len(charge_segs),
                "n_dis_segs": len(discharge_segs),
            }

        records.append({
            "day_idx": d,
            "date": str(data.df.index[d*96].date()),
            "dp_revenue": dp_revenue,
            "quantile_rev": results["quantile"]["bid_revenue"],
            "quantile_recovery": results["quantile"]["recovery_vs_dp"],
            "quantile_rms_diff": results["quantile"]["rms_power_diff"],
            "quantile_comply": results["quantile"]["compliance_ok"],
            "convex_rev": results["convex_hull"]["bid_revenue"],
            "convex_recovery": results["convex_hull"]["recovery_vs_dp"],
            "convex_rms_diff": results["convex_hull"]["rms_power_diff"],
            "convex_comply": results["convex_hull"]["compliance_ok"],
        })

        logger.info(
            f"  {records[-1]['date']}: DP=¥{dp_revenue:>7,.0f}  "
            f"Quantile=¥{results['quantile']['bid_revenue']:>7,.0f} ({results['quantile']['recovery_vs_dp']*100:>5.1f}%)  "
            f"Convex=¥{results['convex_hull']['bid_revenue']:>7,.0f} ({results['convex_hull']['recovery_vs_dp']*100:>5.1f}%)  "
            f"Δ=¥{results['convex_hull']['bid_revenue'] - results['quantile']['bid_revenue']:+,.0f}"
        )

    df = pd.DataFrame(records)

    logger.info(f"\n{'='*80}")
    logger.info(f"  L4 A/B 汇总 (12 天 2025 每月 1 号, {capacity_mw}MW/{capacity_mwh}MWh)")
    logger.info(f"{'='*80}")
    logger.info(f"  DP 原始 (理想):       ¥{df['dp_revenue'].sum():>10,.0f}")
    logger.info(f"  Quantile (启发式):    ¥{df['quantile_rev'].sum():>10,.0f}  "
                f"recovery mean {df['quantile_recovery'].mean()*100:.1f}%  "
                f"合规 {df['quantile_comply'].sum()}/{len(df)}")
    logger.info(f"  Convex Hull (严格):   ¥{df['convex_rev'].sum():>10,.0f}  "
                f"recovery mean {df['convex_recovery'].mean()*100:.1f}%  "
                f"合规 {df['convex_comply'].sum()}/{len(df)}")
    gain = df['convex_rev'].sum() - df['quantile_rev'].sum()
    logger.info(f"  Convex vs Quantile:  ¥{gain:+,.0f} ({gain/df['quantile_rev'].sum()*100:+.2f}%)")
    logger.info(f"\n  RMS 功率偏差 (cleared vs DP):")
    logger.info(f"    Quantile: {df['quantile_rms_diff'].mean():.2f} MW")
    logger.info(f"    Convex:   {df['convex_rms_diff'].mean():.2f} MW")

    return df


if __name__ == "__main__":
    ab_test()
