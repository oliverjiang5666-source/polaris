"""
Polaris 山东 · Script 01 — 规则数据化 + bid curve + 结算 smoke test
======================================================================

验证整条链路:
  1. 读 shandong.yaml → ComplianceRules
  2. 从 Tensor DP 96 点 power 生成 5+5 段 bid curve
  3. compliance.validate + enforce
  4. ShandongEvaluator.settle_from_bid_curve 结算
  5. 对比: 山东 Two-Settlement 结算 vs 原 Polaris 单价过账

⚠️  本 smoke 用 rt_price 作所有 LMP 的 proxy（Q1=A 方案）。
    真实 LMP 数据到位后只需传真实数据，不需改代码。

用法:
    PYTHONPATH=. python3 products/polaris_shandong/scripts/01_verify_rules.py
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
from oracle.lp_oracle import solve_day
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig

from products.polaris_shandong.bid_curve import (
    StorageBidCurve, build_from_tensor_dp_plan,
)
from products.polaris_shandong.compliance import ComplianceRules, validate, enforce
from products.polaris_shandong.evaluator import ShandongEvaluator, SettlementConfig
from products.polaris_shandong.capacity_compensation import (
    daily_available_capacity_mw, estimate_monthly_fee_from_standalone_price,
)


RULES_PATH = ROOT / "products" / "polaris_shandong" / "settlement_rules" / "shandong.yaml"
PROCESSED = ROOT / "data" / "china" / "processed"


def smoke_bid_curve_compliance():
    """Step 1+2+3: yaml → rules → bid curve → compliance"""
    logger.info("=" * 70)
    logger.info("  Step 1: 加载 ComplianceRules")
    logger.info("=" * 70)

    rules = ComplianceRules.from_yaml(RULES_PATH, rated_charge_power_mw=200, rated_discharge_power_mw=200)
    logger.info(f"  n_segments_max/side = {rules.n_segments_max_per_side}")
    logger.info(f"  segment_min_mw = {rules.segment_min_mw}")
    logger.info(f"  rated_charge/discharge = {rules.rated_charge_power_mw}/{rules.rated_discharge_power_mw} MW")
    logger.info(f"  min_continuous_minutes = {rules.min_continuous_minutes}")

    # 生成一个合成 power_96 (典型"低价充午间、高价放晚间"模式)
    np.random.seed(42)
    hours = np.arange(96) / 4
    lmp_96 = 320 - 80 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 20, 96)
    # Tensor DP 最优
    battery = BatteryConfig()
    dp = TensorDP(battery, DPConfig(delta_soc=0.005))
    V = dp.backward_induction(lmp_96[:, None], np.ones((96, 1)))
    sim = dp.forward_simulate(V, lmp_96, init_soc=0.5)
    power_96 = sim["powers"]

    logger.info(f"\n  Tensor DP 96 点 power: 放电 sum={power_96[power_96>0].sum()*0.25:.1f} MWh, "
                f"充电 sum={-power_96[power_96<0].sum()*0.25:.1f} MWh")
    logger.info(f"  Tensor DP revenue (单价过账): ¥{sim['revenue_total']:,.0f}")

    logger.info("\n" + "=" * 70)
    logger.info("  Step 2: Tensor DP → 5+5 段 bid curve (convexification)")
    logger.info("=" * 70)

    charge_segs, discharge_segs = build_from_tensor_dp_plan(
        power_96=power_96,
        lmp_96=lmp_96,
        rated_charge_power=rules.rated_charge_power_mw,
        rated_discharge_power=rules.rated_discharge_power_mw,
        n_segments_each_side=5,
    )

    bid = StorageBidCurve(
        charge_segments=charge_segs,
        discharge_segments=discharge_segs,
        rated_charge_power_mw=rules.rated_charge_power_mw,
        rated_discharge_power_mw=rules.rated_discharge_power_mw,
        soc_min_pct=5.0,
        soc_max_pct=95.0,
        round_trip_efficiency_pct=90.0,
        min_continuous_minutes=15,
        initial_soc_pct=50.0,
        da_charge_upper_96=np.full(96, rules.rated_charge_power_mw),
        da_discharge_upper_96=np.full(96, rules.rated_discharge_power_mw),
        rationale="TensorDP convexified",
    )

    logger.info("  充电段:")
    for i, s in enumerate(bid.charge_segments):
        logger.info(f"    [{i+1}] [{s.start_mw:+7.2f}, {s.end_mw:+7.2f}] MW  price={s.price_yuan_mwh:.2f}")
    logger.info("  放电段:")
    for i, s in enumerate(bid.discharge_segments):
        logger.info(f"    [{i+1}] [{s.start_mw:+7.2f}, {s.end_mw:+7.2f}] MW  price={s.price_yuan_mwh:.2f}")

    logger.info("\n" + "=" * 70)
    logger.info("  Step 3: compliance.validate")
    logger.info("=" * 70)

    v = validate(bid, rules)
    if v.ok:
        logger.info("  ✅ 合规")
    else:
        logger.warning(f"  ❌ 不合规: {v.errors}")
        logger.info("  尝试 enforce...")
        fixed, log = enforce(bid, rules)
        v2 = validate(fixed, rules)
        logger.info(f"  enforce 日志: {log}")
        logger.info(f"  修正后: {'✅ 合规' if v2.ok else '❌ 仍不合规: ' + str(v2.errors)}")
        bid = fixed

    return bid, lmp_96, power_96, sim


def smoke_settlement(bid: StorageBidCurve, lmp_96: np.ndarray, actual_power_96: np.ndarray):
    """Step 4: 结算"""
    logger.info("\n" + "=" * 70)
    logger.info("  Step 4: ShandongEvaluator (Two-Settlement + CfD)")
    logger.info("=" * 70)

    # LMP proxy: 所有价都传 lmp_96（rt_price proxy）
    # ⚠️  TODO: 真实需要区分 da_lmp_gen / rt_lmp_gen / da_lmp_user / rt_lmp_user / rt_unified
    da_lmp = lmp_96        # 日前节点 (这里 proxy)
    rt_lmp = lmp_96        # 实时节点 (这里 proxy - 实际应该和 DA 有 spread)
    unified = lmp_96       # 统一结算点

    evaluator = ShandongEvaluator(SettlementConfig())
    res = evaluator.settle_from_bid_curve(
        bid=bid,
        actual_power_96=actual_power_96,
        da_lmp_gen_96=da_lmp,
        rt_lmp_gen_96=rt_lmp,
        rt_unified_96=unified,
    )
    logger.info("\n" + res.summary())

    return res


def smoke_settlement_with_real_da_rt_spread(bid: StorageBidCurve, lmp_96: np.ndarray, actual_power_96: np.ndarray):
    """
    Step 4b: 更接近真实的结算：DA 和 RT 价不一样
    模拟 DA 预测误差：DA = LMP × (1 + noise_small)，RT = LMP
    """
    logger.info("\n" + "=" * 70)
    logger.info("  Step 4b: 模拟 DA-RT 价差（更接近真实）")
    logger.info("=" * 70)

    rng = np.random.default_rng(123)
    # DA 价 ~ LMP × (1 + 5% noise)
    da_lmp = lmp_96 * (1 + rng.normal(0, 0.05, 96))
    rt_lmp = lmp_96
    unified = lmp_96   # 假设省统一价 ≈ 实时节点

    evaluator = ShandongEvaluator(SettlementConfig())
    res = evaluator.settle_from_bid_curve(
        bid=bid,
        actual_power_96=actual_power_96,
        da_lmp_gen_96=da_lmp,
        rt_lmp_gen_96=rt_lmp,
        rt_unified_96=unified,
    )
    logger.info("\n" + res.summary())

    return res


def smoke_capacity_compensation():
    """Step 5: 容量补偿"""
    logger.info("\n" + "=" * 70)
    logger.info("  Step 5: 容量补偿估算 (§3.4.6 第四项)")
    logger.info("=" * 70)

    daily = daily_available_capacity_mw(
        rated_discharge_power_mw=200.0,
        available_hours_on_day=24.0,
        certified_duration_hours=2.0,
    )
    logger.info(f"  日可用容量: {daily:.2f} MW")

    # 示意电价 100 元/kW/月
    est = estimate_monthly_fee_from_standalone_price(
        rated_discharge_power_mw=200.0,
        certified_duration_hours=2.0,
        capacity_price_yuan_per_kw_month=100.0,
    )
    logger.info(f"  粗估月度容量补偿 (100 元/kW/月): ¥{est:,.0f}")
    logger.info(f"  ⚠️  实际价格需从省发改委核定，上数仅示意")


def main():
    logger.info("\n" + "#" * 70)
    logger.info("#  Polaris 山东版 smoke test - 规则验证 + 结算链路")
    logger.info("#" * 70)

    bid, lmp_96, power_96, tdp_sim = smoke_bid_curve_compliance()
    res_same_price = smoke_settlement(bid, lmp_96, power_96)
    res_with_spread = smoke_settlement_with_real_da_rt_spread(bid, lmp_96, power_96)
    smoke_capacity_compensation()

    logger.info("\n" + "=" * 70)
    logger.info("  对比: 原 Polaris (单价过账) vs 山东 Two-Settlement")
    logger.info("=" * 70)
    logger.info(f"  原 Polaris (Tensor DP,  单价过账): ¥{tdp_sim['revenue_total']:,.0f}")
    logger.info(f"  山东 (同价 DA=RT):                 ¥{res_same_price.total_revenue:,.0f}")
    logger.info(f"  山东 (5% DA-RT spread):            ¥{res_with_spread.total_revenue:,.0f}")
    logger.info(f"")
    logger.info(f"  观察: 同价时山东净结算 ≈ Polaris (因为 CfD 差项为 0, 实时实量结算)")
    logger.info(f"  观察: 有 spread 时日前 CfD 部分带来 ± 差价收益/亏损")


if __name__ == "__main__":
    main()
