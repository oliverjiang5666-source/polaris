"""
Polaris 山东 · Script 03 — 2025 自然年回测 (100MW / 200MWh, 不含容量补偿)
==========================================================================

用 Tensor DP V2 (regime-conditioned) + 山东 Two-Settlement evaluator,
100 MW / 200 MWh 电池, 2025-01-01 ~ 2025-12-31.

三个结算口径对比（同一 DP 决策，不同结算公式）:
  1. 单价过账 (原 Polaris): R = Σ power × rt_price × dt
  2. 按 DA 结算:             R = Σ power × da_price × dt  (等价 §14.5.6 的 Q_actual=Q_DA + rt_unified=rt 假设)
  3. 山东 Two-Settlement:    R = Σ Q_actual × rt + Σ Q_DA_cleared × (da - rt_unified)

训练数据: 2020-11 → 2024-12 (4+ 年历史), 每季度 retrain 分类器
测试: 2025 全年 365 天 walk-forward

⚠️  不含: 容量补偿 / AGC / MLT

用法:
    PYTHONPATH=. python3 products/polaris_shandong/scripts/03_backtest_2025_100mw.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from config import BatteryConfig
from oracle.lp_oracle import solve_day
from optimization.milp.data_loader import load_province
from optimization.milp.scenario_generator import RegimeClassifier
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig
from forecast.mpc_controller import _step_battery


OUTPUT = ROOT / "runs" / "polaris_shandong" / "2025_backtest"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2025 days + quarterly split
# ============================================================

def get_2025_days_quarters(data):
    """返回 2025 自然年 day indices + Q1-Q4 边界"""
    day_starts = data.df.index[::96].normalize()
    y25 = pd.Timestamp("2025-01-01")
    y26 = pd.Timestamp("2026-01-01")

    start_day = int((day_starts >= y25).argmax())
    if not (day_starts >= y26).any():
        end_day = len(day_starts)
    else:
        end_day = int((day_starts >= y26).argmax())

    test_days = list(range(start_day, end_day))

    quarter_starts = [
        pd.Timestamp(f"2025-{m:02d}-01") for m in [1, 4, 7, 10]
    ] + [pd.Timestamp("2026-01-01")]
    bounds = []
    for i in range(4):
        qs = int((day_starts >= quarter_starts[i]).argmax())
        qe = int((day_starts >= quarter_starts[i + 1]).argmax()) if (day_starts >= quarter_starts[i + 1]).any() else end_day
        bounds.append((qs, qe))

    return test_days, bounds


# ============================================================
# Regime-conditioned scenarios (same as script 35)
# ============================================================

def build_scenarios_regime_conditioned(classifier, data, target_day, R_cap, price_attr="rt_prices"):
    """V2: regime-conditioned bootstrap.

    price_attr: "rt_prices" 或 "dam_prices" - 决定场景和 DP forward 用哪个价
    """
    probs, _ = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()
    train_labels = classifier.train_labels
    n_train = len(train_labels)
    prices_all = getattr(data, price_attr)[:n_train]

    n_reg = classifier.n_regimes
    counts = np.array([(train_labels == c).sum() for c in range(n_reg)])
    day_weights = np.zeros(n_train)
    for d in range(n_train):
        reg_d = train_labels[d]
        if counts[reg_d] > 0:
            day_weights[d] = probs[reg_d] / counts[reg_d]
    day_weights = day_weights / day_weights.sum()

    if R_cap is not None and R_cap < n_train:
        idx_sorted = np.argsort(-day_weights)[:R_cap]
        idx_sorted = np.sort(idx_sorted)
        rt_sub = prices_all[idx_sorted]
        w_sub = day_weights[idx_sorted]
        w_sub = w_sub / w_sub.sum()
    else:
        rt_sub = prices_all
        w_sub = day_weights

    return rt_sub.T, np.tile(w_sub[None, :], (96, 1))


# ============================================================
# Three settlement formulas applied to same DP plan
# ============================================================

def settle_three_ways(powers_96, rt_96, da_96, dt=0.25, deg_cost=2.0):
    """
    对同一天的 DP 输出 (power_96) 按三种口径结算.
    Powers 正=放电, 负=充电.
    """
    total_deg = deg_cost * np.abs(powers_96).sum() * dt

    # 1. 单价过账 (原 Polaris)
    rev_rt = float((powers_96 * rt_96 * dt).sum()) - total_deg

    # 2. 按 DA 价结算 (近似 §14.5.6 Q_actual=Q_DA 假设)
    rev_da = float((powers_96 * da_96 * dt).sum()) - total_deg

    # 3. 山东 Two-Settlement:
    #    R = Σ Q_actual × rt + Σ Q_DA × (da - rt_unified)
    #    假设 Q_actual = Q_DA (完美执行), rt_unified ≈ rt
    #    → R = Σ power × rt + Σ power × (da - rt) = Σ power × da (等价于口径 2)
    #
    # 但若我们假设 Q_DA ≠ Q_actual 时有实时偏差，这里储能做的是日前 DP,
    # 实际执行完全按 DP 计划（完美执行），所以 Q_actual = Q_DA
    # Shandong two-settlement 和"按 DA 结算"在完美执行假设下等价
    rev_shandong = rev_da

    return {
        "rev_single_rt": rev_rt,
        "rev_single_da": rev_da,
        "rev_shandong_twosettle": rev_shandong,
        "deg_cost": total_deg,
    }


# ============================================================
# Main backtest
# ============================================================

def run_2025_backtest_100mw(
    province: str = "shandong",
    capacity_mw: float = 100.0,
    capacity_mwh: float = 200.0,
    delta_soc: float = 0.005,
    R_cap: int = 500,
    dp_price_basis: str = "rt_prices",     # "rt_prices" or "dam_prices"
):
    """
    2025 自然年 walk-forward, 季度重训 classifier, 每天独立 Tensor DP.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"  2025 全年 Polaris 山东版回测 - {capacity_mw}MW/{capacity_mwh}MWh")
    logger.info(f"  DP price basis: {dp_price_basis}")
    logger.info(f"{'='*80}")

    # 电池参数覆盖
    battery = BatteryConfig(capacity_mw=capacity_mw, capacity_mwh=capacity_mwh)
    logger.info(f"  Battery: {battery.capacity_mw}MW / {battery.capacity_mwh}MWh, "
                f"RTE={battery.round_trip_efficiency}, deg={battery.degradation_cost_per_mwh}¥/MWh")

    data = load_province(province)
    logger.info(f"  Data: {data.df.index.min().date()} → {data.df.index.max().date()}, {data.n_days} days")

    test_days, qs_bounds = get_2025_days_quarters(data)
    logger.info(f"  Test: 2025 全年 {len(test_days)} days ({data.df.index[test_days[0]*96].date()} → {data.df.index[test_days[-1]*96 + 95].date()})")
    logger.info(f"  Training cutoff per quarter:")
    for qi, (qs, qe) in enumerate(qs_bounds):
        logger.info(f"    Q{qi+1}: train-end = day {qs} ({data.df.index[qs*96].date()}), test = {qe-qs} days")

    # 每季度训练分类器
    classifiers = {}
    for qi, (qs, _) in enumerate(qs_bounds):
        t0 = time.time()
        clf = RegimeClassifier(n_regimes=12)
        clf.fit(data, train_day_end=qs)
        classifiers[qi] = clf
        logger.info(f"  Q{qi+1} classifier trained ({time.time()-t0:.1f}s)")

    # 初始化 DP
    dp = TensorDP(battery, DPConfig(delta_soc=delta_soc))
    logger.info(f"  TensorDP: S={dp.S}, P={dp.P}")

    # 按天回测
    records = []
    t_start = time.time()

    for i, d in enumerate(test_days):
        # 找所属 quarter
        qi_this = 0
        for qi, (qs, qe) in enumerate(qs_bounds):
            if qs <= d < qe:
                qi_this = qi
                break
        clf = classifiers[qi_this]

        # 场景生成（用指定价格基准）
        price_scen, price_probs = build_scenarios_regime_conditioned(
            clf, data, target_day=d, R_cap=R_cap, price_attr=dp_price_basis
        )

        # Backward induction
        V = dp.backward_induction(price_scen, price_probs)

        # Forward simulation (按 dp_price_basis 对应的真实价)
        actual_prices_for_dp = data.rt_prices[d] if dp_price_basis == "rt_prices" else data.dam_prices[d]
        sim = dp.forward_simulate(V, actual_prices_for_dp, init_soc=0.5)
        power_96 = sim["powers"]

        # 拿真实 DA 和 RT 做三种结算
        rt_96 = data.rt_prices[d]
        da_96 = data.dam_prices[d]

        settle = settle_three_ways(power_96, rt_96=rt_96, da_96=da_96, deg_cost=battery.degradation_cost_per_mwh)

        # Oracle (perfect foresight on RT)
        oracle = solve_day(rt_96, battery, init_soc=0.5)
        oracle_rev = oracle["revenue"]

        records.append({
            "day_idx": d,
            "date": str(data.df.index[d*96].date()),
            "quarter": qi_this + 1,
            "rev_single_rt": settle["rev_single_rt"],
            "rev_single_da": settle["rev_single_da"],
            "rev_shandong_twosettle": settle["rev_shandong_twosettle"],
            "oracle_rt": oracle_rev,
            "capture_single_rt": settle["rev_single_rt"] / oracle_rev * 100 if oracle_rev > 0 else 0,
            "rt_mean": float(rt_96.mean()),
            "da_mean": float(da_96.mean()),
            "da_rt_spread_mean": float((da_96 - rt_96).mean()),
        })

        if (i + 1) % 30 == 0 or i == 0:
            logger.info(
                f"    day {d} ({data.df.index[d*96].date()}, {i+1}/{len(test_days)}): "
                f"RT=¥{settle['rev_single_rt']:>9,.0f}  "
                f"DA=¥{settle['rev_single_da']:>9,.0f}  "
                f"Oracle=¥{oracle_rev:>9,.0f}  "
                f"cap={records[-1]['capture_single_rt']:>5.1f}%"
            )

    elapsed = time.time() - t_start
    df = pd.DataFrame(records)

    # ============================================================
    # 汇总
    # ============================================================
    logger.info(f"\n  {len(test_days)} days in {elapsed:.1f}s ({elapsed/len(test_days):.2f}s/day)")

    total_rt = df["rev_single_rt"].sum()
    total_da = df["rev_single_da"].sum()
    total_sd = df["rev_shandong_twosettle"].sum()
    total_oracle = df["oracle_rt"].sum()

    logger.info(f"\n  {'='*70}")
    logger.info(f"  2025 全年 100MW/200MWh 汇总")
    logger.info(f"  {'='*70}")
    logger.info(f"    单价过账 (按 RT 价):         ¥{total_rt:>14,.0f}  capture {total_rt/total_oracle*100:>5.1f}%")
    logger.info(f"    单价过账 (按 DA 价):         ¥{total_da:>14,.0f}  capture {total_da/total_oracle*100:>5.1f}%")
    logger.info(f"    山东 Two-Settlement:          ¥{total_sd:>14,.0f}  capture {total_sd/total_oracle*100:>5.1f}%")
    logger.info(f"    Oracle (RT perfect):         ¥{total_oracle:>14,.0f}  100.0%")
    logger.info(f"    Oracle 500 × 0.5 (200→100):  ¥{total_oracle * 0.5:>14,.0f}  参考：线性折算验证")

    logger.info(f"\n  按季度:")
    for qi in range(4):
        sub = df[df["quarter"] == qi + 1]
        logger.info(
            f"    Q{qi+1} ({len(sub):>3}天): "
            f"RT=¥{sub['rev_single_rt'].sum():>11,.0f}  "
            f"DA=¥{sub['rev_single_da'].sum():>11,.0f}  "
            f"Oracle=¥{sub['oracle_rt'].sum():>11,.0f}  "
            f"cap={sub['rev_single_rt'].sum()/sub['oracle_rt'].sum()*100:>5.1f}%"
        )

    logger.info(f"\n  DA-RT spread 2025 统计:")
    logger.info(f"    mean spread: {df['da_rt_spread_mean'].mean():+.2f} 元/MWh")
    logger.info(f"    日 RT - DA 差异: 平均每天 ¥{(total_da - total_rt)/len(test_days):+,.0f}")

    # 存结果
    csv_path = OUTPUT / f"{province}_2025_{int(capacity_mw)}mw_{dp_price_basis}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n  Per-day results saved: {csv_path}")

    return {
        "total_rt": total_rt,
        "total_da": total_da,
        "total_shandong": total_sd,
        "total_oracle": total_oracle,
        "capture_rt": total_rt / total_oracle * 100,
        "df": df,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-mw", type=float, default=100.0)
    parser.add_argument("--capacity-mwh", type=float, default=200.0)
    parser.add_argument("--delta", type=float, default=0.005)
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--basis", default="rt_prices", choices=["rt_prices", "dam_prices"])
    args = parser.parse_args()

    run_2025_backtest_100mw(
        capacity_mw=args.capacity_mw,
        capacity_mwh=args.capacity_mwh,
        delta_soc=args.delta,
        R_cap=args.R,
        dp_price_basis=args.basis,
    )
