"""
P1a: Worst-case 日诊断
=======================

V2 (regime-conditioned Tensor DP) 在山东全年 capture 68.4%，但 min = -78.9%（负收益！）
找出哪些天 capture 异常低，诊断失败模式：
  - 分类器预测的 regime probs 对不对？
  - DP 选的 action 轨迹为什么差？
  - 真实 RT 价是什么形态？与场景预期差多少？
  - SoC 是否卡到极端（0.05 / 0.95）？

Usage:
    PYTHONPATH=. python3 scripts/34_vfa_worst_day_diagnose.py [--top N]
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from optimization.milp.data_loader import load_province, split_walkforward
from optimization.milp.scenario_generator import RegimeClassifier
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig
from oracle.lp_oracle import solve_day


OUTPUT = Path("runs/vfa_dp/diagnose")
OUTPUT.mkdir(parents=True, exist_ok=True)


def diagnose_one_day(
    data,
    classifier: RegimeClassifier,
    dp: TensorDP,
    target_day: int,
) -> dict:
    """对单日做深度诊断，记录所有中间量"""
    # 1. 场景生成
    probs, _classes = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()

    train_labels = classifier.train_labels
    n_train = len(train_labels)
    train_rt = data.rt_prices[:n_train]

    n_reg = classifier.n_regimes
    counts = np.array([(train_labels == c).sum() for c in range(n_reg)])
    day_weights = np.zeros(n_train)
    for d in range(n_train):
        reg_d = train_labels[d]
        if counts[reg_d] > 0:
            day_weights[d] = probs[reg_d] / counts[reg_d]
    day_weights = day_weights / day_weights.sum()

    # R_cap=500 保持和 backtest 一致
    idx_sorted = np.argsort(-day_weights)[:500]
    idx_sorted = np.sort(idx_sorted)
    rt_sub = train_rt[idx_sorted]
    w_sub = day_weights[idx_sorted]
    w_sub = w_sub / w_sub.sum()

    price_scen = rt_sub.T
    price_probs = np.tile(w_sub[None, :], (96, 1))

    # 期望价格（看场景 vs 实际）
    expected_rt = (price_scen * price_probs).sum(axis=1)

    # 2. DP backward + forward
    V = dp.backward_induction(price_scen, price_probs)
    actual_rt = data.rt_prices[target_day]
    sim = dp.forward_simulate(V, actual_rt, init_soc=0.5)

    # 3. Oracle (perfect foresight)
    battery = dp.battery
    oracle = solve_day(actual_rt, battery, init_soc=0.5)

    # 4. 关键指标
    return {
        "day_idx": target_day,
        "regime_probs_top3": sorted(enumerate(probs), key=lambda x: -x[1])[:3],
        "expected_rt": expected_rt,
        "actual_rt": actual_rt,
        "rt_gap_mean": (actual_rt - expected_rt).mean(),
        "rt_gap_std": (actual_rt - expected_rt).std(),
        "rt_gap_max_abs": np.abs(actual_rt - expected_rt).max(),
        "actual_rt_mean": actual_rt.mean(),
        "actual_rt_std": actual_rt.std(),
        "actual_rt_range": actual_rt.max() - actual_rt.min(),
        "dp_revenue": sim["revenue_total"],
        "oracle_revenue": oracle["revenue"],
        "capture": sim["revenue_total"] / oracle["revenue"] * 100 if oracle["revenue"] > 0 else 0,
        "dp_powers": sim["powers"],
        "oracle_powers": oracle["net_power"],
        "dp_soc": sim["soc_trajectory"],
        "oracle_soc": oracle["soc"],
        "soc_hit_min_steps": int((sim["soc_trajectory"] <= 0.051).sum()),
        "soc_hit_max_steps": int((sim["soc_trajectory"] >= 0.949).sum()),
        "dp_final_soc": sim["final_soc"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="top N worst days")
    parser.add_argument("--province", default="shandong")
    args = parser.parse_args()

    # 读 V2 full 结果
    csv_path = Path("runs/vfa_dp") / f"{args.province}_vfa_dp_full365d_regime_conditioned.csv"
    assert csv_path.exists(), f"未找到 {csv_path}，请先跑 scripts/33 --full --mode regime_conditioned"
    df = pd.read_csv(csv_path)
    df = df.sort_values("capture_pct")
    logger.info(f"全 365 天 capture 分布：min={df['capture_pct'].min():.1f}%, "
                f"p25={df['capture_pct'].quantile(0.25):.1f}%, "
                f"median={df['capture_pct'].median():.1f}%, "
                f"p75={df['capture_pct'].quantile(0.75):.1f}%, "
                f"max={df['capture_pct'].max():.1f}%")

    worst = df.head(args.top).copy()
    logger.info(f"\nTop {args.top} worst days (lowest capture):")
    for _, row in worst.iterrows():
        logger.info(f"  day {int(row['day_idx'])}: DP ¥{row['vfa_dp_revenue']:>10,.0f}  "
                    f"Oracle ¥{row['oracle_revenue']:>10,.0f}  "
                    f"capture {row['capture_pct']:>6.1f}%")

    # 加载数据和分类器
    data = load_province(args.province)
    battery = BatteryConfig()
    dp = TensorDP(battery, config=DPConfig(delta_soc=0.005))
    quarters = split_walkforward(data)

    # 为每个 worst day 找到对应 quarter 的分类器
    def _quarter_of(d):
        for qi, (qs, qe) in enumerate(quarters):
            if qs <= d < qe:
                return qi
        return -1

    unique_quarters = sorted(set(_quarter_of(int(r['day_idx'])) for _, r in worst.iterrows()))
    quarter_classifiers = {}
    for qi in unique_quarters:
        qs, qe = quarters[qi]
        logger.info(f"\n训练 Q{qi+1} classifier (train end={qs})...")
        clf = RegimeClassifier(n_regimes=12)
        clf.fit(data, train_day_end=qs)
        quarter_classifiers[qi] = clf

    # 诊断每个 worst day
    all_diag = []
    for _, row in worst.iterrows():
        d = int(row['day_idx'])
        qi = _quarter_of(d)
        clf = quarter_classifiers[qi]
        diag = diagnose_one_day(data, clf, dp, d)
        all_diag.append(diag)

        logger.info(f"\n{'='*80}")
        logger.info(f"DAY {d}  capture={diag['capture']:+.1f}%  "
                    f"DP=¥{diag['dp_revenue']:,.0f}  Oracle=¥{diag['oracle_revenue']:,.0f}")
        logger.info(f"{'='*80}")

        # Regime 预测
        logger.info(f"  Regime 预测 top3:")
        for reg, p in diag['regime_probs_top3']:
            logger.info(f"    regime {reg}: {p*100:5.1f}%")

        # 预测误差
        logger.info(f"  预测 vs 实际 RT 统计:")
        logger.info(f"    实际 RT:     mean={diag['actual_rt_mean']:.1f}  "
                    f"std={diag['actual_rt_std']:.1f}  "
                    f"range={diag['actual_rt_range']:.1f}")
        logger.info(f"    预测误差:    mean={diag['rt_gap_mean']:+.1f}  "
                    f"std={diag['rt_gap_std']:.1f}  "
                    f"max|gap|={diag['rt_gap_max_abs']:.1f}")

        # SoC 边界
        logger.info(f"  SoC 轨迹:")
        logger.info(f"    撞 SoC_min 次数: {diag['soc_hit_min_steps']}/96")
        logger.info(f"    撞 SoC_max 次数: {diag['soc_hit_max_steps']}/96")
        logger.info(f"    最终 SoC: {diag['dp_final_soc']:.3f}")

        # 决策差异：放电总量 & 充电总量
        dp_charge = np.abs(np.minimum(diag['dp_powers'], 0)).sum() * 0.25
        dp_dis = np.maximum(diag['dp_powers'], 0).sum() * 0.25
        o_charge = np.abs(np.minimum(diag['oracle_powers'], 0)).sum() * 0.25
        o_dis = np.maximum(diag['oracle_powers'], 0).sum() * 0.25
        logger.info(f"  能量吞吐 (MWh):")
        logger.info(f"    DP:     放电 {dp_dis:>6.1f}   充电 {dp_charge:>6.1f}")
        logger.info(f"    Oracle: 放电 {o_dis:>6.1f}   充电 {o_charge:>6.1f}")

    # 聚合保存
    diag_df = pd.DataFrame([{
        "day_idx": d["day_idx"],
        "capture": d["capture"],
        "dp_revenue": d["dp_revenue"],
        "oracle_revenue": d["oracle_revenue"],
        "actual_rt_mean": d["actual_rt_mean"],
        "actual_rt_std": d["actual_rt_std"],
        "actual_rt_range": d["actual_rt_range"],
        "rt_gap_mean": d["rt_gap_mean"],
        "rt_gap_std": d["rt_gap_std"],
        "rt_gap_max_abs": d["rt_gap_max_abs"],
        "soc_hit_min_steps": d["soc_hit_min_steps"],
        "soc_hit_max_steps": d["soc_hit_max_steps"],
        "regime_top1": d["regime_probs_top3"][0][0],
        "regime_top1_prob": d["regime_probs_top3"][0][1],
    } for d in all_diag])
    diag_df.to_csv(OUTPUT / "worst_days_diagnose.csv", index=False)
    logger.info(f"\n保存诊断到 {OUTPUT}/worst_days_diagnose.csv")

    # 整体发现
    logger.info(f"\n{'='*80}")
    logger.info("整体失败模式归因")
    logger.info(f"{'='*80}")
    logger.info(f"  worst {len(worst)} 天 avg capture:        {worst['capture_pct'].mean():.1f}%")
    logger.info(f"  worst 天 avg RT gap (预测 vs 实际):       {diag_df['rt_gap_mean'].abs().mean():.1f}")
    logger.info(f"  worst 天 avg RT std (当天波动):           {diag_df['actual_rt_std'].mean():.1f}")
    logger.info(f"  worst 天 avg soc_hit_min+max:            "
                f"{(diag_df['soc_hit_min_steps']+diag_df['soc_hit_max_steps']).mean():.1f}/96")

    # 对比全年平均 RT std
    all_rt_stds = np.array([data.rt_prices[d].std() for d in range(data.n_days)])
    logger.info(f"  全年 RT std 平均:                        {all_rt_stds.mean():.1f}")
    logger.info(f"  全年 RT std p95:                         {np.percentile(all_rt_stds, 95):.1f}")
    logger.info(f"  worst 天 std vs 全年平均:                "
                f"{(diag_df['actual_rt_std'].mean() / all_rt_stds.mean() - 1)*100:+.0f}%")


if __name__ == "__main__":
    main()
