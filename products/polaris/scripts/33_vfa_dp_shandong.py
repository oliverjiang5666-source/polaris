"""
Track 2 Day 3: Tensor DP on 山东（对比 Regime V3）
==================================================

核心问题：Lee-Sun 2025 的 stagewise stochastic DP 在山东数据上比 Regime V3 强多少？

算法对比：
  Regime V3:    expected price DP（先取期望再 DP）—— 当前 capture 64.8%
  Tensor DP:    stagewise stochastic DP（R 个支撑点 → E_r[max_a Q]）

场景生成（V1 simple）：
  对每个时段 t (0..95)：
    支撑点 = 训练集所有天的第 t 时段 RT 价
    权重 = 1/n_train_days（均匀）
  这不利用 regime，是纯 empirical stagewise。后续 V2 会加 regime conditioning。

walk-forward 按季度 retrain（和 scripts/22_regime_v3_allprov.py 对齐）。

Usage:
    # Smoke: 10 天
    PYTHONPATH=. python3 scripts/33_vfa_dp_shandong.py --smoke

    # Full: 365 天
    PYTHONPATH=. python3 scripts/33_vfa_dp_shandong.py --full
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from optimization.milp.data_loader import load_province, split_walkforward
from optimization.milp.scenario_generator import RegimeClassifier
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig
from oracle.lp_oracle import solve_day


OUTPUT = Path("runs/vfa_dp")
OUTPUT.mkdir(parents=True, exist_ok=True)


def build_stagewise_scenarios_simple(
    train_rt: np.ndarray,   # shape (n_train_days, 96)
    subsample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    V1 场景：每时段 t 用训练集所有天的第 t 时段 RT 价作为支撑点。均匀权重。
    """
    if subsample is not None and subsample < len(train_rt):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_rt), size=subsample, replace=False)
        sub = train_rt[idx]
    else:
        sub = train_rt

    price_scenarios = sub.T                            # shape (96, n_train_days)
    R = sub.shape[0]
    price_probs = np.ones((96, R)) / R

    return price_scenarios, price_probs


def build_stagewise_scenarios_regime_conditioned(
    classifier: RegimeClassifier,
    province_data,
    target_day: int,
    subsample: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    V2 场景：regime-conditioned stagewise。

    每时段 t 的支撑点 = 训练集每天的第 t 时段 RT 价（同 V1）
    但权重按该天所属 regime 的预测概率加权：
        day_weight[d] = regime_probs[label(d)] / count(label(d) in train)

    这样：
      - 支撑点覆盖 = 全部训练天（R 很大）
      - 权重分布 = 集中在"和明天 regime 相似"的历史天
      - 每时段独立估计 E[max_a Q(a, price_t)]

    对比 Regime V3:
      V3 先把 regime_probs × profile_t 合并成一个 expected_price_t，然后 DP
      V2 保留每个训练天的实际价格作为支撑点，DP 里做 E_r[max_a Q]
      若 max_a Q(a, price) 对 price 是非线性（凸 / 凹）的，V2 ≥ V3（Jensen）
    """
    # 预测明天 regime 概率
    probs, _ = classifier.predict_regime_probs(province_data, target_day)
    probs = probs / probs.sum()

    # 训练集 labels & RT
    train_labels = classifier.train_labels
    n_train = len(train_labels)
    train_rt = province_data.rt_prices[:n_train]

    # 每个 regime 的天数
    n_reg = classifier.n_regimes
    counts = np.array([(train_labels == c).sum() for c in range(n_reg)])

    # 每天权重 = probs[c] / counts[c]
    day_weights = np.zeros(n_train)
    for d in range(n_train):
        reg_d = train_labels[d]
        if counts[reg_d] > 0:
            day_weights[d] = probs[reg_d] / counts[reg_d]
    # 归一化（兜底）
    day_weights = day_weights / day_weights.sum()

    # 可选 subsample（按权重）
    if subsample is not None and subsample < n_train:
        rng = np.random.default_rng(42 + target_day)
        # 为了保留信号，对 weight 最大的 subsample 个天截断
        idx_sorted = np.argsort(-day_weights)[:subsample]
        idx_sorted = np.sort(idx_sorted)  # 保持时间顺序（可选）
        rt_sub = train_rt[idx_sorted]
        w_sub = day_weights[idx_sorted]
        w_sub = w_sub / w_sub.sum()
    else:
        rt_sub = train_rt
        w_sub = day_weights

    price_scenarios = rt_sub.T                              # (96, R)
    price_probs = np.tile(w_sub[None, :], (96, 1))          # (96, R)

    return price_scenarios, price_probs


def run_day(
    dp: TensorDP,
    scenario_mode: str,
    train_rt_window: np.ndarray,
    actual_rt: np.ndarray,
    init_soc: float = 0.5,
    R_cap: int | None = None,
    classifier: RegimeClassifier | None = None,
    province_data=None,
    target_day: int | None = None,
) -> dict:
    """单日 stagewise stochastic DP."""
    if scenario_mode == "simple":
        price_scen, price_probs = build_stagewise_scenarios_simple(
            train_rt_window, subsample=R_cap)
    elif scenario_mode == "regime_conditioned":
        assert classifier is not None and province_data is not None and target_day is not None
        price_scen, price_probs = build_stagewise_scenarios_regime_conditioned(
            classifier, province_data, target_day, subsample=R_cap)
    else:
        raise ValueError(f"unknown scenario_mode: {scenario_mode}")

    V = dp.backward_induction(price_scen, price_probs)
    sim = dp.forward_simulate(V, actual_rt, init_soc=init_soc)
    return {
        "revenue": sim["revenue_total"],
        "powers": sim["powers"],
        "soc_trajectory": sim["soc_trajectory"],
        "final_soc": sim["final_soc"],
        "R_used": price_scen.shape[1],
    }


def walk_forward_vfa_dp(
    province: str = "shandong",
    test_days: list[int] | None = None,
    delta_soc: float = 0.005,
    R_cap: int | None = 500,
    init_soc_daily: float = 0.5,
    verbose: bool = True,
    scenario_mode: str = "simple",   # "simple" / "regime_conditioned"
    final_soc_penalty: float = 0.0,
    final_soc_target: float = 0.5,
) -> pd.DataFrame:
    """Walk-forward backtest"""
    data = load_province(province)
    battery = BatteryConfig()
    quarters = split_walkforward(data)

    if test_days is None:
        qs, qe = quarters[-1]
        test_days = list(range(qs, qe))

    dp_config = DPConfig(
        delta_soc=delta_soc,
        final_soc_penalty=final_soc_penalty,
        final_soc_target=final_soc_target,
    )
    dp = TensorDP(battery, config=dp_config)
    logger.info(f"TensorDP grid: S={dp.S}, P={dp.P}, R_cap={R_cap}, mode={scenario_mode}, "
                f"final_soc_penalty={final_soc_penalty}, target={final_soc_target}")

    # 为 regime_conditioned 模式：按 quarter 重训分类器
    quarter_classifiers: dict[int, RegimeClassifier] = {}
    if scenario_mode == "regime_conditioned":
        # 哪天属于哪个 quarter
        def _find_quarter(d):
            for qi, (qs, qe) in enumerate(quarters):
                if qs <= d < qe:
                    return qi
            return -1

        unique_quarters = sorted(set(_find_quarter(d) for d in test_days))
        for qi in unique_quarters:
            qs, qe = quarters[qi]
            logger.info(f"  Training classifier for Q{qi+1} (train end = {qs})...")
            t0 = time.time()
            clf = RegimeClassifier(n_regimes=12)
            clf.fit(data, train_day_end=qs)
            quarter_classifiers[qi] = clf
            logger.info(f"    done in {time.time() - t0:.1f}s")

    records = []
    t_start = time.time()

    for i, d in enumerate(test_days):
        train_rt = data.rt_prices[:d]
        actual_rt = data.rt_prices[d]

        classifier = None
        if scenario_mode == "regime_conditioned":
            for qi, (qs, qe) in enumerate(quarters):
                if qs <= d < qe:
                    classifier = quarter_classifiers[qi]
                    break

        t0 = time.time()
        out = run_day(
            dp, scenario_mode=scenario_mode,
            train_rt_window=train_rt, actual_rt=actual_rt,
            init_soc=init_soc_daily, R_cap=R_cap,
            classifier=classifier, province_data=data, target_day=d,
        )
        dp_time = time.time() - t0

        oracle = solve_day(actual_rt, battery, init_soc=init_soc_daily)

        records.append({
            "day_idx": d,
            "vfa_dp_revenue": out["revenue"],
            "oracle_revenue": oracle["revenue"],
            "capture_pct": out["revenue"] / oracle["revenue"] * 100 if oracle["revenue"] > 0 else 0,
            "R_used": out["R_used"],
            "final_soc": out["final_soc"],
            "dp_time": dp_time,
            "mode": scenario_mode,
        })

        if verbose and (i < 3 or (i + 1) % 20 == 0 or i == len(test_days) - 1):
            logger.info(f"  day {d} ({i+1}/{len(test_days)}): "
                        f"DP ¥{out['revenue']:>10,.0f}  "
                        f"Oracle ¥{oracle['revenue']:>10,.0f}  "
                        f"capture {records[-1]['capture_pct']:>5.1f}%  "
                        f"R={out['R_used']}  "
                        f"t={dp_time:.2f}s")

    df = pd.DataFrame(records)
    total_elapsed = time.time() - t_start
    logger.info(f"\nTotal walk-forward: {len(test_days)} days in {total_elapsed:.1f}s "
                f"({total_elapsed/len(test_days):.2f}s/day)")

    return df


def summarize(df: pd.DataFrame, regime_v3_ref: dict | None = None):
    total_dp = df["vfa_dp_revenue"].sum()
    total_oracle = df["oracle_revenue"].sum()
    capture = total_dp / total_oracle * 100 if total_oracle > 0 else 0

    logger.info(f"\n{'=' * 70}")
    logger.info(f"Tensor DP (Lee-Sun 2025) 山东 backtest 汇总")
    logger.info(f"{'=' * 70}")
    logger.info(f"  天数:            {len(df)}")
    logger.info(f"  DP 总收入:       ¥{total_dp:>14,.0f}")
    logger.info(f"  Oracle 总收入:   ¥{total_oracle:>14,.0f}")
    logger.info(f"  Capture:         {capture:>14.2f}%")

    if regime_v3_ref is not None:
        regime_v3_total = regime_v3_ref["revenue"]
        regime_v3_capture = regime_v3_ref["capture"]
        if len(df) < 365:
            # 按比例外推
            scale = 365 / len(df)
            extrapolated = total_dp * scale
            logger.info(f"\n  外推到 365 天:   ¥{extrapolated:>14,.0f}")
            logger.info(f"  Regime V3 (365天): ¥{regime_v3_total:>14,.0f}  capture {regime_v3_capture}%")
            delta_abs = extrapolated - regime_v3_total
            delta_pct = delta_abs / regime_v3_total * 100
            logger.info(f"  Delta vs Regime V3: {delta_abs:>+14,.0f}  ({delta_pct:+.1f}%)")
        else:
            delta_abs = total_dp - regime_v3_total
            delta_pct = delta_abs / regime_v3_total * 100
            logger.info(f"  Regime V3 全年:  ¥{regime_v3_total:>14,.0f}  capture {regime_v3_capture}%")
            logger.info(f"  Delta vs Regime V3: {delta_abs:>+14,.0f}  ({delta_pct:+.1f}%)")

    # Per-day distribution
    logger.info(f"\n  per-day capture 分布:")
    logger.info(f"    min:   {df['capture_pct'].min():>6.1f}%")
    logger.info(f"    p25:   {df['capture_pct'].quantile(0.25):>6.1f}%")
    logger.info(f"    med:   {df['capture_pct'].median():>6.1f}%")
    logger.info(f"    p75:   {df['capture_pct'].quantile(0.75):>6.1f}%")
    logger.info(f"    max:   {df['capture_pct'].max():>6.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="10 天 smoke")
    parser.add_argument("--full", action="store_true", help="全测试期")
    parser.add_argument("--delta", type=float, default=0.005, help="δ_soc")
    parser.add_argument("--R", type=int, default=500, help="每时段支撑点数")
    parser.add_argument("--province", default="shandong")
    parser.add_argument("--mode", default="simple",
                        choices=["simple", "regime_conditioned"],
                        help="场景生成模式")
    parser.add_argument("--final-soc-penalty", type=float, default=0.0,
                        help="末态 SoC 惩罚（V_T(s) = -λ × (target - s)^+ ）")
    parser.add_argument("--final-soc-target", type=float, default=0.3,
                        help="末态 SoC 目标")
    args = parser.parse_args()

    data = load_province(args.province)
    quarters = split_walkforward(data)

    if args.smoke or not args.full:
        qs, qe = quarters[-1]
        test_days = list(range(qs, min(qs + 10, qe)))
        tag = f"smoke10d_{args.mode}"
    else:
        test_days = []
        for qs, qe in quarters:
            test_days.extend(range(qs, qe))
        tag = f"full{len(test_days)}d_{args.mode}"

    logger.info(f"=== Tensor DP 山东 backtest ({tag}) ===")
    logger.info(f"  test days: {test_days[0]} ~ {test_days[-1]}, total {len(test_days)}")
    logger.info(f"  δ_soc={args.delta}, R={args.R}, mode={args.mode}")

    df = walk_forward_vfa_dp(
        province=args.province,
        test_days=test_days,
        delta_soc=args.delta,
        R_cap=args.R,
        scenario_mode=args.mode,
        final_soc_penalty=args.final_soc_penalty,
        final_soc_target=args.final_soc_target,
    )

    csv_path = OUTPUT / f"{args.province}_vfa_dp_{tag}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n结果保存到 {csv_path}")

    # Regime V3 reference
    REGIME_V3_REF = {
        "shandong": {"revenue": 53_810_000, "capture": 64.8},
        "shanxi":   {"revenue": 62_440_000, "capture": 63.2},
        "guangdong": {"revenue": 35_590_000, "capture": 56.2},
        "gansu":     {"revenue": 32_420_000, "capture": 41.1},
    }
    summarize(df, regime_v3_ref=REGIME_V3_REF.get(args.province))


if __name__ == "__main__":
    main()
