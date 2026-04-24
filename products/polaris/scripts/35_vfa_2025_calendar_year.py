"""
2025 自然年（2025-01-01 ~ 2025-12-31）Polaris V2 回测
======================================================

对车总 / 中广核山东汇报用 · 干净的"2025 全年"数字。

4 省完整跑一遍，和之前的 "2025-04 ~ 2026-04 rolling" 数字做对比。

Usage:
    PYTHONPATH=. python3 scripts/35_vfa_2025_calendar_year.py
    PYTHONPATH=. python3 scripts/35_vfa_2025_calendar_year.py --province shandong
"""
from __future__ import annotations

import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

from config import BatteryConfig
from optimization.milp.data_loader import load_province
from optimization.milp.scenario_generator import RegimeClassifier
from optimization.vfa_dp.tensor_dp import TensorDP, DPConfig
from oracle.lp_oracle import solve_day
from scripts import __init__ as _  # noqa


OUTPUT = Path("runs/vfa_dp_2025")
OUTPUT.mkdir(parents=True, exist_ok=True)


def get_2025_test_days_and_quarters(data):
    """返回 2025 自然年的 test_days 列表 + 4 个 quarter 边界"""
    # 找 2025-01-01 与 2026-01-01 对应的 day index
    day_start_dates = data.df.index[::96].normalize()
    year_2025_start_ts = pd.Timestamp("2025-01-01")
    year_2026_start_ts = pd.Timestamp("2026-01-01")

    # 第一个 >= 2025-01-01 的 day index
    match_25 = (day_start_dates >= year_2025_start_ts)
    if not match_25.any():
        raise RuntimeError("数据中没有 2025 年")
    year_start_day = int(match_25.argmax())

    match_26 = (day_start_dates >= year_2026_start_ts)
    if not match_26.any():
        year_end_day = len(day_start_dates)
    else:
        year_end_day = int(match_26.argmax())

    test_days = list(range(year_start_day, year_end_day))

    # 按季度切：Q1 (Jan-Mar), Q2 (Apr-Jun), Q3 (Jul-Sep), Q4 (Oct-Dec)
    # 精确按月份切
    quarter_bounds = []
    q_starts_ts = [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-04-01"),
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-10-01"),
        pd.Timestamp("2026-01-01"),
    ]
    for i in range(4):
        qs_match = (day_start_dates >= q_starts_ts[i])
        qs = int(qs_match.argmax()) if qs_match.any() else year_start_day
        qe_match = (day_start_dates >= q_starts_ts[i + 1])
        qe = int(qe_match.argmax()) if qe_match.any() else year_end_day
        quarter_bounds.append((qs, qe))

    return test_days, quarter_bounds


def build_scenarios_regime_conditioned(classifier, data, target_day, subsample=500):
    """与 scripts/33 完全同款场景生成"""
    probs, _ = classifier.predict_regime_probs(data, target_day)
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

    if subsample is not None and subsample < n_train:
        idx_sorted = np.argsort(-day_weights)[:subsample]
        idx_sorted = np.sort(idx_sorted)
        rt_sub = train_rt[idx_sorted]
        w_sub = day_weights[idx_sorted]
        w_sub = w_sub / w_sub.sum()
    else:
        rt_sub = train_rt
        w_sub = day_weights

    price_scen = rt_sub.T
    price_probs = np.tile(w_sub[None, :], (96, 1))
    return price_scen, price_probs


def run_2025_province(province: str, delta_soc: float = 0.005, R_cap: int = 500,
                      scale_mw_ratio: float = 1.0):
    """跑 2025 自然年 backtest

    scale_mw_ratio: 电池规模缩放倍数（1.0 = 200MW/400MWh; 0.5 = 100MW/200MWh）
    """
    data = load_province(province)
    battery = BatteryConfig()
    logger.info(f"\n{'='*70}\n  {province.upper()} — 2025 自然年 backtest\n{'='*70}")
    logger.info(f"  数据: {data.df.index.min().date()} ~ {data.df.index.max().date()}, {data.n_days} 天")

    test_days, quarter_bounds = get_2025_test_days_and_quarters(data)
    logger.info(f"  2025 test range: day {test_days[0]} ~ {test_days[-1]} ({len(test_days)} 天)")
    logger.info(f"  对应日期: {data.df.index[test_days[0]*96].date()} ~ {data.df.index[test_days[-1]*96 + 95].date()}")

    # 初始化 DP
    dp = TensorDP(battery, config=DPConfig(delta_soc=delta_soc))
    logger.info(f"  TensorDP: S={dp.S}, P={dp.P}")

    # 每季度一个 classifier，训练集 = 该季度起始日前的所有数据
    classifiers = {}
    for qi, (qs, qe) in enumerate(quarter_bounds):
        logger.info(f"\n  训练 Q{qi+1} classifier (train_end={qs}, 即 {data.df.index[qs*96].date()} 之前)")
        t0 = time.time()
        clf = RegimeClassifier(n_regimes=12)
        clf.fit(data, train_day_end=qs)
        classifiers[qi] = clf
        logger.info(f"    done in {time.time()-t0:.1f}s")

    # 按天跑 backtest
    records = []
    t_start = time.time()
    for i, d in enumerate(test_days):
        # 确定属于哪个 quarter
        qi_this = None
        for qi, (qs, qe) in enumerate(quarter_bounds):
            if qs <= d < qe:
                qi_this = qi
                break

        classifier = classifiers[qi_this]
        price_scen, price_probs = build_scenarios_regime_conditioned(
            classifier, data, target_day=d, subsample=R_cap,
        )
        V = dp.backward_induction(price_scen, price_probs)

        actual_rt = data.rt_prices[d]
        sim = dp.forward_simulate(V, actual_rt, init_soc=0.5)

        oracle = solve_day(actual_rt, battery, init_soc=0.5)

        records.append({
            "day_idx": d,
            "date": data.df.index[d*96].date(),
            "quarter": qi_this + 1,
            "vfa_dp_revenue": sim["revenue_total"],
            "oracle_revenue": oracle["revenue"],
            "capture_pct": sim["revenue_total"] / oracle["revenue"] * 100 if oracle["revenue"] > 0 else 0,
        })

        if (i + 1) % 30 == 0 or i == 0:
            logger.info(f"    day {d} ({data.df.index[d*96].date()}, {i+1}/{len(test_days)}): "
                        f"DP=¥{sim['revenue_total']:>10,.0f}  Oracle=¥{oracle['revenue']:>10,.0f}  "
                        f"capture={records[-1]['capture_pct']:>5.1f}%")

    elapsed = time.time() - t_start
    df = pd.DataFrame(records)

    # 按季度汇总
    logger.info(f"\n  Total {len(test_days)} days in {elapsed:.1f}s ({elapsed/len(test_days):.2f}s/day)")
    logger.info(f"\n  按季度统计:")
    for qi in range(4):
        sub = df[df["quarter"] == qi + 1]
        if len(sub) == 0:
            continue
        dp_rev = sub["vfa_dp_revenue"].sum()
        oracle_rev = sub["oracle_revenue"].sum()
        capture = dp_rev / oracle_rev * 100 if oracle_rev > 0 else 0
        logger.info(f"    Q{qi+1} ({len(sub)} 天): DP=¥{dp_rev:>12,.0f} / Oracle=¥{oracle_rev:>12,.0f} = {capture:>5.1f}%")

    total_dp = df["vfa_dp_revenue"].sum()
    total_oracle = df["oracle_revenue"].sum()
    total_capture = total_dp / total_oracle * 100
    logger.info(f"\n  2025 全年（按 200MW/400MWh 电池 = 原回测参数）:")
    logger.info(f"    Polaris V2:  ¥{total_dp:>14,.0f}  ({total_capture:.2f}% of Oracle)")
    logger.info(f"    Oracle:      ¥{total_oracle:>14,.0f}")

    # 线性 scale 到 100MW/200MWh
    scale = 0.5
    logger.info(f"\n  按 100 MW / 200 MWh 线性折算:")
    logger.info(f"    Polaris V2:  ¥{total_dp * scale:>14,.0f}")
    logger.info(f"    Oracle:      ¥{total_oracle * scale:>14,.0f}")

    csv_path = OUTPUT / f"{province}_2025_calendar_year.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n  per-day 结果保存到 {csv_path}")

    return {
        "province": province,
        "days": len(test_days),
        "total_dp_200mw": total_dp,
        "total_oracle_200mw": total_oracle,
        "capture": total_capture,
        "total_dp_100mw": total_dp * scale,
        "total_oracle_100mw": total_oracle * scale,
        "per_day_df": df,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--province", default="shandong")
    parser.add_argument("--all", action="store_true", help="跑 4 省")
    parser.add_argument("--delta", type=float, default=0.005)
    parser.add_argument("--R", type=int, default=500)
    args = parser.parse_args()

    provinces = ["shandong", "shanxi", "guangdong", "gansu"] if args.all else [args.province]

    all_results = []
    for prov in provinces:
        try:
            r = run_2025_province(prov, delta_soc=args.delta, R_cap=args.R)
            all_results.append(r)
        except Exception as e:
            logger.exception(f"{prov} failed: {e}")

    # 汇总所有省
    if len(all_results) > 1:
        logger.info(f"\n{'='*80}\n  4 省 2025 自然年汇总\n{'='*80}")
        logger.info(f"  {'省':<10}{'Polaris @200MW':>20}{'Oracle @200MW':>20}{'Capture':>10}{'@100MW':>18}")
        for r in all_results:
            logger.info(f"  {r['province']:<10}"
                        f"¥{r['total_dp_200mw']:>18,.0f}  "
                        f"¥{r['total_oracle_200mw']:>18,.0f}  "
                        f"{r['capture']:>8.2f}%  "
                        f"¥{r['total_dp_100mw']:>15,.0f}")

        total_dp = sum(r['total_dp_200mw'] for r in all_results)
        total_oracle = sum(r['total_oracle_200mw'] for r in all_results)
        logger.info(f"  {'合计':<10}"
                    f"¥{total_dp:>18,.0f}  "
                    f"¥{total_oracle:>18,.0f}  "
                    f"{total_dp/total_oracle*100:>8.2f}%  "
                    f"¥{total_dp*0.5:>15,.0f}")


if __name__ == "__main__":
    main()
