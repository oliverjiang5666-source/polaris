"""
Polaris 山东 · Script 05 — 2025 全年回测 (加 L3-A 跨日 TVF)
===========================================================

和 Script 03 一模一样的 walk-forward 框架, 唯一区别:
  Script 03: V[T, :] = 0  (每天独立)
  Script 05: V[T, :] = V[1, :] from 前一天 (rolling horizon TVF)

理论上 TVF 应让储能在晚间保留 SoC 以备次日高价, 整体 capture 应升.
实测看数字.

Usage:
    PYTHONPATH=. python3 products/polaris_shandong/scripts/05_backtest_2025_tvf.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
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


OUTPUT = ROOT / "runs" / "polaris_shandong" / "2025_backtest"
OUTPUT.mkdir(parents=True, exist_ok=True)


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


def get_2025_days_quarters(data):
    day_starts = data.df.index[::96].normalize()
    y25 = pd.Timestamp("2025-01-01")
    y26 = pd.Timestamp("2026-01-01")
    start = int((day_starts >= y25).argmax())
    end = int((day_starts >= y26).argmax()) if (day_starts >= y26).any() else len(day_starts)
    test_days = list(range(start, end))
    qs_bounds = []
    q_starts = [pd.Timestamp(f"2025-{m:02d}-01") for m in [1, 4, 7, 10]] + [pd.Timestamp("2026-01-01")]
    for i in range(4):
        qs = int((day_starts >= q_starts[i]).argmax())
        qe = int((day_starts >= q_starts[i + 1]).argmax()) if (day_starts >= q_starts[i + 1]).any() else end
        qs_bounds.append((qs, qe))
    return test_days, qs_bounds


def run_with_tvf(
    capacity_mw: float = 100.0,
    capacity_mwh: float = 200.0,
    delta_soc: float = 0.005,
    R_cap: int = 500,
    use_tvf: bool = True,
    tvf_discount: float = 0.3,
    tvf_normalize: str = "relative",
    soc_carry_enabled: bool = False,           # 改默认 False: 每天重置 0.5 避免放大囤电
):
    battery = BatteryConfig(capacity_mw=capacity_mw, capacity_mwh=capacity_mwh)
    data = load_province("shandong")
    test_days, qs_bounds = get_2025_days_quarters(data)

    # 每季度重训 classifier
    classifiers = {}
    for qi, (qs, _) in enumerate(qs_bounds):
        clf = RegimeClassifier(n_regimes=12)
        clf.fit(data, train_day_end=qs)
        classifiers[qi] = clf

    base_cfg = DPConfig(delta_soc=delta_soc)
    dp = TensorDP(battery, base_cfg)
    logger.info(
        f"TensorDP: S={dp.S}, P={dp.P}, TVF={'ON' if use_tvf else 'OFF'}  "
        f"discount={tvf_discount}, normalize={tvf_normalize}, soc_carry={soc_carry_enabled}"
    )

    records = []
    t_start = time.time()

    prev_v1 = None
    soc_carry = 0.5

    for i, d in enumerate(test_days):
        qi_this = 0
        for qi, (qs, qe) in enumerate(qs_bounds):
            if qs <= d < qe:
                qi_this = qi
                break
        clf = classifiers[qi_this]

        scen, probs = build_scenarios_regime_conditioned(clf, data, d, R_cap)

        cfg_today = DPConfig(
            delta_soc=delta_soc,
            external_tvf=prev_v1 if use_tvf else None,
            tvf_discount=tvf_discount,
            tvf_normalize=tvf_normalize,
            tvf_ref_soc=0.5,
        )
        dp_today = TensorDP(battery, cfg_today)
        V = dp_today.backward_induction(scen, probs)

        init_soc = soc_carry if (use_tvf and soc_carry_enabled) else 0.5
        sim = dp_today.forward_simulate(V, data.rt_prices[d], init_soc=init_soc)
        power_96 = sim["powers"]
        rt_96 = data.rt_prices[d]

        rev = float((power_96 * rt_96 * 0.25).sum() - 2.0 * np.abs(power_96).sum() * 0.25)
        oracle = solve_day(rt_96, battery, init_soc=init_soc)
        oracle_rev = oracle["revenue"]

        records.append({
            "day_idx": d,
            "date": str(data.df.index[d*96].date()),
            "quarter": qi_this + 1,
            "revenue": rev,
            "oracle_rev": oracle_rev,
            "capture": rev / oracle_rev * 100 if oracle_rev > 0 else 0,
            "init_soc": init_soc,
            "final_soc": sim["final_soc"],
        })

        if use_tvf:
            prev_v1 = V[1, :].copy()
            if soc_carry_enabled:
                soc_carry = sim["final_soc"]

        if (i + 1) % 30 == 0 or i == 0:
            logger.info(
                f"    day {d} ({data.df.index[d*96].date()}, {i+1}/{len(test_days)}): "
                f"rev=¥{rev:>8,.0f} oracle=¥{oracle_rev:>8,.0f} cap={records[-1]['capture']:>5.1f}% "
                f"init_soc={init_soc:.2f} → {sim['final_soc']:.2f}"
            )

    elapsed = time.time() - t_start
    df = pd.DataFrame(records)
    total = df["revenue"].sum()
    total_oracle = df["oracle_rev"].sum()
    logger.info(f"\n  {len(test_days)} days in {elapsed:.0f}s")
    logger.info(f"  Total revenue: ¥{total:,.0f}")
    logger.info(f"  Oracle total:  ¥{total_oracle:,.0f}")
    logger.info(f"  Capture:       {total/total_oracle*100:.2f}%")

    tag = "tvf" if use_tvf else "no_tvf"
    csv_path = OUTPUT / f"shandong_2025_100mw_{tag}.csv"
    df.to_csv(csv_path, index=False)
    return df, total, total_oracle


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("  TVF OFF (baseline)")
    logger.info("=" * 70)
    df_off, tot_off, oracle_off = run_with_tvf(use_tvf=False)

    logger.info("\n" + "=" * 70)
    logger.info("  TVF ON (rolling horizon)")
    logger.info("=" * 70)
    df_on, tot_on, oracle_on = run_with_tvf(use_tvf=True)

    logger.info("\n" + "=" * 70)
    logger.info("  L3-A TVF A/B 汇总 (2025 全年, 100MW/200MWh)")
    logger.info("=" * 70)
    logger.info(f"  TVF OFF: ¥{tot_off:>14,.0f}  capture {tot_off/oracle_off*100:>5.2f}%")
    logger.info(f"  TVF ON:  ¥{tot_on:>14,.0f}  capture {tot_on/oracle_on*100:>5.2f}%")
    gain = tot_on - tot_off
    logger.info(f"  TVF gain: ¥{gain:+,.0f} ({gain/tot_off*100:+.2f}%)")
    logger.info(f"  Capture improvement: {(tot_on/oracle_on - tot_off/oracle_off)*100:+.2f} pp")
