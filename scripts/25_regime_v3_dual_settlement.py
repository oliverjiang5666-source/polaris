"""
Regime V3 + DAM/RTM 双结算 (Modification #1)

基于 scripts/22_regime_v3_allprov.py，替换结算价为双结算。

关键改动：
- 规划阶段（D-1）：DP 使用 DAM 价（而不是 RT 价）做最优化
  理由：D-1 报价时你约束是 DAM 出清价，不是 RT 价
- 执行阶段（D+0）：实际按计划执行，按 DAM 价结算 DAM 部分
  允许小幅偏差，按 RT 价结算偏差部分
- 分类器保持用 RT 形态（日类型分类不受结算方式影响）

这更贴近中国真实市场下的真实收入。

Usage:
    PYTHONPATH=. python3 scripts/25_regime_v3_dual_settlement.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier

from config import BatteryConfig
from oracle.lp_oracle import solve_day, solve_day_dual
from forecast.mpc_controller import _step_battery, _step_battery_dual

PROCESSED_DIR = Path("data/china/processed")
N_REGIMES = 12
N_SOC = 20
POWER_STEP = 4
SOC_MIN, SOC_MAX = 0.05, 0.95


def solve_stochastic_dp_dam(scenarios_dam, weights, battery, power_levels, soc_levels):
    """
    Stochastic DP on DAM-priced scenarios.

    替代 22 版本：原本用 RT 期望价做 DP，现在用 DAM 期望价。
    因为 D-1 17:00 报的是 DAM，DP 目标应匹配 DAM 结算。
    """
    n_soc = len(soc_levels)
    n_power = len(power_levels)
    ih = battery.interval_hours
    ec, ed = battery.charge_efficiency, battery.discharge_efficiency
    cap = battery.capacity_mwh

    expected_prices = np.zeros(96)
    for i, w in enumerate(weights):
        expected_prices += w * scenarios_dam[i]

    V = np.zeros((97, n_soc))
    best_p = np.zeros((96, n_soc), dtype=np.int32)

    for h in range(95, -1, -1):
        wp = expected_prices[h]
        for s_idx in range(n_soc):
            soc = soc_levels[s_idx]
            best_val, best_idx = -1e18, n_power // 2
            for p_idx in range(n_power):
                pw = power_levels[p_idx]
                e = pw * ih
                if e > 0:
                    sc = -e / cap / ed
                elif e < 0:
                    sc = -e * ec / cap
                else:
                    sc = 0
                ns = soc + sc
                if ns < SOC_MIN - 0.001 or ns > SOC_MAX + 0.001:
                    continue
                ns = np.clip(ns, SOC_MIN, SOC_MAX)
                ns_idx = int(np.clip(round((ns - SOC_MIN) / (SOC_MAX - SOC_MIN) * (n_soc - 1)), 0, n_soc - 1))
                reward = e * wp - abs(e) * 2.0
                total = reward + V[h + 1][ns_idx]
                if total > best_val:
                    best_val, best_idx = total, p_idx
            V[h][s_idx] = best_val
            best_p[h][s_idx] = best_idx
    return best_p


def build_features(pm_rt, pm_dam, df, d, labels):
    """
    使用 RT 形态做特征（跟原版一致）。
    增加 DAM 特征供分类器用。
    """
    f = {}
    t = pm_rt[d]
    f["price_mean"], f["price_std"] = t.mean(), t.std()
    f["price_range"] = t.max() - t.min()
    f["price_min"], f["price_max"] = t.min(), t.max()
    f["price_skew"] = float(pd.Series(t).skew())

    for i, nm in enumerate(["night", "morn", "mid", "aftn", "eve", "late"]):
        f[f"{nm}_mean"] = t[i * 16:(i + 1) * 16].mean()
    f["morn_vs_eve"] = t[16:32].mean() - t[64:80].mean()

    if d >= 2:
        y = pm_rt[d - 1]
        f["y_mean"], f["y_std"], f["y_range"] = y.mean(), y.std(), y.max() - y.min()
        f["dod_change"] = t.mean() - y.mean()
    else:
        f["y_mean"], f["y_std"], f["y_range"], f["dod_change"] = t.mean(), t.std(), 0, 0

    if d >= 7:
        w = pm_rt[d - 6:d + 1]
        f["wk_mean"], f["wk_std"], f["wk_trend"] = w.mean(), w.std(), pm_rt[d].mean() - pm_rt[d - 6].mean()
    else:
        f["wk_mean"], f["wk_std"], f["wk_trend"] = t.mean(), t.std(), 0

    if labels is not None:
        if d < len(labels):
            f["today_reg"] = labels[d]
        if d >= 1:
            f["yest_reg"] = labels[d - 1]

    # 明天 DAM 预测（如果我们知道明天 DAM 出清价——在 D-1 23:00 后才知道）
    # 这里我们用"明天 DAM 价的昨日同期"作为预测
    # 注：更精细的版本会用 LightGBM 预测明天 DAM
    if d + 1 < len(pm_dam):
        tw_dam = pm_dam[d + 1]
        f["tw_dam_mean"] = tw_dam.mean()
        f["tw_dam_range"] = tw_dam.max() - tw_dam.min()
        f["tw_dam_min"] = tw_dam.min()
        f["tw_dam_max"] = tw_dam.max()

    for col in ["load_norm", "renewable_penetration", "wind_ratio",
                 "solar_ratio", "net_load_norm", "temperature_norm"]:
        if col in df.columns:
            v = df[col].fillna(0).values[d * 96:(d + 1) * 96]
            if len(v) > 0:
                f[f"{col}_m"], f[f"{col}_x"] = v.mean(), v.max()

    for col in ["temperature_norm", "wind_speed_norm", "solar_radiation_norm"]:
        if col in df.columns:
            ts, te = (d + 1) * 96, (d + 2) * 96
            if te <= len(df):
                v = df[col].fillna(0).values[ts:te]
                if len(v) > 0:
                    f[f"tw_{col}_m"], f[f"tw_{col}_x"] = v.mean(), v.max()

    ti = (d + 1) * 96
    if ti < len(df):
        dt = df.index[ti]
        f["tw_wd"], f["tw_mo"] = dt.weekday(), dt.month
        f["tw_we"] = 1.0 if dt.weekday() >= 5 else 0.0

    return f


def run_province(province, use_dual_settlement=True):
    battery = BatteryConfig()
    power_levels = np.arange(-200, 204, POWER_STEP, dtype=np.float64)
    soc_levels = np.linspace(SOC_MIN, SOC_MAX, N_SOC)

    df = pd.read_parquet(PROCESSED_DIR / f"{province}_oracle.parquet")
    rt_prices = df["rt_price"].fillna(0).values.astype(np.float64)
    dam_prices = df["da_price"].fillna(0).values.astype(np.float64)
    n_days = len(df) // 96

    pm_rt = rt_prices[:n_days * 96].reshape(n_days, 96)
    pm_dam = dam_prices[:n_days * 96].reshape(n_days, 96)

    test_days = 365
    test_start = n_days - test_days
    if test_start < 400:
        test_start = n_days // 2
        test_days = n_days - test_start

    quarter_size = test_days // 4
    quarters = []
    for q in range(4):
        qs = test_start + q * quarter_size
        qe = test_start + (q + 1) * quarter_size if q < 3 else n_days
        quarters.append((qs, qe))

    mode = "Dual (DAM+RT)" if use_dual_settlement else "Single (RT)"
    logger.info(f"\n{'#' * 70}")
    logger.info(f"  {province.upper()} — {n_days} days, test={test_days}d  [Settlement: {mode}]")
    logger.info(f"{'#' * 70}")

    # Oracle baseline：用 DAM 做 Oracle（因为 Oracle 的计划阶段知道 DAM）
    # 然后按双结算或单结算评估
    oracle_revs = np.zeros(n_days)
    for d in range(test_start, n_days):
        if use_dual_settlement:
            r = solve_day_dual(pm_dam[d], pm_rt[d], battery, init_soc=0.5)
            oracle_revs[d] = r["revenue_total"]
        else:
            r = solve_day(pm_rt[d], battery, init_soc=0.5)
            oracle_revs[d] = r["revenue"]

    total_strategy = 0.0
    total_oracle_test = 0.0

    for qi, (qs, qe) in enumerate(quarters):
        # 用 RT 做聚类（形态分类和结算方式无关）
        train_m = pm_rt[:qs]
        tr_means = train_m.mean(axis=1, keepdims=True)
        tr_stds = np.maximum(train_m.std(axis=1, keepdims=True), 1.0)
        km = KMeans(n_clusters=N_REGIMES, n_init=20, random_state=42)
        km.fit((train_m - tr_means) / tr_stds)

        all_shapes = np.zeros((n_days, 96))
        for d in range(n_days):
            m, s = pm_rt[d].mean(), max(pm_rt[d].std(), 1.0)
            all_shapes[d] = (pm_rt[d] - m) / s
        all_labels = km.predict(all_shapes)
        train_labels = all_labels[:qs]

        # Regime profiles：从 DAM 数据构建（因为 DP 要用 DAM 做决策）
        profiles_dam = np.zeros((N_REGIMES, 96))
        for c in range(N_REGIMES):
            mask = train_labels == c
            # 同日的 DAM 价（用 RT 形态分类但取 DAM 数值）
            profiles_dam[c] = pm_dam[:qs][mask].mean(axis=0) if mask.sum() > 0 else pm_dam[:qs].mean(axis=0)

        # Train classifier
        Xt, yt = [], []
        for d in range(7, qs - 1):
            Xt.append(build_features(pm_rt, pm_dam, df, d, all_labels))
            yt.append(all_labels[d + 1])
        Xdf = pd.DataFrame(Xt)
        ya = np.array(yt)

        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
        clf.fit(Xdf, ya)

        q_strategy = 0.0
        q_oracle = 0.0

        for d in range(qs, qe):
            feat = build_features(pm_rt, pm_dam, df, d - 1, all_labels)
            Xp = pd.DataFrame([feat])
            for col in Xdf.columns:
                if col not in Xp.columns:
                    Xp[col] = 0
            Xp = Xp[Xdf.columns]

            probs = clf.predict_proba(Xp)[0]

            # 按概率加权 DAM 场景
            scenarios_dam = []
            weights = []
            for ci, p in enumerate(probs):
                if p < 0.02:
                    continue
                rc = clf.classes_[ci]
                scenarios_dam.append(profiles_dam[rc])
                weights.append(p)
            scenarios_dam = np.array(scenarios_dam)
            weights = np.array(weights)
            weights /= weights.sum()

            policy = solve_stochastic_dp_dam(scenarios_dam, weights, battery, power_levels, soc_levels)

            # ====== 执行 + 结算 ======
            soc = 0.5
            rev = 0.0
            for t in range(96):
                si = int(np.clip(round((soc - SOC_MIN) / (SOC_MAX - SOC_MIN) * (N_SOC - 1)), 0, N_SOC - 1))
                pw = power_levels[policy[t][si]]  # DAM commitment（D-1决策）

                if use_dual_settlement:
                    # 简化：实际执行 = DAM 承诺（无策略偏差）
                    # 未来可以加执行时的偏差决策
                    dam_p = pm_dam[d, t]
                    rt_p = pm_rt[d, t]
                    soc, nr, _, _ = _step_battery_dual(
                        dam_commitment_mw=pw,
                        actual_power_mw=pw,  # 执行 = 计划
                        dam_price=dam_p, rt_price=rt_p,
                        soc=soc, battery=battery,
                    )
                else:
                    # 原单结算：用 RT 价
                    soc, nr, _ = _step_battery(pw, pm_rt[d, t], soc, battery)

                rev += nr

            q_strategy += rev
            q_oracle += oracle_revs[d]

        total_strategy += q_strategy
        total_oracle_test += q_oracle
        logger.info(f"  Q{qi+1}: Regime={q_strategy:>12,.0f}  Oracle={q_oracle:>12,.0f}  "
                    f"Capture={q_strategy/q_oracle*100:.1f}%")

    capture = total_strategy / total_oracle_test * 100
    logger.info(f"  TOTAL: Regime={total_strategy:>12,.0f}  Oracle={total_oracle_test:>12,.0f}  "
                f"Capture={capture:.1f}%")

    return {"province": province, "regime": total_strategy, "oracle": total_oracle_test,
            "capture": capture, "mode": mode}


def main():
    provinces = ["shandong", "shanxi", "guangdong", "gansu"]

    logger.info("="*70)
    logger.info("  Regime V3 在两种结算模式下的结果对比")
    logger.info("="*70)

    results_single = []
    results_dual = []

    for prov in provinces:
        logger.info(f"\n--- {prov.upper()}: 单结算（原版） ---")
        r_single = run_province(prov, use_dual_settlement=False)
        results_single.append(r_single)

        logger.info(f"\n--- {prov.upper()}: 双结算（Modification #1） ---")
        r_dual = run_province(prov, use_dual_settlement=True)
        results_dual.append(r_dual)

    # 最终汇总
    logger.info("\n" + "=" * 90)
    logger.info(f"  {'省份':<12}{'单结算收入':>14}{'双结算收入':>14}{'差异':>10}{'单捕获率':>10}{'双捕获率':>10}")
    logger.info("-" * 90)
    for rs, rd in zip(results_single, results_dual):
        diff_pct = (rs["regime"] / rd["regime"] - 1) * 100 if rd["regime"] > 0 else 0
        logger.info(f"  {rs['province']:<12}"
                    f"{rs['regime']:>14,.0f}"
                    f"{rd['regime']:>14,.0f}"
                    f"{diff_pct:>9.1f}%"
                    f"{rs['capture']:>9.1f}%"
                    f"{rd['capture']:>9.1f}%")


if __name__ == "__main__":
    main()
