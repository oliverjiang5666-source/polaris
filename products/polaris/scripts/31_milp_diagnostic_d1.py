"""
诊断 D1: MILP 场景生成器消融
============================================

目的：定位 MILP v0.1 输 Regime V3 的根因是不是"场景生成"。

4 个变体（信息量逐级升级）：
  A1 (baseline 复刻)  = bootstrap 单日 RT + 均匀 1/K 权重           K=200
  A2 (regime profile) = K_eff 个 regime profiles（类内均值）+ regime_probs 权重  K≈5-10
  A3 (bootstrap prob-weighted) = bootstrap 单日 + prob/count 权重    K=200
  A4 (regime mean, full K)     = K 个场景，每个 = 该 regime 天的均值（重复）+ regime_probs  K=200

诊断矩阵：
  A1 ≈ A3 ≈ A2 ≈ A4  → 场景生成不是罪魁，问题在两阶段结构或实现（H3/H4）
  A1 < A3 ≈ A2 ≈ A4  → 采样方差是主因（改权重就够）
  A1 ≈ A3 << A2 ≈ A4 → 单日噪声是主因（要用类内均值代替单日）
  A2 ≈ A4 追平 Regime V3 → H1+H2 坐实，v0.2 直接换场景生成器

Usage:
    # Smoke test (10 天，单线程，约 5 分钟)
    PYTHONPATH=. python3 scripts/31_milp_diagnostic_d1.py --smoke

    # 全量 (365 天，本地 4 并发，约 6 小时)
    PYTHONPATH=. python3 scripts/31_milp_diagnostic_d1.py --full
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import time
import pickle
from pathlib import Path
from multiprocessing import get_context
from collections import Counter
from loguru import logger

import pyomo.environ as pyo

from optimization.milp.data_loader import load_province, split_walkforward
from optimization.milp.scenario_generator import (
    RegimeClassifier, ScenarioSet, generate_scenarios_bootstrap,
)
from optimization.milp.milp_formulation import (
    build_two_stage_lp, BatteryParams, MILPConfig, extract_solution, simulate_on_actual,
)
from optimization.milp.stochastic_solver import SolverAdapter


OUTPUT = Path("runs/d1_diagnostic")
OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4 个场景生成器
# ============================================================


def gen_A1_bootstrap_uniform(classifier, data, target_day, K, rng):
    """A1 = 现状复刻"""
    return generate_scenarios_bootstrap(classifier, data, target_day, K=K, rng=rng)


def gen_A2_regime_profile(classifier, data, target_day, K=None, rng=None):
    """
    A2 = 12 regime profiles + regime_probs 权重。
    每个场景 = 训练集某 regime 所有天的 RT 均值（类内平均）。
    K = 实际保留的 regime 数（只保留 prob > 0.02）。
    """
    probs, _ = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()

    train_labels = classifier.train_labels
    train_rt = data.rt_prices[:len(train_labels)]
    train_dam = data.dam_prices[:len(train_labels)]

    n_reg = classifier.n_regimes
    rt_profiles = np.zeros((n_reg, 96))
    dam_profiles = np.zeros((n_reg, 96))
    for c in range(n_reg):
        mask = train_labels == c
        if mask.sum() > 0:
            rt_profiles[c] = train_rt[mask].mean(axis=0)
            dam_profiles[c] = train_dam[mask].mean(axis=0)
        else:
            rt_profiles[c] = train_rt.mean(axis=0)
            dam_profiles[c] = train_dam.mean(axis=0)

    # 只保留 prob > 0.02（和 Regime V3 scripts/22 保持一致）
    kept = probs > 0.02
    if kept.sum() == 0:
        kept = probs > 0  # fallback
    kept_probs = probs[kept]
    kept_probs = kept_probs / kept_probs.sum()
    kept_rt = rt_profiles[kept]
    kept_dam = dam_profiles[kept]

    dam_forecast = (kept_probs[:, None] * kept_dam).sum(axis=0)

    return ScenarioSet(
        rt_scenarios=kept_rt,
        dam_forecast=dam_forecast,
        dam_scenarios=kept_dam,
        weights=kept_probs,
        method="regime_profile",
        meta={"regime_probs": probs, "K_eff": int(kept.sum())},
    )


def gen_A3_bootstrap_weighted(classifier, data, target_day, K, rng):
    """A3 = bootstrap 单日 + prob/count 权重"""
    if rng is None:
        rng = np.random.default_rng(42)

    probs, _ = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()

    train_labels = classifier.train_labels
    train_rt = data.rt_prices[:len(train_labels)]
    train_dam = data.dam_prices[:len(train_labels)]

    # 按 prob 采 K 个 regime
    sampled_regimes = rng.choice(len(probs), size=K, p=probs)
    counts = Counter(sampled_regimes.tolist())

    rt_scenarios = np.zeros((K, 96))
    dam_scenarios = np.zeros((K, 96))
    weights = np.zeros(K)

    for k in range(K):
        reg = sampled_regimes[k]
        candidate_days = np.where(train_labels == reg)[0]
        if len(candidate_days) == 0:
            candidate_days = np.arange(len(train_labels))
        day_idx = rng.choice(candidate_days)
        rt_scenarios[k] = train_rt[day_idx]
        dam_scenarios[k] = train_dam[day_idx]
        # prob/count 保证 sum(weights) = sum_reg prob(reg) = 1
        weights[k] = probs[reg] / counts[reg]

    # 归一化 (防浮点误差)
    weights = weights / weights.sum()
    dam_forecast = (weights[:, None] * dam_scenarios).sum(axis=0)

    return ScenarioSet(
        rt_scenarios=rt_scenarios,
        dam_forecast=dam_forecast,
        dam_scenarios=dam_scenarios,
        weights=weights,
        method="bootstrap_weighted",
        meta={"regime_probs": probs},
    )


def gen_A4_regime_mean_full_K(classifier, data, target_day, K, rng):
    """
    A4 = K 个场景，每个 = 被采 regime 的 class profile（类内均值）。
    相当于 A2 但 K 放大（重复相同 profile，权重稀释）→ 对 MILP 来说数学等价于 A2。
    这是一个对照：验证"MILP 求解是否对 K 的名义大小敏感"（技术边界条件）。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    probs, _ = classifier.predict_regime_probs(data, target_day)
    probs = probs / probs.sum()

    train_labels = classifier.train_labels
    train_rt = data.rt_prices[:len(train_labels)]
    train_dam = data.dam_prices[:len(train_labels)]

    n_reg = classifier.n_regimes
    rt_profiles = np.zeros((n_reg, 96))
    dam_profiles = np.zeros((n_reg, 96))
    for c in range(n_reg):
        mask = train_labels == c
        if mask.sum() > 0:
            rt_profiles[c] = train_rt[mask].mean(axis=0)
            dam_profiles[c] = train_dam[mask].mean(axis=0)
        else:
            rt_profiles[c] = train_rt.mean(axis=0)
            dam_profiles[c] = train_dam.mean(axis=0)

    sampled_regimes = rng.choice(len(probs), size=K, p=probs)
    rt_scenarios = rt_profiles[sampled_regimes]
    dam_scenarios = dam_profiles[sampled_regimes]
    weights = np.ones(K) / K

    dam_forecast = (weights[:, None] * dam_scenarios).sum(axis=0)

    return ScenarioSet(
        rt_scenarios=rt_scenarios,
        dam_forecast=dam_forecast,
        dam_scenarios=dam_scenarios,
        weights=weights,
        method="regime_mean_K",
        meta={"regime_probs": probs},
    )


SCENARIO_GENS = {
    "A1_bootstrap_uniform": gen_A1_bootstrap_uniform,
    "A2_regime_profile":    gen_A2_regime_profile,
    "A3_bootstrap_weighted": gen_A3_bootstrap_weighted,
    "A4_regime_mean_K":     gen_A4_regime_mean_full_K,
    # A5/A6 复用 A1/A2 的场景生成器，但用 RT-only formulation
    "A5_bootstrap_RT_only":  gen_A1_bootstrap_uniform,
    "A6_profile_RT_only":    gen_A2_regime_profile,
}

# 每个变体用哪种求解器 formulation
VARIANT_FORMULATION = {
    "A1_bootstrap_uniform":   "two_stage",
    "A2_regime_profile":      "two_stage",
    "A3_bootstrap_weighted":  "two_stage",
    "A4_regime_mean_K":       "two_stage",
    "A5_bootstrap_RT_only":   "rt_only",
    "A6_profile_RT_only":     "rt_only",
}


# ============================================================
# RT-only 单市场 stochastic LP
# ============================================================


def build_rt_only_lp(
    rt_scenarios: np.ndarray,       # [K, 96]
    weights: np.ndarray,             # [K]
    battery: BatteryParams,
    config: MILPConfig,
) -> pyo.ConcreteModel:
    """
    RT-only 单市场 stochastic LP（对标 Regime V3 DP）

    场景加权期望收益最大化，无 DAM/RT 两阶段分解。
    决策变量只有每时段的 p_charge[t], p_discharge[t]（场景无关，即 "here-and-now"）。
    SoC 是确定性轨迹（因为动作确定）。

    这在数学上等价于：用 expected_rt = Σ_s w_s × rt[s] 作为价格做 deterministic LP。
    和 Regime V3 的"expected price DP"一样，只是 LP vs DP 的离散粒度差异。

    目标：max Σ_t (p_discharge[t] − p_charge[t]) × E[rt[t]] × dt − deg × (p_ch + p_dis) × dt
    约束：SoC 动力学、SoC 边界、初始/结束 SoC
    """
    K, T = rt_scenarios.shape
    m = pyo.ConcreteModel(name="RTOnlyStochasticLP")
    m.T = pyo.RangeSet(0, T - 1)
    m.T1 = pyo.RangeSet(0, T)

    # 计算期望 RT
    expected_rt = (weights[:, None] * rt_scenarios).sum(axis=0)  # [T]
    m.exp_rt = pyo.Param(m.T, initialize={t: float(expected_rt[t]) for t in range(T)})

    P_max = battery.P_max
    m.p_charge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.p_discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.soc = pyo.Var(m.T1, domain=pyo.NonNegativeReals,
                    bounds=(battery.soc_min, battery.soc_max))

    # SoC 动力学
    def _soc_dyn(m, t):
        return m.soc[t + 1] == m.soc[t] + \
            battery.dt * (battery.eta_c * m.p_charge[t] - m.p_discharge[t] / battery.eta_d) / battery.E_max
    m.c_soc_dyn = pyo.Constraint(m.T, rule=_soc_dyn)
    m.c_soc_init = pyo.Constraint(expr=m.soc[0] == config.init_soc)
    if config.final_soc_min is not None:
        m.c_soc_final = pyo.Constraint(expr=m.soc[T] >= config.final_soc_min)

    # 目标函数
    def _obj(m):
        total = 0.0
        for t in m.T:
            total += (m.p_discharge[t] - m.p_charge[t]) * m.exp_rt[t] * battery.dt
            total -= battery.deg_cost * (m.p_charge[t] + m.p_discharge[t]) * battery.dt
        return total
    m.obj = pyo.Objective(rule=_obj, sense=pyo.maximize)
    return m


def extract_rt_only_solution(model: pyo.ConcreteModel, T: int = 96) -> dict:
    net_power = np.array([
        pyo.value(model.p_discharge[t]) - pyo.value(model.p_charge[t])
        for t in range(T)
    ])
    charge = np.array([pyo.value(model.p_charge[t]) for t in range(T)])
    discharge = np.array([pyo.value(model.p_discharge[t]) for t in range(T)])
    soc = np.array([pyo.value(model.soc[t]) for t in range(T + 1)])
    return {
        "net_power": net_power,
        "charge": charge,
        "discharge": discharge,
        "soc": soc,
        "objective": pyo.value(model.obj),
    }


def simulate_rt_only(
    net_power: np.ndarray,           # [96]
    actual_rt: np.ndarray,            # [96]
    battery: BatteryParams,
    config: MILPConfig,
) -> dict:
    """
    RT-only 单结算仿真，和 Regime V3（scripts/22）的 _step_battery 结算语义对齐。
    revenue = Σ (discharge[t] - charge[t]) × rt[t] × dt − deg × |energy| × dt
    SoC 越界时截断（和 Regime V3 相同）。
    """
    T = len(net_power)
    dt = battery.dt
    soc = config.init_soc
    total_rev = 0.0
    total_deg = 0.0
    total_charge_mwh = 0.0
    total_discharge_mwh = 0.0
    actual_power = np.zeros(T)

    for t in range(T):
        pw = float(np.clip(net_power[t], -battery.P_max, battery.P_max))
        if pw > 0:  # 放电
            energy = pw * dt
            soc_need = energy / (battery.E_max * battery.eta_d)
            if soc - soc_need < battery.soc_min:
                available = max((soc - battery.soc_min) * battery.E_max * battery.eta_d, 0)
                energy = available
                pw = energy / dt
                soc = battery.soc_min
            else:
                soc -= soc_need
            total_discharge_mwh += energy
        elif pw < 0:  # 充电
            energy = -pw * dt
            soc_add = energy * battery.eta_c / battery.E_max
            if soc + soc_add > battery.soc_max:
                available = max((battery.soc_max - soc) * battery.E_max / battery.eta_c, 0)
                energy = available
                pw = -energy / dt
                soc = battery.soc_max
            else:
                soc += soc_add
            total_charge_mwh += energy
        actual_power[t] = pw
        total_rev += pw * actual_rt[t] * dt
        total_deg += battery.deg_cost * abs(pw) * dt

    return {
        "revenue_total": total_rev - total_deg,
        "revenue_dam": 0.0,
        "revenue_dev": total_rev,   # 都记在 dev 里（单市场）
        "degradation": total_deg,
        "actual_power": actual_power,
        "final_soc": soc,
        "total_charge_mwh": total_charge_mwh,
        "total_discharge_mwh": total_discharge_mwh,
    }


# ============================================================
# 子进程单日求解
# ============================================================


def _solve_one(args):
    """子进程调用的单日求解。根据 variant 选 formulation。"""
    day_idx = args["day_idx"]
    variant = args["variant"]
    scen = args["scen"]
    actual_rt = args["actual_rt"]
    actual_dam = args["actual_dam"]
    battery = args["battery"]
    config = args["config"]
    formulation = args["formulation"]

    t0 = time.time()
    K = scen.rt_scenarios.shape[0]

    if formulation == "two_stage":
        model = build_two_stage_lp(
            scen.dam_forecast, scen.rt_scenarios, scen.weights,
            battery, config,
        )
        adapter = SolverAdapter(backend="appsi_highs", threads=2, verbose=False)
        result = adapter.solve(model, time_limit=180)
        if result.status != "optimal":
            return {"day_idx": day_idx, "variant": variant, "status": result.status,
                    "actual_revenue": 0.0, "objective": None, "solve_time": time.time() - t0,
                    "formulation": formulation}
        sol = extract_solution(model, K, 96)
        p_dam = sol["p_dam"]
        sim = simulate_on_actual(p_dam, actual_rt, actual_dam, battery, config)
        return {
            "day_idx": day_idx, "variant": variant, "status": "optimal",
            "formulation": formulation,
            "objective": result.objective,
            "actual_revenue": sim["revenue_total"],
            "revenue_dam": sim["revenue_dam"],
            "revenue_dev": sim["revenue_dev"],
            "p_dam_mean": float(p_dam.mean()),
            "p_dam_std": float(p_dam.std()),
            "p_dam_abs_max": float(np.abs(p_dam).max()),
            "K_used": int(K),
            "solve_time": time.time() - t0,
            "scenario_method": scen.method,
        }

    elif formulation == "rt_only":
        model = build_rt_only_lp(scen.rt_scenarios, scen.weights, battery, config)
        adapter = SolverAdapter(backend="appsi_highs", threads=2, verbose=False)
        result = adapter.solve(model, time_limit=60)
        if result.status != "optimal":
            return {"day_idx": day_idx, "variant": variant, "status": result.status,
                    "actual_revenue": 0.0, "objective": None, "solve_time": time.time() - t0,
                    "formulation": formulation}
        sol = extract_rt_only_solution(model, T=96)
        net = sol["net_power"]
        sim = simulate_rt_only(net, actual_rt, battery, config)
        return {
            "day_idx": day_idx, "variant": variant, "status": "optimal",
            "formulation": formulation,
            "objective": result.objective,
            "actual_revenue": sim["revenue_total"],
            "revenue_dam": sim["revenue_dam"],
            "revenue_dev": sim["revenue_dev"],
            "p_dam_mean": float(net.mean()),
            "p_dam_std": float(net.std()),
            "p_dam_abs_max": float(np.abs(net).max()),
            "K_used": int(K),
            "solve_time": time.time() - t0,
            "scenario_method": scen.method,
        }

    else:
        raise ValueError(f"unknown formulation: {formulation}")


# ============================================================
# 主流程
# ============================================================


def run_diagnostic(
    province: str = "shandong",
    test_days: list[int] | None = None,
    K: int = 200,
    n_workers: int = 4,
    variants: list[str] | None = None,
) -> pd.DataFrame:
    """跑 D1 诊断"""
    if variants is None:
        variants = list(SCENARIO_GENS.keys())

    data = load_province(province)
    quarters = split_walkforward(data)

    if test_days is None:
        # 默认：最后一个季度的前 10 天（smoke）
        qs, qe = quarters[-1]
        test_days = list(range(qs, min(qs + 10, qe)))

    # 训练分类器（训到第一个 test day 之前）
    train_end = test_days[0]
    logger.info(f"Training classifier on [0, {train_end}) ...")
    t0 = time.time()
    classifier = RegimeClassifier(n_regimes=12)
    classifier.fit(data, train_day_end=train_end)
    logger.info(f"  done in {time.time() - t0:.1f}s")

    battery = BatteryParams()
    config = MILPConfig(deviation_bound=0.10, final_soc_min=0.3)

    # 准备任务（在主进程生成所有场景）
    tasks = []
    for d in test_days:
        for variant in variants:
            gen = SCENARIO_GENS[variant]
            # 每个 (day, variant) 用独立 rng 子流（可复现）
            sub_rng = np.random.default_rng(42 + d * 17 + hash(variant) % 1000)
            scen = gen(classifier, data, target_day=d, K=K, rng=sub_rng)
            tasks.append({
                "day_idx": d,
                "variant": variant,
                "scen": scen,
                "actual_rt": data.rt_prices[d],
                "actual_dam": data.dam_prices[d],
                "battery": battery,
                "config": config,
                "formulation": VARIANT_FORMULATION[variant],
            })

    logger.info(f"Prepared {len(tasks)} tasks ({len(test_days)} days × {len(variants)} variants)")

    # 并行求解
    t1 = time.time()
    if n_workers == 1:
        records = [_solve_one(t) for t in tasks]
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            records = pool.map(_solve_one, tasks)
    total_t = time.time() - t1
    logger.info(f"All solves done in {total_t:.1f}s ({total_t/len(tasks):.1f}s/task avg)")

    df = pd.DataFrame(records)
    return df


def analyze_and_print(df: pd.DataFrame):
    """按 variant 聚合并打印对比"""
    logger.info("\n" + "=" * 90)
    logger.info("D1 诊断结果（每 variant 的总收入，以及平均决策特征）")
    logger.info("=" * 90)

    agg = df.groupby("variant").agg(
        days=("day_idx", "count"),
        total_revenue=("actual_revenue", "sum"),
        mean_revenue_per_day=("actual_revenue", "mean"),
        mean_objective=("objective", "mean"),
        mean_solve_time=("solve_time", "mean"),
        mean_p_dam_abs=("p_dam_abs_max", "mean"),
        mean_p_dam_std=("p_dam_std", "mean"),
        mean_K=("K_used", "mean"),
    ).round(2).sort_values("total_revenue", ascending=False)

    logger.info("\n" + agg.to_string())

    # 用 A1 作基准
    if "A1_bootstrap_uniform" in agg.index:
        base = agg.loc["A1_bootstrap_uniform", "total_revenue"]
        logger.info("\n相对 A1 基准（总收入）:")
        for v in agg.index:
            r = agg.loc[v, "total_revenue"]
            delta_pct = (r / base - 1) * 100 if base != 0 else 0
            logger.info(f"  {v:<28s}: ¥{r:>12,.0f}  ({delta_pct:+6.1f}%)")

    # 配对差分（同一天上不同 variant 对比）
    logger.info("\n按天配对对比（A? vs A1 的 per-day 差）:")
    pivot = df.pivot_table(index="day_idx", columns="variant", values="actual_revenue")
    if "A1_bootstrap_uniform" in pivot.columns:
        for v in [c for c in pivot.columns if c != "A1_bootstrap_uniform"]:
            diff = pivot[v] - pivot["A1_bootstrap_uniform"]
            positive_share = (diff > 0).mean() * 100
            logger.info(f"  {v:<28s}: 均值差 ¥{diff.mean():>+10,.0f}  中位数 ¥{diff.median():>+10,.0f}  "
                        f">0 的天占 {positive_share:.0f}%")

    # 诊断诠释
    logger.info("\n" + "-" * 90)
    logger.info("诊断诠释")
    logger.info("-" * 90)

    def _pct(a, b): return (a / b - 1) * 100 if b else 0
    def _get(v): return agg.loc[v, "total_revenue"] if v in agg.index else None

    A1 = _get("A1_bootstrap_uniform")
    A2 = _get("A2_regime_profile")
    A3 = _get("A3_bootstrap_weighted")
    A4 = _get("A4_regime_mean_K")
    A5 = _get("A5_bootstrap_RT_only")
    A6 = _get("A6_profile_RT_only")

    # 场景生成维度（two_stage 固定）
    if all(x is not None for x in [A1, A2, A3]):
        logger.info(f"\n  [two_stage 下] A2 vs A1: {_pct(A2, A1):+.1f}%   A3 vs A1: {_pct(A3, A1):+.1f}%")

    # Formulation 维度
    if A1 is not None and A5 is not None:
        logger.info(f"  [同 bootstrap 场景] A5 RT-only vs A1 two-stage: {_pct(A5, A1):+.1f}%")
    if A2 is not None and A6 is not None:
        logger.info(f"  [同 profile 场景] A6 RT-only vs A2 two-stage: {_pct(A6, A2):+.1f}%")

    # RT-only 场景维度
    if A5 is not None and A6 is not None:
        logger.info(f"  [同 RT-only formulation] A6 profile vs A5 bootstrap: {_pct(A6, A5):+.1f}%")

    # Regime V3 reference
    # scripts/22 山东全年 ¥5,381万 → 平均 ¥147,425/day（365 天基准）
    # 但实际山东 test 是 n_days - 400 ~ 365 天，每天大致均匀
    REGIME_V3_PER_DAY_REF = {
        "shandong": 53_810_000 / 365,
        "shanxi":   62_440_000 / 365,
        "guangdong": 35_590_000 / 365,
        "gansu":     32_420_000 / 365,
    }
    logger.info("\n  参考：Regime V3 每日均值（按全年摊）:")
    for prov, v in REGIME_V3_PER_DAY_REF.items():
        logger.info(f"    {prov}: ¥{v:,.0f}/day")

    n_days = agg["days"].max() if "days" in agg.columns else 10
    logger.info(f"\n  按当前 {n_days} 天 per-day 均值对比 Regime V3 shandong 基准 ¥{REGIME_V3_PER_DAY_REF['shandong']:,.0f}:")
    for v in agg.index:
        per_day = agg.loc[v, "mean_revenue_per_day"]
        gap = _pct(per_day, REGIME_V3_PER_DAY_REF['shandong'])
        logger.info(f"    {v:<28s}: ¥{per_day:>9,.0f}/day  ({gap:+6.1f}% vs Regime V3)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="10 天快速诊断")
    parser.add_argument("--full", action="store_true", help="365 天全量诊断")
    parser.add_argument("--province", default="shandong")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--K", type=int, default=200)
    args = parser.parse_args()

    data = load_province(args.province)
    quarters = split_walkforward(data)

    if args.smoke or not args.full:
        qs, qe = quarters[-1]
        test_days = list(range(qs, min(qs + 10, qe)))
        tag = "smoke10d"
        logger.info(f"SMOKE mode: {args.province} 最后一个季度前 10 天 {test_days[0]}~{test_days[-1]}")
    else:
        # full：整个测试期（4 季度）
        test_days = []
        for qs, qe in quarters:
            test_days.extend(range(qs, qe))
        tag = "full365d"
        logger.info(f"FULL mode: {args.province} 全测试期 {len(test_days)} 天")

    df = run_diagnostic(
        province=args.province,
        test_days=test_days,
        K=args.K,
        n_workers=args.workers,
    )

    csv_path = OUTPUT / f"{args.province}_d1_{tag}.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n保存详细结果到 {csv_path}")

    analyze_and_print(df)

    # 保存 aggregated summary
    agg = df.groupby("variant").agg(
        days=("day_idx", "count"),
        total_revenue=("actual_revenue", "sum"),
        mean_revenue_per_day=("actual_revenue", "mean"),
        mean_objective=("objective", "mean"),
        mean_solve_time=("solve_time", "mean"),
    ).round(2)
    agg_path = OUTPUT / f"{args.province}_d1_{tag}_summary.csv"
    agg.to_csv(agg_path)
    logger.info(f"保存摘要到 {agg_path}")


if __name__ == "__main__":
    main()
