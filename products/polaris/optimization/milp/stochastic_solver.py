"""
求解器统一适配层：Gurobi / HiGHS / COPT

用法：
    solver = SolverAdapter(backend='auto')  # 自动选最优
    result = solver.solve(model, time_limit=120)
"""
from __future__ import annotations

import pyomo.environ as pyo
from loguru import logger
from dataclasses import dataclass


@dataclass
class SolverResult:
    status: str
    objective: float | None
    solve_time: float
    backend: str
    raw_results: object = None


class SolverAdapter:
    """
    支持以下 backend（按优先级自动尝试）：
      - gurobi   (需 license，最快)
      - copt     (需 license，次快)
      - appsi_highs (Pyomo+HiGHS，免费)
      - highs    (alternative HiGHS)
      - glpk     (最后 fallback)
    """

    BACKENDS_ORDER = ["gurobi", "copt", "appsi_highs", "highs", "glpk"]

    def __init__(self, backend: str = "auto", threads: int = 2, verbose: bool = False):
        self.backend = backend
        self.threads = threads
        self.verbose = verbose
        self.available = None

        if backend == "auto":
            self.backend = self._detect_best_backend()
            logger.info(f"Auto-selected backend: {self.backend}")

    def _detect_best_backend(self) -> str:
        for b in self.BACKENDS_ORDER:
            try:
                factory = pyo.SolverFactory(b)
                if factory.available():
                    return b
            except Exception:
                pass
        raise RuntimeError("No solver available. Install at least highspy: pip install highspy")

    def solve(
        self,
        model: pyo.ConcreteModel,
        time_limit: float = 120.0,
        mip_gap: float = 0.01,
    ) -> SolverResult:
        import time
        t0 = time.time()

        solver = pyo.SolverFactory(self.backend)

        # 设置求解器参数
        if self.backend == "gurobi":
            solver.options["TimeLimit"] = time_limit
            solver.options["Threads"] = self.threads
            solver.options["MIPGap"] = mip_gap
            solver.options["Method"] = 2  # barrier
            solver.options["Crossover"] = 0  # 不做 crossover 加速
        elif self.backend == "copt":
            solver.options["TimeLimit"] = time_limit
            solver.options["Threads"] = self.threads
            solver.options["RelGap"] = mip_gap
        elif self.backend == "appsi_highs":
            # appsi_highs 配置通过 highs_options 传给底层 HiGHS
            solver.config.time_limit = time_limit
            solver.highs_options = {
                "threads": self.threads,
                "parallel": "on",
                "mip_rel_gap": mip_gap,
            }
        elif self.backend == "highs":
            solver.options["time_limit"] = time_limit
            solver.options["mip_rel_gap"] = mip_gap
            solver.options["threads"] = self.threads

        try:
            results = solver.solve(model, tee=self.verbose)
            solve_time = time.time() - t0

            # 规范化返回
            tc = str(results.solver.termination_condition) if hasattr(results, 'solver') else 'unknown'
            try:
                obj = pyo.value(model.obj)
            except Exception:
                obj = None

            return SolverResult(
                status=tc,
                objective=obj,
                solve_time=solve_time,
                backend=self.backend,
                raw_results=results,
            )
        except Exception as e:
            logger.error(f"Solver error ({self.backend}): {e}")
            return SolverResult(
                status="error",
                objective=None,
                solve_time=time.time() - t0,
                backend=self.backend,
            )


def quick_solve(
    model: pyo.ConcreteModel,
    backend: str = "auto",
    threads: int = 2,
    time_limit: float = 120.0,
) -> SolverResult:
    """一步 API：创建 adapter 并求解"""
    return SolverAdapter(backend=backend, threads=threads).solve(model, time_limit=time_limit)


if __name__ == "__main__":
    import numpy as np
    from optimization.milp.milp_formulation import (
        build_two_stage_lp, BatteryParams, MILPConfig, extract_solution,
    )

    # Detect solver
    adapter = SolverAdapter(backend="auto")
    logger.info(f"Detected backend: {adapter.backend}")

    # Quick benchmark: K=50, 96 timesteps
    np.random.seed(42)
    T = 96
    for K in [20, 50, 100, 200]:
        hours = np.arange(T) / 4
        base = 320 - 80 * np.cos(2 * np.pi * hours / 24)
        rt_scenarios = np.array([base + np.random.normal(0, 30, T) for _ in range(K)])
        dam_forecast = rt_scenarios.mean(axis=0)
        weights = np.ones(K) / K

        battery = BatteryParams()
        config = MILPConfig()

        model = build_two_stage_lp(dam_forecast, rt_scenarios, weights, battery, config)
        result = adapter.solve(model, time_limit=60)
        logger.info(f"K={K:3d}: status={result.status}, obj=¥{result.objective:,.0f}, "
                    f"time={result.solve_time:.2f}s")
