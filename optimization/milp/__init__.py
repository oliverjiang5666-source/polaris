"""
MILP 多市场联合优化 (路线 C 实验)

架构：
  data_loader          ←  读数据（复用现有 parquet）
  scenario_generator   ←  K=200 场景生成（Bootstrap / Quantile / VAE）
  milp_formulation     ←  Pyomo 两阶段随机 LP 模型
  stochastic_solver    ←  HiGHS/Gurobi/COPT 求解器适配
  parallel_backtest    ←  multiprocessing walk-forward
  experiment_runner    ←  实验矩阵编排
  analyze_results      ←  结果分析与可视化

与 Regime V3 的对照实验目的：
  在相同数据 + 相同场景集下，测试 MILP 能否超越 Stochastic DP。
"""
