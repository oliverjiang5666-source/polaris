# Polaris · 北辰

> **电力市场 AI 决策系统**
> 为储能与新能源电站提供"预测—决策—报价"闭环优化

---

## 项目简介

Polaris 是为中国电力现货市场（日前 + 实时双结算）量身定制的 AI 决策系统，面向 200 MW 级储能电站和新能源电站，能在 SoC 安全约束、电池寿命预算、中长期合约交付等多约束下实现收益最优化。

核心算法基于 **多阶段随机动态规划（multistage stochastic DP）**，复现了 MIT Sloan 团队 2025 年 11 月最新论文（[Lee & Sun, arXiv:2511.15629](https://arxiv.org/abs/2511.15629)），并结合自研 **电价类型分类器（regime classifier）** 做中国市场条件化升级。

---

## 实测数据（2024 全年 walk-forward 回测）

| 省份 | 电池配置 | Oracle 理论上限 | Polaris 实测 | Capture |
|---|---|---|---|---|
| 山东 | 200 MW × 400 MWh | ¥ 8,308 万 / 年 | **¥ 5,681 万 / 年** | **68.4%** |
| 山西 | 同上 | ¥ 9,878 万 / 年 | ¥ 5,111 万 / 年 | 51.8% |
| 广东 | 同上 | ¥ 6,335 万 / 年 | ¥ 3,042 万 / 年 | 48.0% |
| 甘肃 | 同上 | ¥ 7,878 万 / 年 | ¥ 4,754 万 / 年 | 60.4% |

对比基准：Regime V3（自研 expected-price DP）= 56.2% 四省合计；MILP（工业金标准）= 50.3%（山东单测）。

---

## 核心能力

- **多市场联合报价**：日前 / 实时双结算、AGC 调频辅助服务、中长期合约统筹决策
- **储能全生命周期调度**：SoC ∈ [5%, 95%] 硬约束，循环预算，SoH 衰减模型，温度修正
- **状态反馈式闭环控制**：值函数 `V(t, SoC)` 查表，每步按真实 SoC 调整决策
- **决策可解释 / 可审计**：输出值函数 + bid curve，每个动作可追溯

---

## 目录结构

```
energy-storage-rl/
├── optimization/
│   ├── vfa_dp/                    核心：Lee-Sun 2025 tensor-based DP 复现
│   │   └── tensor_dp.py
│   ├── milp/                      对比基线：two-stage stochastic MILP
│   ├── agc_dp.py                  AGC 联合优化（DP 版本，待整合）
│   ├── cvar_dp.py                 CVaR 风险约束
│   └── mlt_allocator.py           中长期合约分解
├── oracle/lp_oracle.py            理论上限参考（perfect foresight LP）
├── forecast/                      价格预测模型（LightGBM、PatchTST 等）
├── env/battery_physics.py         SoH / 效率 / 温度物理模型
├── scripts/
│   ├── 22_regime_v3_allprov.py   自研 Regime V3 baseline
│   ├── 31_milp_diagnostic_d1.py   MILP 场景生成器消融
│   ├── 33_vfa_dp_shandong.py     Tensor DP 四省 backtest
│   └── 34_vfa_worst_day_diagnose.py  失败日诊断
└── data/china/processed/          真实市场数据（未入 repo，联系获取）
```

---

## 技术栈

- Python 3.9+ · NumPy · pandas · scikit-learn · LightGBM · PyTorch
- 优化：Pyomo + HiGHS（免费）/ Gurobi（可选）
- 核心求解器：**纯 NumPy 张量化实现**，不依赖商业求解器授权
- 硬件：MacBook Pro M4（10 核 CPU / 24 GB），全年 backtest 141 秒 / 省

---

## 快速运行

```bash
# 1. 环境
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 数据准备（需单独获取，不在 repo 内）
# data/china/processed/{shandong,shanxi,guangdong,gansu}_oracle.parquet

# 3. 跑 Tensor DP（山东全年）
PYTHONPATH=. python3 scripts/33_vfa_dp_shandong.py \
    --full --mode regime_conditioned --delta 0.005 --R 500

# 4. 对比 Regime V3 baseline
PYTHONPATH=. python3 scripts/22_regime_v3_allprov.py
```

---

## 关于

**一成智能（北京）科技有限公司**

团队来自智能体（Agent）与强化学习领域一线研发机构。核心技术栈：随机优化、动态规划、深度学习、电力市场微观结构建模。

---

## 授权

本仓库为一成智能内部研发代码，**未开源授权**。请勿未经许可转发、商用或用于训练公开模型。
