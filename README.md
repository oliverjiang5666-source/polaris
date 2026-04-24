# 一成智能 · 电力市场 AI 决策平台

> 为储能与新能源电站提供"预测—决策—报价"闭环优化

---

## 两条产品线

### Polaris · 北辰
独立储能电站（200 MW 级）的 AI 充放电决策。核心算法：**Tensor DP**（Lee & Sun 2025, arXiv:2511.15629）+ **Regime Classifier**（自研）。

- 2025 山东自然年实测：¥5,833 万 / 年（100 MW = ¥2,916 万）
- 理论上限 Capture：**69.54%**
- 详见 [products/polaris/README.md](products/polaris/README.md)

### Logan · 观云
新能源场站（风电 / 光伏）的 AI 报量报价策略。核心算法：**Optimal Bid = Breakpoint 搜索 + LP + Empirical Copula** + **合规 bid validation**。

- 甘肃 100 MW 光伏合规回测（甘肃 V3.2 规则）：相对 Naive 老实报 **+6.41%**（5 seed × 3 window 均值）
- Sharpe 从 6.07 → 9.57（年收入波动率降低 33%）
- 详见 [products/logan/README.md](products/logan/README.md)

---

## 目录结构

```
energy-storage-rl/
│
├── core/                              # 跨产品共享底座
│   ├── regime_classifier.py          #   Logan 独立版（Polaris 有单独一份，待合并）
│   ├── supply_curve.py                #   DA = f(净负荷)
│   ├── net_load_forecaster.py        #   净负荷预测
│   ├── joint_distribution.py         #   (DA, RT) Empirical copula
│   └── calendar_features.py          #   节假日 / 时段工具
│
├── products/
│   ├── polaris/                       # 储能产品（物理位置）
│   │   ├── optimization/              #   TensorDP / MILP / AGC / CVaR
│   │   ├── oracle/lp_oracle.py        #   LP Oracle
│   │   ├── forecast/                  #   LGBM / PatchTST / MPC
│   │   ├── env/battery_physics.py     #   SoH 物理模型
│   │   ├── agent/                     #   早期 RL（弃用）
│   │   ├── backtest/                  #   Walk-forward 框架
│   │   ├── scripts/                   #   01~35 运行脚本
│   │   ├── config.py                  #   BatteryConfig / ACTIONS
│   │   └── README.md
│   │
│   └── logan/                         # 新能源产品
│       ├── optimal_bid.py             #   理论最优 bid 生成
│       ├── dfl_bid_curve.py           #   SAA 版本
│       ├── bid_curve_generator.py     #   Heuristic 版本
│       ├── regime_aware_bid.py        #   Regime-Aware 版本
│       ├── oracle_bid.py              #   Perfect foresight 上限
│       ├── compliance.py              #   省级规则合规验证
│       ├── evaluator.py               #   真实结算引擎
│       ├── settlement_rules/          #   gansu.yaml 等规则
│       ├── scripts/                   #   01~09 运行脚本
│       └── README.md
│
├── optimization/ → products/polaris/optimization/    # symlink (backwards compat)
├── oracle/       → products/polaris/oracle/
├── forecast/     → products/polaris/forecast/
├── env/          → products/polaris/env/
├── agent/        → products/polaris/agent/
├── backtest/     → products/polaris/backtest/
├── scripts/      → products/polaris/scripts/
├── config.py     → products/polaris/config.py
│
├── data/                              # 共享数据（4 省 parquet + 爬取数据）
│   └── china/processed/
├── crawlers/                          # 共享爬虫基础设施
├── models/                            # 共享模型存储
│   ├── logan/{province}/              #   Logan 模型
│   └── (polaris 模型散落于其他位置)
├── runs/                              # 共享实验结果
│   ├── logan/
│   ├── vfa_dp_2025/                   #   Polaris 2025 自然年
│   └── milp_experiments/              #   Polaris MILP 实验
│
├── login_dianchacha.py                # 共享登录脚本
├── crawl_all_provinces.py             # 共享爬虫 driver
├── requirements.txt
│
├── README.md                          # (本文件)
└── HANDOFF_*.md                       # 交接文档（保留在根，避免外链断）
```

---

## 共享组件

两款产品共享以下模块（位于 `core/` 或 repo 根部）：

| 模块 | 作用 | 被谁用 |
|---|---|---|
| `core/regime_classifier.py` | 12 类日子分类（KMeans + GBM） | Logan |
| `core/supply_curve.py` | DA ≈ f(净负荷) 单调拟合 | Logan；Polaris 未来可接入 |
| `core/net_load_forecaster.py` | LightGBM 净负荷预测 | Logan |
| `core/joint_distribution.py` | (DA, RT) empirical copula | Logan |
| `core/calendar_features.py` | 节假日 / 时段 / 季节特征 | Logan |
| `data/china/processed/` | 4 省 15 分钟 parquet | 两者 |
| `crawlers/`, `login_dianchacha.py` | 电查查爬虫 | 两者 |

---

## 快速上手

### 跑 Polaris（储能）

```bash
# 从 repo 根目录
PYTHONPATH=. python3 scripts/33_vfa_dp_shandong.py --full --mode regime_conditioned
```

### 跑 Logan（新能源）

```bash
# 训练
PYTHONPATH=. python3 products/logan/scripts/02_train_all_heads.py --province gansu --recent-days 540

# 回测
PYTHONPATH=. python3 products/logan/scripts/07_gansu_realistic_backtest.py --days 90
```

---

## 技术栈

- Python 3.9+ · NumPy · pandas · scikit-learn · LightGBM · PyTorch（可选）
- 优化求解：scipy.linprog (LP) · Pyomo + HiGHS (MILP) · 纯 NumPy 张量化（Tensor DP）
- 硬件：MacBook Pro M4（单进程 CPU）

---

## 关于

**一成智能（北京）科技有限公司**

团队来自智能体与强化学习领域一线研发机构。核心技术栈：随机优化、动态规划、联合分布建模、电力市场微观结构建模。

---

## 授权

本仓库为一成智能内部研发代码，**未开源授权**。请勿未经许可转发、商用或用于训练公开模型。
