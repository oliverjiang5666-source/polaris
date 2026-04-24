# Polaris · 北辰

> **独立储能电站的 AI 充放电决策系统**

针对 200 MW 级电池储能电站，在现货市场（DA + RT）中做 96 步/天的充放电策略优化。

## 架构（L1 数据 → L2 预测 → L3 决策）

```
L1 数据层
├── data/china/processed/*.parquet        # 4 省历史数据
├── ../../crawlers/                        # 爬虫（共享）
└── scripts/00_ingest_china.py             # 数据入库

L2 预测层
├── forecast/lgbm_forecaster.py            # LightGBM 多 horizon（生产）
├── forecast/patchtst_model.py             # PatchTST 集成（研究）
└── optimization/milp/scenario_generator.py  # RegimeClassifier（场景生成）

L3 决策层
├── oracle/lp_oracle.py                    # LP Oracle 理论上限
├── optimization/vfa_dp/tensor_dp.py       # Tensor DP 主力（Lee-Sun 2025）
├── scripts/22_regime_v3_allprov.py        # Regime V3 自研
├── optimization/milp/                     # MILP 对照（已弃用）
├── optimization/agc_dp.py                 # AGC 联合（未整合）
├── optimization/cvar_dp.py                # CVaR 风险（未整合）
└── optimization/mlt_allocator.py          # 中长期合约（未整合）

辅助
├── env/battery_physics.py                 # SoH / 温度（未接入）
├── agent/                                  # 早期 RL agent（已弃用）
├── backtest/                               # Walk-forward 框架
└── config.py                               # BatteryConfig / ACTIONS
```

## Import 路径（保留旧路径不变）

Polaris 代码物理位置在 `products/polaris/`，但通过根目录 symlinks 保留原 import 路径：

```python
# 以下 import 都继续 work（从 repo 根运行）
from oracle.lp_oracle import solve_day
from optimization.vfa_dp.tensor_dp import TensorDP
from optimization.milp.scenario_generator import RegimeClassifier
from forecast.mpc_controller import simulate_mpc
from env.battery_physics import ...
from config import BatteryConfig
```

## 实测数据（2025 自然年 walk-forward，200 MW/400 MWh）

| 省份 | Oracle | Polaris (Tensor DP) | Capture |
|---|---|---|---|
| 山东 | ¥8,387 万 | **¥5,833 万** | **69.54%** |
| 山西 | ¥10,903 万 | ¥5,674 万 | 52.05% |
| 广东 | ¥6,235 万 | ¥3,127 万 | 50.16% |
| 甘肃 | ¥9,178 万 | ¥5,732 万 | 62.45% |

详见 [runs/vfa_dp_2025/](../../runs/vfa_dp_2025/)。

## 核心脚本入口

```bash
# 从 repo 根目录
PYTHONPATH=. python3 scripts/33_vfa_dp_shandong.py --full --mode regime_conditioned
PYTHONPATH=. python3 scripts/22_regime_v3_allprov.py
PYTHONPATH=. python3 scripts/35_vfa_2025_calendar_year.py
```

## 交接文档

主干 HANDOFF 留在 repo 根目录：
- `../../HANDOFF_POLARIS_2026-04-23.md`（最新）
- `../../HANDOFF_MILP_ROUTE_C.md`
- `../../HANDOFF_REGIME.md`
- `../../HANDOFF_TRANSFORMER.md` / `HANDOFF_PATCHTST.md`
- `../../HANDOFF_CHINA.md`
- `../../HANDOFF.md`

## 技术债

- `optimization/milp/scenario_generator.py` 里的 `RegimeClassifier` 和 `../../core/regime_classifier.py` 的 `RegimeClassifier` 是两份独立实现，接口不同。未来应合并到 `core/` 统一版本。
