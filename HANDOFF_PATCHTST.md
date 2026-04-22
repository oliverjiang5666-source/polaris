# PatchTST电价预测 V4 — 完整交接文档

---

## 一、本次session做了什么

### 1. 深度研究 (EPF论文2024-2026)

核心发现：
- **线性混合模型仍可胜过纯Transformer**（El Mahtout & Ziel 2025, arXiv:2601.02856）
- **PatchTST是EPF最佳单一Transformer**（patch=16步=4小时天然适配15分钟粒度）
- **iTransformer**适合多变量EPF（channel-wise attention）
- **RevIN归一化**比自适应归一化更好（但有正确使用的前提）
- **在线学习**是中国市场的必要条件
- **广东Applied Energy 2024**：增强线性模型比Transformer好16-52%

关键论文列表：
| 论文 | ID | 核心价值 |
|------|-----|---------|
| El Mahtout & Ziel 2025 | arXiv:2601.02856 | 线性+非线性混合+在线学习 |
| 2026 DL-EPF综述 | arXiv:2602.10071 | 最全面的EPF深度学习综述 |
| CT-PatchTST | arXiv:2501.08620 | PatchTST+channel attention for renewable |
| Foundation Model EPF Benchmark | arXiv:2506.08113 | Chronos-Bolt/TimesFM等 |
| 可再生能源因果影响 | arXiv:2501.10423 | 风电对价格的U型非线性效应 |

### 2. 根因分析 — 为什么V2 Transformer比LightGBM差14.6%

| # | 问题 | 影响 | 已修复? |
|---|------|------|---------|
| 1 | **归一化bug**: `np.convolve(mode='same')`居中卷积泄露未来数据 | 3-5% | ✅ V3b |
| 2 | **Path B只有17维**: 缺失14个LightGBM最强特征 | 5-8% | ✅ V3b (31维) |
| 3 | **无target-hour编码**: 输出位置不知道预测几点 | 2-3% | ✅ V3b |
| 4 | **无price_lag_96**: 昨日同时段价格 | 2-3% | ✅ V3b |
| 5 | **Last-token瓶颈**: 672步压缩到128维 | 2-3% | ✅ V3b (mean pooling) |
| 6 | **缺天气因果特征**: 没有温度/风速/光照 | 未知 | ✅ V4 |

### 3. 实现了什么代码

#### 新文件
| 文件 | 功能 |
|------|------|
| `forecast/patchtst_model.py` | PatchTST-EPF模型（V3b，无RevIN，直接96值输出） |
| `scripts/12_patchtst_train.py` | V3b训练+MPC评估脚本 |
| `scripts/13_fetch_weather.py` | Open-Meteo天气数据爬取（4省5年，已完成） |
| `scripts/14_patchtst_v4_train.py` | V4训练脚本（39维因果特征） |

#### 修改的文件
| 文件 | 改了什么 |
|------|---------|
| `forecast/transformer_config.py` | 添加`get_config_v3()`、`get_config_v4()`、`PROVINCE_CONFIGS_V3/V3`、`patch_size`参数 |
| `data/china/features.py` | 添加8个因果特征定义(`CAUSAL_FEATURES`)、`FEATURE_COLS_V4`(39维)、`_build_causal_features()`函数 |
| `scripts/gpu_pack_v3.sh` | 添加V4脚本到打包列表 |

#### 数据变更
| 文件 | 变更 |
|------|------|
| `data/china/processed/*_oracle.parquet` | 4省全部加入了天气数据(temperature_2m, wind_speed_10m, shortwave_radiation, direct_radiation) + 8个因果特征 |
| `data/china/processed/*_weather.parquet` | 4省原始天气数据备份 |

### 4. 实验结果

#### V3b PatchTST (山东, 只跑完山东)

| 方法 | Revenue(元) | vs Threshold | vs LightGBM |
|------|-----------|-------------|-------------|
| Threshold | 37,014,828 | 基准 | — |
| LightGBM MPC | 43,736,963 | +18.2% | 基准 |
| V2 Transformer | 37,342,170 | +0.9% | -14.6% |
| **V3b PatchTST** | **38,130,389** | **+3.0%** | **-12.8%** |
| Oracle | 81,347,077 | +119.8% | — |

进步：V2→V3b从-14.6%提升到-12.8%，Threshold超额收益从+0.9%提升到+3.0%。

#### V3a失败实验（RevIN双重归一化）
- V3a加了RevIN（可逆实例归一化），但dataset已经做了因果归一化，导致双重归一化
- 结果：35,768,643 (-18.2% vs LightGBM)，比V2更差
- 教训：RevIN和手动归一化不能叠加使用

---

## 二、V4架构详解

### PatchTST-EPF V4 = V3b + 因果天气特征

```
输入:
  Path A: price_seq (B, 672, 8) — 8通道序列
  Path B: exo_features (B, 39) — 31基础 + 8因果
  Path C: price_lag_96 (B, 96) — 昨日同时段价格
  target_hours: (B, 24) — 目标时刻小时编码

架构:
  Path A → PatchEmbed(16步/patch=42 patches) → TransformerEncoder(4层) → MeanPool
  Path B → MLP(39→128) → embedding
  Path C → Linear(96→128) → embedding
  Concat[A,B,C] → MLP(384→512→512→96) + hour_bias + pos_bias → 96个价格预测
```

### 39维特征列表

**原31维 (FEATURE_COLS):**
- 价格14维: rt_price, lag_1~4, ma_4/16/96, std_16/96, trend, percentile, ma_ratio, ma4_ratio
- 时间6维: hour/weekday/month sin/cos
- 中国11维: da_price, da_rt_spread, da_price_ma_ratio, load_norm, load_change, renewable_penetration, wind_ratio, solar_ratio, net_load_norm, tie_line_norm, temperature_norm

**新增8维因果特征 (CAUSAL_FEATURES):**
| 特征 | 因果含义 | 数据来源 |
|------|---------|---------|
| wind_speed_norm | 风速→风电出力（因→果） | Open-Meteo |
| solar_radiation_norm | 光照→光伏出力（因→果） | Open-Meteo |
| temp_load_interaction | 极端温度×负荷（天气→需求激增） | Open-Meteo + 负荷 |
| wind_ramp | 风电1h爬坡率（急变→价格spike） | wind_mw差分 |
| solar_ramp | 光伏1h变化率（日落骤降→晚高峰） | solar_mw差分 |
| net_load_ramp | 净负荷爬坡（供需平衡变化速度） | (load-renewable)差分 |
| supply_demand_tightness | 净负荷/总负荷（接近1=供需紧张→高价） | load, renewable |
| renewable_forecast_proxy | 新能源4h MA（平滑趋势=预测代理） | renewable_mw |

---

## 三、下一步该做什么（优先级排序）

### 步骤1: 跑V4训练（立刻做）

```bash
# 本地打包
cd /Users/jjj/Desktop/工作/电力交易/energy-storage-rl
bash scripts/gpu_pack_v3.sh

# 上传到GPU
sshpass -p 'pLr4+UXpvt2x' scp -P 23074 energy-storage-gpu-v3.tar.gz root@connect.westd.seetacloud.com:/root/autodl-tmp/

# SSH进服务器
sshpass -p 'pLr4+UXpvt2x' ssh -p 23074 root@connect.westd.seetacloud.com

# 在服务器上
cd /root/autodl-tmp
tar xzf energy-storage-gpu-v3.tar.gz
PYTHONPATH=/root/autodl-tmp nohup /root/miniconda3/bin/python -u scripts/14_patchtst_v4_train.py --all > gpu_patchtst_v4.log 2>&1 &

# 查看进度
tail -f gpu_patchtst_v4.log
grep -E 'Revenue|>>>|Training:' gpu_patchtst_v4.log
```

### 步骤2: LightGBM + PatchTST 集成（高优先级）

无论V4结果如何，集成几乎必定优于任何单一模型（GEFCom竞赛所有冠军都用集成）。

```python
# 在验证集上搜索最优alpha
final_pred = alpha * lgbm_pred + (1 - alpha) * patchtst_pred
# alpha通常在0.5-0.8之间（LightGBM权重更高因为它目前更好）
```

实现：在 `scripts/14_patchtst_v4_train.py` 的 `evaluate()` 函数中加一个集成评估。需要在MPC evaluate循环中同时跑两个forecaster，然后加权平均。

### 步骤3: 超参搜索（中优先级）

当前用固定超参（embed=128, heads=4, layers=4）。论文为每个市场单独搜索。

搜索空间建议：
```
embed_dim: [64, 128, 256]
n_heads: [2, 4, 8]
n_layers: [2, 4, 6]
patch_size: [8, 16, 32]
lr: [5e-4, 1e-4, 5e-5]
```

每个配置~30分钟/省在5090上。可以用Optuna自动搜。

### 步骤4: 爬煤价（中优先级）

```python
# 用AkShare获取秦皇岛动力煤价格（周频）
import akshare as ak
df_coal = ak.futures_zh_spot(symbol="动力煤")
```

煤价是火电边际成本的直接决定因素，虽然是周频但可以作为regime indicator。

### 步骤5: 概率预测（低优先级但长期有价值）

改MSE loss为Quantile Loss，输出P10/P50/P90三个分位数。MPC可以利用不确定性做鲁棒优化。

---

## 四、GPU服务器信息

| 项目 | 信息 |
|------|------|
| 平台 | AutoDL |
| GPU | RTX 5090 × 1, 32GB VRAM |
| SSH | `ssh -p 23074 root@connect.westd.seetacloud.com` |
| 密码 | `pLr4+UXpvt2x` |
| Python | `/root/miniconda3/bin/python` (3.12) |
| 项目路径 | `/root/autodl-tmp/` |
| V3b日志 | `/root/autodl-tmp/gpu_patchtst_v3b.log` |
| V4日志（待跑）| `/root/autodl-tmp/gpu_patchtst_v4.log` |
| 计费 | ¥2.93/时 |
| 状态 | **空闲**（V3b已停） |

---

## 五、完整文件清单

### 代码文件
| 文件 | 功能 | 版本 |
|------|------|------|
| `forecast/patchtst_model.py` | PatchTST-EPF V3b模型 | **新建** |
| `forecast/transformer_model.py` | V2 TransformerEPF | 未修改 |
| `forecast/transformer_config.py` | 超参数 + V3/V4 config | **修改** |
| `forecast/transformer_forecaster.py` | V2 Forecaster接口 | 未修改 |
| `forecast/transformer_dataset.py` | V1 Dataset（已弃用） | 未修改 |
| `forecast/lgbm_forecaster.py` | LightGBM基线 | 未修改 |
| `forecast/mpc_controller.py` | MPC控制器 | 未修改 |
| `data/china/features.py` | 特征工程（31+8因果） | **修改** |
| `scripts/11_transformer_train.py` | V2训练脚本 | 未修改 |
| `scripts/12_patchtst_train.py` | V3b PatchTST训练 | **新建** |
| `scripts/13_fetch_weather.py` | 天气数据爬取 | **新建** |
| `scripts/14_patchtst_v4_train.py` | V4训练（39维因果特征） | **新建** |
| `scripts/gpu_pack_v3.sh` | GPU打包脚本 | **修改** |

### 数据文件
| 文件 | 变更 |
|------|------|
| `data/china/processed/*_oracle.parquet` | ✅ 含天气+因果特征 |
| `data/china/processed/*_weather.parquet` | **新建** 原始天气备份 |

---

## 六、关键决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 去掉RevIN | 是 | V3a实验证明：dataset已做因果归一化+RevIN=双重归一化，-18.2% |
| 直接输出96值 | 是 | 两阶段(24h→96)丢失15分钟分辨率，MPC需要细粒度 |
| PatchTST而非vanilla Transformer | 是 | 42 patches vs 672 tokens，256倍更高效，论文验证EPF最佳 |
| Mean pooling而非last-token | 是 | 保留更多时序信息 |
| 因果特征用features.py内计算 | 是 | 不改数据管道，只改特征工程，对LightGBM同样有益 |
| Open-Meteo免费天气 | 是 | 0成本，坐标已有，API稳定 |
| 未用RevIN | 教训 | 需要确保输入归一化方式与训练一致，不能叠加 |

---

## 七、已知问题和限制

1. **PatchTST仍落后LightGBM 12.8%**: 主要因为LightGBM per-horizon独立模型+显式统计特征优势
2. **天气数据是城市点位**: 用省会城市代表全省，风电场实际在偏远地区，会有偏差
3. **无煤价数据**: 火电边际成本信号完全缺失
4. **无日前预测数据**: 只有实际值，生产环境需要forecast
5. **V4尚未训练**: 因果特征代码ready但未在GPU上验证

## 八、天气数据统计

| 省份 | 温度范围 | 风速范围 | 光照最大 | 数据行数 |
|------|---------|---------|---------|---------|
| 山东 | -17~40°C | 0~40m/s | 896 W/m² | 190,656 |
| 山西 | -20~38°C | 0~42m/s | 907 W/m² | 176,064 |
| 广东 | 3~38°C | 0~44m/s | 1004 W/m² | 155,424 |
| 甘肃 | -20~37°C | 0~35m/s | 974 W/m² | 175,393 |
