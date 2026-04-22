# Transformer电价预测系统 — 完整交接文档

---

## 一、项目背景

储能电站交易策略优化系统。核心流程：预测未来电价 → LP线性规划求最优充放电 → MPC滚动执行。

原有方案用LightGBM做价格预测，在4省回测中3/4省超越规则策略。本次工作是将预测模型从LightGBM升级为Transformer，基于EPF-Transformer论文(arXiv:2403.16108)。

电池参数：200MW/400MWh，效率90%（单程94.87%），降解成本2元/MWh，15分钟粒度。

---

## 二、已有的LightGBM MPC基线（之前session的结果）

| 省份 | LightGBM MPC(元/年) | vs 规则策略 | Oracle上限 |
|------|-------------------|-----------|-----------|
| 山东 | 43,736,963 | +18.2% | 81,347,077 |
| 山西 | 55,010,000 | +44.8% | — |
| 广东 | 21,970,000 | +32.6% | — |
| 甘肃 | 31,290,000 | -16.0% | — |

LightGBM架构：11个独立horizon模型(h=1,2,4,8,16,24,32,48,64,80,96)，每个horizon训练一个LGBMRegressor。输入31维手工特征+price_lag_96+target_hour编码=34维。中间步长线性插值。训练时间~18秒/省。

31维特征定义在 `data/china/features.py`：
- 价格特征14维：rt_price, lag_1~4, ma_4/16/96, std_16/96, trend, percentile, ma_ratio, ma4_ratio
- 时间特征6维：hour/weekday/month的sin/cos编码
- 中国市场特征11维：da_price, da_rt_spread, da_price_ma_ratio, load_norm, load_change, renewable_penetration, wind_ratio, solar_ratio, net_load_norm, tie_line_norm, temperature_norm

---

## 三、Transformer设计决策记录

### 决策1：模型架构 — 纯Transformer编码器，双路径

参考论文：arXiv:2403.16108

采用论文的双路径架构：
- Path A：价格历史序列 → Linear projection → Positional Encoding → TransformerEncoder → 取最后token
- Path B：外源变量 → 2层MLP → embedding
- 拼接后 → 3层MLP → 输出

论文用ReLU，我们改用GELU和Pre-LayerNorm（更稳定训练）。

### 决策2：序列长度 — 672步（7天）

论文用336-672步（14-28天小时级数据）。我们用672步（7天×96步/天15分钟级），匹配论文上界。

实验中因M4 MPS太慢，先用192步（2天）做了一轮测试，再在GPU上跑672步。

### 决策3：输出粒度 — 直接输出96个15分钟值

论文输出24个小时级值。我们输出96个15分钟值，因为LP Oracle在15分钟粒度上运行。

这是与论文的最大差异：输出维度4倍，任务难度更高。

### 决策4：序列输入特征 — 8通道

第一版（V1）只用3通道：rt_price, da_price, spread
第二版（V2）扩展到8通道：rt_price, da_price, spread, load_norm, renewable_penetration, net_load_norm, wind_ratio, solar_ratio

### 决策5：外源变量（Path B）— 17维

使用现有features.py中的17个特征作为外源输入。没有使用LightGBM的全部31维特征（ma_ratio、lag等统计特征没有包含在Path B中）。

### 决策6：归一化 — 滑动窗口自适应

与论文一致：减去前24小时（96步）的均值并除以标准差。价格通道(0,1,2)用自适应归一化，非价格通道(3-7)在features.py中已经归一化。

### 决策7：训练策略

- Loss: MSE
- Optimizer: Adam(lr=1e-4, weight_decay=1e-5)
- Scheduler: CosineAnnealingWarmRestarts(T_0=10)
- 梯度裁剪: max_norm=0.3
- Early stopping: patience=15 on val MAE
- 数据增强: 高斯噪声(std=0.02) + 窗口抖动(±4步)
- 甘肃：rolling_window=365天，lr=5e-5

### 决策8：超参数

embed_dim=128, n_heads=4, n_layers=4, ff_dim=512, dropout=0.15

参数量：1,256,800

论文的搜索范围：embed_dim=64-512, heads=1-8, layers=4-6。我们取了中间值，没有做超参搜索。

### 决策9：数据切分

山东为例（1986天总数据）：
- Train: 1441天
- Val: 180天
- Test: 365天

### 决策10：MPC接口 — TransformerForecaster是LGBMForecaster的drop-in替换

TransformerForecaster实现相同的 `predict(features_t, idx, horizon=96) -> np.ndarray` 接口。MPC控制器代码零修改。

---

## 四、实验结果

### 实验1：M4 MPS，192步序列，3通道（V1）

```
训练样本：34,512（stride=4）
序列特征：3通道（rt_price, da_price, spread）
设备：Apple M4 MPS
训练：69 epochs, early stop, best val MAE=91.6元/MWh, 耗时约70分钟
```

| 方法 | Revenue(元) | vs Threshold |
|------|-----------|-------------|
| Threshold | 37,014,828 | 基准 |
| LightGBM MPC | 43,736,963 | +18.2% |
| Transformer MPC | 36,685,169 | **-0.9%** |

### 实验2：5090 CUDA，672步序列，8通道（V2）

```
训练样本：137,568（stride=1）
序列特征：8通道（+load, renewable, net_load, wind, solar）
设备：RTX 5090 32GB CUDA
训练：21 epochs, early stop, best val MAE=93.1元/MWh, 耗时2262秒
```

| 方法 | Revenue(元) | vs Threshold |
|------|-----------|-------------|
| Threshold | 37,014,828 | 基准 |
| LightGBM MPC | 43,736,963 | +18.2% |
| Transformer MPC | 37,342,170 | **+0.9%** |

### 对比

| 配置 | val MAE | MPC Revenue | vs LightGBM |
|------|---------|-------------|-------------|
| V1 (192步, 3通道, M4) | 91.6元/MWh | 36,685,169 | -16.1% |
| V2 (672步, 8通道, 5090) | 93.1元/MWh | 37,342,170 | -14.6% |

V2的val MAE反而比V1高（93.1 vs 91.6），但MPC revenue更好。val MAE和MPC revenue不完全正相关。

### 正在进行

山西、广东、甘肃在5090上训练中。
- GPU服务器：AutoDL 西北B区 403机
- SSH：`ssh -p 23074 root@connect.westd.seetacloud.com`，密码 `pLr4+UXpvt2x`
- 日志：`/root/autodl-tmp/gpu_training.log`
- 查看进度：`tail -30 /root/autodl-tmp/gpu_training.log`
- 山西截至最新：epoch 6, val MAE=133.2元/MWh，还在下降

---

## 五、论文与我们实现的对比

### 论文结果（arXiv:2403.16108）

| 市场 | MAE(€/MWh) | 均价(€/MWh) | 相对MAE |
|------|-----------|-----------|---------|
| NordPool | 2.33 | ~40 | 5.8% |
| PJM | 3.67 | ~30 | 12.2% |
| EPEX-DE | 4.03 | ~50 | 8.1% |
| EPEX-BE | 6.54 | ~45 | 14.5% |
| EPEX-FR | 4.91 | ~35 | 14.0% |

### 我们的结果

| 市场 | MAE(元/MWh) | 均价(元/MWh) | 相对MAE |
|------|-----------|-----------|---------|
| 山东 | 93.1 | ~340 | 27.4% |

### 差异来源

| 维度 | 论文 | 我们 |
|------|------|------|
| 输出粒度 | 24值（小时级） | 96值（15分钟级） |
| 序列长度 | 336-672步（小时级） | 672步（15分钟级，覆盖7天 vs 论文14-28天） |
| 数据量 | 6年（4年训练） | 5.4年（4年训练） |
| 市场波动性 | 欧洲成熟市场 | 中国现货（山东18%时段负电价） |
| 超参搜索 | 每个市场单独调优 | 固定配置，未搜索 |
| 外源变量 | 论文用了需求预测等 | 我们用17维基础特征 |
| Path B特征 | 包含价格统计特征 | 不包含LightGBM的统计特征（ma_ratio等） |

---

## 六、参考论文

| # | 论文 | 发表 | 开源 | 与本项目的关系 |
|---|------|------|------|-------------|
| 1 | Lago et al. "Forecasting day-ahead electricity prices" | Applied Energy 2021 | [epftoolbox](https://github.com/jeslago/epftoolbox) | EPF领域标准benchmark，定义了DNN/LEAR基线和评估方法（DM检验） |
| 2 | González & Portela "A Transformer approach for EPF" | arXiv:2403.16108 | [epf-transformers](https://github.com/osllogon/epf-transformers) | 我们的Transformer架构直接参考此论文，双路径设计 |
| 3 | Nie et al. "A Time Series is Worth 64 Words" (PatchTST) | ICLR 2023 | [PatchTST](https://github.com/yuqinie98/PatchTST) | Patch机制可将672步压缩为42个token，降低attention复杂度256倍 |
| 4 | Weron "Electricity price forecasting: A review" | Int J Forecasting 2014 | 无 | 经典综述，calibration window、spikes处理、概率预测 |
| 5 | Hong et al. "Energy Forecasting: A Review and Outlook" | IEEE 2020 | 无 | GEFCom组织者综述，多模型集成持续胜出，概率预测比点预测更重要 |
| 6 | THUML Time-Series-Library | — | [TSLib](https://github.com/thuml/Time-Series-Library) | 30+种时序模型统一框架，含PatchTST/iTransformer/TimesNet |

---

## 七、我们做了哪些决策，每个决策的论文依据和事实对比

| 决策 | 论文做法 | 我们的做法 | 事实差异 |
|------|---------|----------|---------|
| 输出96值 | 输出24值 | 输出96值 | 论文未验证96值输出；我们的相对MAE(27%)远高于论文(6-15%) |
| 双路径架构 | 历史价格路径+外源变量路径 | 相同 | 架构一致 |
| 序列长度672 | 336-672步(小时级=14-28天) | 672步(15分钟级=7天) | 虽然步数相同，但我们覆盖的时间范围是论文的1/2~1/4 |
| 8通道序列输入 | 仅历史价格序列 | 价格+负荷+新能源等8通道 | 我们加了更多序列特征，论文只用价格 |
| Path B 17维 | 论文用外源变量(需求预测等) | 17维基础特征 | 我们没有把LightGBM的统计特征(ma_ratio等)放入Path B |
| 固定超参 | 每个市场单独搜索最优超参 | embed=128,heads=4,layers=4 | 论文为每个市场找了不同的最优配置 |
| MSE loss | 论文未明确说明 | MSE | — |
| Adam lr=1e-4 | 1e-4~1e-5 | 1e-4 | 在论文范围内 |
| 归一化 | 减去前一天均值和标准差 | 减去前96步(24h)均值和标准差 | 基本一致 |
| Early stopping patience=15 | 论文用固定epoch+验证集选模型 | patience=15 | 我们的early stopping可能太早了（epoch 21 stop，论文跑更多epoch） |
| 无LEAR基线 | 必须有LEAR基线对比 | 没有跑LEAR | 缺少标准基线 |
| 无DM检验 | 必须做显著性检验 | 只比了revenue | 缺少统计显著性验证 |

---

## 八、后续要做的步骤

### 步骤1：等待4省GPU训练结果

查看方式：
```bash
sshpass -p 'pLr4+UXpvt2x' ssh -p 23074 root@connect.westd.seetacloud.com "tail -30 /root/autodl-tmp/gpu_training.log"
```
预计完成时间：山西/广东/甘肃各约1-2小时，总计约4-6小时。

### 步骤2：把LightGBM的31维特征全部喂给Transformer Path B

当前Path B只用了17维基础特征，缺少了LightGBM最强的特征（ma_ratio、lag、std等）。

改动：
```python
# 在 scripts/11_transformer_train.py 中
EXO_COLS = FEATURE_COLS  # 从17维改为全部31维

# 在 forecast/transformer_config.py 中
n_exo_features: int = 31  # 从17改为31
```

然后重新训练4省，对比17维 vs 31维的MPC revenue差异。

### 步骤3：改输出为24值（小时级）+ MLP细化到96值

当前直接输出96个15分钟值。改为两阶段：
1. Transformer输出24个小时级值（与论文一致）
2. 加一个小MLP将24值展开到96值

改动：修改 `transformer_model.py` 的 output_head，加入中间层。

### 步骤4：跑PatchTST对比

PatchTST(ICLR 2023)将672步分成patch(每16步一个=42 tokens)，可能更适合长序列。

选项A：从头实现PatchTST模型
选项B：用Time-Series-Library(https://github.com/thuml/Time-Series-Library)的PatchTST实现，适配我们的数据

### 步骤5：做超参搜索

论文为每个市场单独搜索了最优超参。我们目前用固定配置(embed=128, heads=4, layers=4)。

搜索空间：
- embed_dim: [64, 128, 256]
- n_heads: [2, 4, 8]
- n_layers: [2, 4, 6]
- lr: [1e-3, 5e-4, 1e-4, 5e-5]
- sequence_length: [192, 384, 672]

每个配置训练一轮（~30分钟/省在5090上），12-24个配置×4省 = 需要约20-50 GPU小时。

### 步骤6：加LEAR基线

EPFToolbox论文(Lago et al. 2021)要求必须有LEAR(LASSO Estimated AutoRegressive)基线。

安装：`pip install epftoolbox`
实现：用epftoolbox的LEAR模型在我们的数据上训练和预测，接入MPC评估。

### 步骤7：集成方法

将LightGBM和Transformer的预测做加权平均：
```python
final_pred = alpha * lgbm_pred + (1 - alpha) * transformer_pred
```
在验证集上搜索最优alpha。GEFCom竞赛冠军都用集成方法。

### 步骤8：概率预测

从点预测(输出1个值)改为概率预测(输出分位数P10/P50/P90)。Loss改为Quantile Loss。MPC可以利用不确定性做鲁棒优化。

---

## 九、文件清单

### 代码文件

| 文件 | 功能 | 状态 |
|------|------|------|
| `forecast/transformer_model.py` | TransformerEPF nn.Module定义 | 完成 |
| `forecast/transformer_dataset.py` | EPFDataset滑动窗口Dataset | 完成（V1用，V2在训练脚本中） |
| `forecast/transformer_config.py` | 超参数dataclass + 省级override | 完成 |
| `forecast/transformer_forecaster.py` | TransformerForecaster（与LGBMForecaster同接口） | 完成 |
| `scripts/11_transformer_train.py` | 训练+MPC评估（含EPFDatasetV2: 8通道） | 完成 |
| `scripts/gpu_pack.sh` | 打包项目给GPU服务器 | 完成 |
| `scripts/gpu_setup.sh` | GPU服务器环境安装 | 完成 |
| `scripts/gpu_run.sh` | GPU训练启动脚本 | 完成 |
| `config.py` | 电池参数、动作空间 | 未修改 |
| `oracle/lp_oracle.py` | LP线性规划求解 | 未修改 |
| `forecast/mpc_controller.py` | MPC控制器+电池仿真 | 未修改 |
| `forecast/lgbm_forecaster.py` | LightGBM预测器 | 未修改 |
| `data/china/features.py` | 31维特征工程 | 未修改 |
| `scripts/09_mpc_eval.py` | LightGBM MPC评估 | 未修改 |

### 数据文件

| 文件 | 内容 |
|------|------|
| `data/china/processed/shandong_oracle.parquet` | 山东 190,656行 (1986天) |
| `data/china/processed/shanxi_oracle.parquet` | 山西 (1834天) |
| `data/china/processed/guangdong_oracle.parquet` | 广东 (1619天) |
| `data/china/processed/gansu_oracle.parquet` | 甘肃 (1827天) |

各省数据来源和质量问题见 `HANDOFF_CHINA.md`。

### GPU服务器

| 项目 | 信息 |
|------|------|
| 平台 | AutoDL |
| 位置 | 西北B区 403机 |
| GPU | RTX 5090 × 1, 32GB VRAM |
| SSH | `ssh -p 23074 root@connect.westd.seetacloud.com` |
| 密码 | `pLr4+UXpvt2x` |
| Python | `/root/miniconda3/bin/python` (3.12) |
| PyTorch | 2.8.0+cu128 |
| 项目路径 | `/root/autodl-tmp/` |
| 训练日志 | `/root/autodl-tmp/gpu_training.log` |
| 计费 | ¥2.93/时，按量计费 |
| 状态 | 正在跑4省训练（山东完成，山西/广东/甘肃进行中） |

---

## 十、运行命令

### 查看GPU训练进度
```bash
sshpass -p 'pLr4+UXpvt2x' ssh -p 23074 root@connect.westd.seetacloud.com "tail -30 /root/autodl-tmp/gpu_training.log"
```

### 查看关键结果
```bash
sshpass -p 'pLr4+UXpvt2x' ssh -p 23074 root@connect.westd.seetacloud.com "grep -E '(Revenue|>>>|====|Method|FINAL)' /root/autodl-tmp/gpu_training.log"
```

### 本地跑LightGBM基线
```bash
cd /Users/jjj/Desktop/工作/电力交易/energy-storage-rl
PYTHONPATH=. python3 scripts/09_mpc_eval.py --all
```

### 本地快速测试Transformer（192步）
```bash
# 先改 forecast/transformer_config.py: window_size=192, device="mps"
PYTHONPATH=. python3 scripts/11_transformer_train.py --province shandong
```

### 重新打包上传到GPU
```bash
bash scripts/gpu_pack.sh
sshpass -p 'pLr4+UXpvt2x' scp -P 23074 energy-storage-gpu.tar.gz root@connect.westd.seetacloud.com:/root/autodl-tmp/
```
