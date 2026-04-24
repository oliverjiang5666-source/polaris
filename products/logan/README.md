# Logan · 观云

> **新能源场站的"电价感知报价策略"产品**

Logan 解决的问题：新能源场站进入现货市场后，只有功率预测的"老实报"策略会
踩到偏差考核和反向加罚。Logan 在客户已有的功率预测之上加一层 **DA/RT/spread
方向/系统偏差** 预测，生成"偏差方向和价格方向对齐"的最优申报曲线。

## 架构（第一性原理）

```
共享底座（core/）
├── Regime Classifier            12 类日子类型，用 D-1 特征预测明天类型
├── Supply Curve                 DA = f(净负荷)，Isotonic + LGBM 残差 + 分位数
└── Net Load Forecaster          LightGBM 多 horizon → 96 点净负荷预测

四个 head（products/logan/）
├── Head 1 · DA Forecaster       supply_curve(net_load_pred) + residual quantile
├── Head 2 · RT Forecaster       Kalman: RT = DA + AR(1) + drivers + noise
├── Head 3 · Spread Direction    Regime-Conditioned P(sign(RT-DA))
└── Head 4 · System Deviation    反向加罚代理（真实版待客户数据）

产品层
├── Bid Curve Generator          四头输出 + 客户功率预测 → 阶梯报价
└── Evaluator                    结算模型 + 对照"老实报"基线
```

## 为什么这个架构

详细推导见 session 记录。核心要点：

1. **DA 价 ≈ 供给曲线(净负荷) + 残差**。物理结构约束比纯 ML 样本效率高
2. **RT = DA + AR 残差**。分解成零误差锚 + 均值回归残差，Kalman 结构上最优
3. **spread 方向**的分类问题比回归样本效率高 2-3 倍，Regime 提供强先验
4. **四个 head 共享 Regime + 供给曲线**，最大化信息复用

不用神经网络原因：陕甘宁数据量可能只有 1-2 年，深模型过拟合风险；
结构式 + 经典统计在此数据规模下最优。

## 代码组织

```
energy-storage-rl/
├── core/                                  ← 共享底座
│   ├── calendar_features.py
│   ├── regime_classifier.py
│   ├── supply_curve.py
│   └── net_load_forecaster.py
│
└── products/logan/
    ├── da_forecaster.py                   Head 1
    ├── rt_forecaster.py                   Head 2
    ├── spread_direction.py                Head 3
    ├── system_deviation.py                Head 4
    ├── bid_curve_generator.py             最终产品
    ├── evaluator.py                       回测
    └── scripts/
        ├── 01_build_supply_curve.py       诊断 + 可视化
        ├── 02_train_all_heads.py          训练
        └── 03_backtest_gansu.py           端到端回测
```

## 快速运行

```bash
# 1. 验证 supply curve 能拟合（甘肃 test RMSE 目标 < 90 元/MWh）
PYTHONPATH=. python3 products/logan/scripts/01_build_supply_curve.py

# 2. 训练所有 head → 保存到 models/logan/gansu/
PYTHONPATH=. python3 products/logan/scripts/02_train_all_heads.py --province gansu

# 3. 端到端回测：Logan vs "老实报"
PYTHONPATH=. python3 products/logan/scripts/03_backtest_gansu.py --province gansu --days 90
```

## 需要客户提供的数据（当前是 proxy）

当前用甘肃 parquet 里的新能源总出力作**功率预测的 proxy**（+ 8% 高斯噪声
模拟预测误差）。上线时需要换成：
- 客户功率预测历史（3 年，15 分钟）
- 客户功率预测实时接入（D-1 10:00 前推送 D 日 96 点预测）

见 session 记录里给中能建的《数据需求清单》。

## 待完善

- [ ] `system_deviation.fit_full()`：全省级实际 vs 计划数据到位后实现
- [ ] Bid Curve Generator：加入中长期合约约束（目前只优化现货申报）
- [ ] RT Forecaster：当前是离线 fit，上线后接实时数据流 + online Kalman update
- [ ] 多省（陕西、宁夏）适配：数据到位后复用代码
- [ ] 考核规则参数化：按省写 `settlement_rules/{province}.yaml`
