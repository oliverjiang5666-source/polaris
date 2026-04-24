# Polaris · 山东版（独立新型储能）

> 基于《山东电力市场规则（试行）2026年3月征求意见稿》（363页）的独立新型储能报价决策

---

## 与 Polaris 通用版的差异

| 维度 | Polaris 通用 | Polaris 山东 |
|---|---|---|
| 价格信号 | 省均价 `rt_price` | 物理节点 15min LMP（§3.2.28-29）|
| 结算模型 | 单价过账 `Σ power × price × dt` | **Two-Settlement + CfD**（§14.3.4/14.5.6）|
| 输出形态 | 96 点连续 power 或 5 动作 | **充放各 5 段阶梯 bid curve**（§7.2.8）|
| 收入来源 | 电能量套利 | 电能量 + **容量补偿**（§3.4.6）+ **AGC 调频**（§11.1）|
| DP state | SoC | SoC（AGC 作为 action 扩展）|
| DP action | 5 离散或连续功率 | **(p_arb, c^AGC_up, c^AGC_down) 三维**（§11.2.3 储能调频预留）|

---

## 模块结构

```
products/polaris_shandong/
├── settlement_rules/
│   └── shandong.yaml              # 全部规则数据化（12 大章节）
├── bid_curve.py                   # StorageBidCurve（充 5 + 放 5 段）
├── compliance.py                  # validate / enforce（§7.2.8）
├── evaluator.py                   # Two-Settlement + CfD 结算引擎
├── capacity_compensation.py       # 容量补偿计算（§3.4.6 第四项）
└── scripts/
    └── 01_verify_rules.py         # smoke test 全链路

# 配套升级（在 Polaris 下）
products/polaris/optimization/vfa_dp/
    tensor_dp_joint.py              # 3D action Tensor DP (arb + AGC up + AGC down)
```

---

## 核心数据流

```
Tensor DP Joint (3D action)
    → 96 点 (p^arb, c^up, c^down) 计划
    → convexification (bid_curve.build_from_tensor_dp_plan)
    → StorageBidCurve (充 5 + 放 5 段)
    → compliance.enforce
    → 封存申报 (D-1 15:00)

Actual run day:
    → cleared_power(bid, LMP)  各时段按节点价清算
    → evaluator.settle_from_bid_curve()
    → R_实时 + R_日前 CfD + R_中长期 CfD + AGC + 容量补偿
```

---

## 关键规则条款速查

| 主题 | 条款 | 公式/要求 |
|---|---|---|
| 申报时点 | §6.3.2 §7.2.3 | D-1 09:45 首次申报，15:00 经济出清截止 |
| Bid curve 格式 | §7.2.8 | 充/放各 5 段 ≥ 2 MW，单调非递减 |
| 结算价 | §3.2.28 §3.2.29 | 发/用 15min 节点 LMP（独立储能用电侧特例）|
| SCUC/SCED | §8.4.4 §7.2.16 | 储能充费 + 放费作成本，max 社会福利 |
| 两结算公式 | §14.3.4 §14.3.5 | R_实时 + R_日前CfD + R_中长期CfD |
| 容量补偿 | §3.4.6(四) | 日容量 = P_放电 × K × H / 24 |
| AGC 调频 | §11.1.2 §14.8.3 | 费用 = 里程 × 性能系数 × 出清价 |
| AGC 储能预留 | §11.2.3 | 日前申报可调功率+SoC 基础上预留 |
| 封存报价 | §9.2.1 §10.3.1 | 日内/实时用日前封存申报，日内不发布价 |
| 调度强制调用 | §8.5.11 §14.10.3 | 紧张时段可覆盖市场出清，按损耗公式补偿 |

---

## LMP proxy 说明（TODO）

当前实现接口用 LMP：
```python
evaluator.settle_from_bid_curve(
    da_lmp_gen_96=...,   # 日前发电侧节点价
    rt_lmp_gen_96=...,   # 实时发电侧节点价
    da_lmp_user_96=...,  # 日前用户侧节点价
    rt_lmp_user_96=...,  # 实时用户侧节点价
    rt_unified_96=...,   # 实时市场统一结算点价
)
```

现阶段 `data/china/processed/shandong_oracle.parquet` 的 `rt_price` 是**省均价**，不是节点价。`01_verify_rules.py` 用同一个 `rt_price` 作所有五个 LMP 字段的 proxy。

真实节点 LMP 数据到位后，只需换数据源，不改代码。

---

## Smoke test 结果

```bash
PYTHONPATH=. python3 products/polaris_shandong/scripts/01_verify_rules.py
```

| 策略 | 收入 |
|---|---|
| Tensor DP 单价过账（原 Polaris）| ¥97,577 |
| 山东 Two-Settlement (DA=RT=LMP proxy) | ¥99,641 |
| 山东 Two-Settlement (5% DA-RT spread) | ¥107,143 |

- 同价时两者差 ¥2,064 = degradation cost（evaluator 不扣降解）
- 有 DA-RT spread 时 CfD 贡献 ¥4,303 额外收益

---

## Phase 1 范围（本 session）

✅ 规则数据化（12 章 shandong.yaml）
✅ Bid curve 格式 + compliance + enforce
✅ Two-Settlement + CfD evaluator
✅ 容量补偿公式
✅ Tensor DP Joint（3D action：p_arb + AGC up + AGC down）
✅ Smoke test 全链路验证

## Phase 2 待办（需外部数据或客户配合）

- [ ] 真实节点 LMP 数据接入（向山东电力交易中心申请）
- [ ] AGC 市场出清价历史（用于训练场景）
- [ ] AGC 性能系数 K_pd（调度测试认定后的真实值）
- [ ] 中长期合约实际信息（客户提供合约电量 + 价格 + 分解曲线）
- [ ] 容量补偿电价（省发改委政策文件）
- [ ] 全年 walk-forward 回测（含山东 2025 自然年数据）
- [ ] Tensor DP Joint vs 原 Tensor DP 的 capture 对比
- [ ] PMOS 交易系统的报价 XML 格式对齐

---

## 快速启动

```bash
# 从 repo 根目录
cd /Users/jjj/Desktop/工作/电力交易/energy-storage-rl/

# 跑 smoke test（合成数据）
PYTHONPATH=. python3 products/polaris_shandong/scripts/01_verify_rules.py

# 跑 Tensor DP Joint 的 sanity check（AGC 和没 AGC 的对比）
PYTHONPATH=. python3 -m optimization.vfa_dp.tensor_dp_joint
```
