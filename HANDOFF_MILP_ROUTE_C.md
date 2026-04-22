# Route C (Stochastic MILP) 实验交接文档

> **接手时间**：2026-04-21 11:30（上海时间）
> **上一个 session 交接**：Claude Opus
> **实验状态**：两台 Mac 同时跑，Mac #1 今晚完，Mac #2 预计 +30h

---

## 🎯 这是什么项目

用户（JJJ）要做中国电力 AI 交易公司（对标 enspired）。核心储能调度算法目前用 **Regime V3 Stochastic DP**（他自创），捕获率山东 64.8%。要验证 **Route C: Stochastic MILP**（工业金标准，enspired/Fluence/Tesla Autobidder 都用）是否更好。

**研究问题**：在**同样数据、同样场景集**下，MILP 比 Regime V3 DP 能多捕获多少收益？

---

## 🔧 当前实验（正在跑）

### Mac #1（本地，jjj, M4 10 核 24GB）
- **PID 61875**（nohup 启动，caffeinate 防睡眠）
- **脚本**：`scripts/27_milp_shandong_all.py`
- **实验**：13 个山东消融（K 消融 / dev_bound 消融 / degradation 消融 / scen 方法）
- **日志**：`runs/milp_shandong.log`
- **结果输出**：`runs/milp_experiments/shandong_*.pkl`
- **进度查看**：`ps -p 61875`, `ls runs/milp_experiments/shandong_*.pkl | wc -l`

**最新状态（2026-04-21 11:30）**：
- 完成 9/13（baseline, dev_00, dev_20, dev_50, deg_08, deg_20, deg_50, K_050, K_100）
- 正在跑 K_500（K=500，~15 分钟/季度）
- 预计 14:00 前全部完成

### Mac #2（局域网 jiang@192.168.0.21, M4 10 核 24GB）
- **PID 38438**（nohup）
- **脚本**：`~/energy-storage-rl/scripts/28_milp_other_provinces.py`
- **实验**：9 个跨省实验（山西/广东/甘肃 的 baseline + dev_00/dev_20）
- **日志**：`~/energy-storage-rl/runs/milp_other.log`
- **进度查看**：`ssh jiang@192.168.0.21 'ps -p 38438; ls ~/energy-storage-rl/runs/milp_experiments/*.pkl | wc -l'`

**最新状态**：
- 完成 2/9（shanxi_baseline, guangdong_baseline）
- 正在跑 gansu_baseline
- Mac #2 **比 Mac #1 慢 4 倍**（46s/day vs 11s/day）——**原因未查清**，可能是某些后台进程或温度节流。预计 **40+ 小时**才全部完成

### 监控
- **Cron**：`crontab -l` 有 `*/30 * * * * bash scripts/monitor_experiments.sh`
- **Daemon**：`monitor_daemon.sh` 作为 nohup 后台进程（PID 63726），每 30 分钟打一次状态到 `runs/monitor.log`
- **告警文件**：`runs/ALERT.log`（如果有任一任务挂掉会写入）
- **查看最新**：`tail -50 runs/monitor.log`

### 防干扰
- Mac #1 和 Mac #2 **都已禁用自动更新**（defaults del AutoInstallProductKeys / FirstInstallTonightDateDictionary）
- Mac #1 已 `pmset -a sleep 0 disksleep 0`
- Mac #2 同上
- **不要手动重启任何一台 Mac！会丢失进度**

### 认证
- Mac #2 密码：（见 1Password / 团队密码管理器，条目名 "energy-rl-mac2"）
- Mac #1 `sudo` 密码：（同上）
- 用户邮箱：（见联系方式文档，不在此文件留存）

---

## 📊 已知初步结果（K_500 还没跑完）

### 山东完整回测（365 天，K=200 默认）

| 实验 | K | dev_bound | deg | 收入 / 年 |
|---|---|---|---|---|
| **baseline** | **200** | 10% | ¥2 | **¥2,834 万** |
| dev_00 | 200 | 0% | ¥2 | （看 pkl）|
| dev_20 | 200 | 20% | ¥2 | （看 pkl）|
| dev_50 | 200 | 50% | ¥2 | （看 pkl）|
| deg_08/20/50 | 200 | 10% | 8/20/50 | （看 pkl）|
| **K_050** | **50** | 10% | ¥2 | **¥3,903 万** |
| **K_100** | **100** | 10% | ¥2 | **¥4,054 万** |

### 🚨 重大反直觉发现

**K=50 和 K=100 比 K=200 多挣 38-43%！**

理论上 K 越大 Monte Carlo 越准，应该**单调不减**。但实测相反。

**三个候选假设**（正在调查）：
1. **Bootstrap 尾部坍缩**：K 大时主导 regime 的样本占比过高（某天 7 号 regime = 92%），稀释尾部高波动日的价值 → MILP 看到"更平滑的期望"→ DAM 承诺更保守
2. **场景种子 bug**：`rng = np.random.default_rng(seed=42)` 在每次 `prepare_tasks` 调用里重置。K=200 和 K=100 前 100 个样本应该相同——但 MILP 结果差这么多，**可能不是数值噪声**
3. **MILP time_limit 截断**：K=500 可能达到 time_limit，但 baseline K=200 应该在时限内

**调查脚本**：`scripts/30_k_mystery_investigation.py`
- 跑 5 个 K 值 × 2 seed × 10 天
- 记录场景统计 + DAM 承诺 + actual revenue
- ⚠️ **不要在 Mac #1 跑完前运行**（会竞争 CPU）

---

## ✅ 本次 session 完成的事

### 1. 修复 `oracle/lp_oracle.py` 的 `solve_day_dual()`
- **之前**：用启发式（基于价差符号）决定偏差方向
- **之后**：正经 LP（3n 变量：charge/discharge/dam，带偏差界和 SoC 约束）
- **验证**：5 个 sanity test 全过
- **警告**：当 `deviation_bound > 20%` 时，LP 会发现"虚拟短空 DAM + 实际不执行"的财务套利（副总警告的灰色行为）。已在 docstring 加 caveat。推荐 `dev_bound ∈ [0.05, 0.15]`
- **影响**：
  - 不影响正在跑的 MILP 实验（MILP 用 `optimization/milp/milp_formulation.py`，不走 solve_day_dual）
  - 重新跑 `scripts/24_dual_settlement_backtest.py`，**昨天的 Oracle 数字要改**

### 2. 新 Oracle 数字（Q4 2024，用**正确**的 LP）
| 省 | 单结算(RT) | 双结算(LP) | 变化 |
|---|---|---|---|
| 山东 | 2098 万 | **2225 万** | 旧 heuristic **低估 12%** |
| 广东 | 1691 万 | **1472 万** | 旧低估 19% |
| 山西 | 2304 万 | **2325 万** | 旧低估 12% |

**修正后的业务结论**（要写进 pitch deck）：
- 山东：单结算比可实现 Oracle **低估** 6%（双结算能额外套利）
- 广东：单结算比可实现 Oracle **高估** 15%（广东 RT 波动大，RT-only 不可实现）
- 山西：单结算 ≈ Oracle

---

## 🎯 下一步（按优先级）

### P0 — 等实验跑完（今晚 + 接下来 2 天）

**Mac #1 跑完后（今晚）**：
1. 运行 `scripts/30_k_mystery_investigation.py`，诊断 K 反直觉现象
2. 运行分析脚本（还没写，见下面 P1 #4）

**Mac #2 跑完前**：
- 每天查 `tail -50 runs/monitor.log` 看是否有 ALERT
- 若 Mac #2 挂了：
  ```bash
  ssh jiang@192.168.0.21
  cd ~/energy-storage-rl
  tail -100 runs/milp_other.log  # 看为啥挂
  # 如果只是被杀了，resume：
  PYTHONPATH=. nohup ~/miniforge3/bin/python -u scripts/28_milp_other_provinces.py >> runs/milp_other.log 2>&1 &
  disown
  ```

### P1 — 结果分析（Mac #1 完成后）

1. **rsync Mac #2 结果回来**：
   ```bash
   rsync -avz jiang@192.168.0.21:~/energy-storage-rl/runs/milp_experiments/ runs/milp_experiments/
   ```

2. **聚合分析**：写 `scripts/29_milp_analyze.py` 做：
   - 所有 22 个实验的 (省 × 配置 × 季度 × 收入) pivot
   - 对比 MILP vs Regime V3 的捕获率
   - 绘图（matplotlib）

3. **调查 K-mystery**：
   - 跑 `scripts/30_k_mystery_investigation.py`
   - 根据结果决定是 bug 还是现象
   - 如果是 bug：修 `optimization/milp/scenario_generator.py`
   - 如果是现象：考虑上 importance sampling / K-medoids 场景约简

4. **生成报告**：markdown 格式，包含：
   - 核心数字（MILP vs Regime V3 对比）
   - K 现象结论
   - 决策：生产部署 MILP 还是保持 Regime V3

### P2 — Route C v1.0 设计（下周）

专家推荐的 Route C "完整版" 组件（当前只做了 Δ1）：

```
Route C v1.0 = Stochastic MILP + 多市场 + 跨日 TVF + 报价曲线 + 高级场景

当前 v0.1（正在跑）：
  ✅ Stochastic MILP (DAM + RT)
  ❌ 多市场（AGC 缺）              ← 增量 +15-25%
  ❌ 跨日 TVF（仍每天 reset）       ← 增量 +5-8%
  ❌ 报价曲线（只点竞价）           ← 增量 +32%
  ❌ 高级场景（只 Bootstrap）       ← 增量 +3-8%
  ⚠️  CVaR（实现了但 baseline 没开）
```

**优先做的是**：AGC 联合 MILP（代码框架已在 `optimization/agc_dp.py`，需移到 MILP 里）。这是最大 ROI 的升级。

### P3 — 给 Thomas Lee 发邮件（用户批准过）

用户之前说"可以 按你的建议做吧"——我建议的 Path 1（技术咨询）邮件模板在上一轮对话里。

**重要上下文**：
- Thomas Lee 是 MIT IDSS PhD student（2022- 3 年级）
- 沃顿 M&T 本科 + JPMorgan/AQR 经历 + MIT Martin Trust Center 创业兴趣
- 导师 Andy Sun（MIT Sloan tenured）
- 联系：`tlee@mit.edu`（推测）或 https://idss.mit.edu/staff/thomas-lee/
- 他刚发 `arXiv 2511.15629`（GPU-DP + 电池套利）

建议邮件走 Path 1：先谈技术咨询（$300-500/h × 4h/月），不谈全职。

---

## 📁 代码结构地图

```
energy-storage-rl/
├── HANDOFF_*.md                    ← 各个阶段的交接文档
├── HANDOFF_MILP_ROUTE_C.md         ← 本文件
│
├── optimization/
│   ├── milp/                       ⭐ 新（Route C 实验主框架）
│   │   ├── data_loader.py          数据加载（复用 parquet）
│   │   ├── scenario_generator.py   Bootstrap/Quantile 场景
│   │   ├── milp_formulation.py     ⭐ Pyomo 两阶段 LP
│   │   ├── stochastic_solver.py    HiGHS/Gurobi/COPT 适配
│   │   ├── parallel_backtest.py    multiprocessing walk-forward
│   │   └── experiment_runner.py    实验矩阵编排
│   ├── agc_dp.py                   DP + AGC 联合（已实现但不在 MILP 里）
│   ├── cvar_dp.py                  CVaR DP（Modification #5）
│   └── mlt_allocator.py            MLT 约束（Modification #2）
│
├── oracle/
│   └── lp_oracle.py                ⭐ 今天修过 solve_day_dual（正经 LP）
│
├── forecast/
│   └── mpc_controller.py           ⭐ _step_battery_dual 也是今天加的
│
├── env/
│   └── battery_physics.py          物理升级（SoH, 效率, 温度）
│
├── scripts/
│   ├── 22_regime_v3_allprov.py     Regime V3 baseline（对照组）
│   ├── 24_dual_settlement_backtest.py  Oracle 双结算对比（已修复）
│   ├── 25_regime_v3_dual_settlement.py Regime V3 双结算版
│   ├── 27_milp_shandong_all.py     ⭐ Mac #1 正在跑
│   ├── 28_milp_other_provinces.py  ⭐ Mac #2 正在跑
│   ├── 29_milp_analyze.py          ❌ 还没写（P1 任务）
│   ├── 30_k_mystery_investigation.py ⭐ 今天新加（等 Mac #1 完跑）
│   ├── monitor_experiments.sh      监控脚本（cron）
│   └── monitor_daemon.sh           后台 daemon
│
├── runs/
│   ├── milp_experiments/           ⭐ 每个实验一个 .pkl + _per_day.csv
│   ├── milp_shandong.log           Mac #1 日志
│   ├── milp_other.log              Mac #2 日志
│   ├── monitor.log                 监控每 30 分钟记录
│   ├── ALERT.log                   挂了才会有
│   └── k_mystery/                  调查脚本输出（还没有）
│
└── data/china/processed/
    └── {shandong,shanxi,guangdong,gansu}_oracle.parquet  ⭐ 主数据
```

---

## 🔑 关键技术决策记录

### 为什么跑 Route C 实验
- 专家输入（昨天会议）：Route C 是**工业金标准**
  - Fluence Mosaic（16 GW）、Wärtsilä GEMS 7、Tesla Autobidder、DNV HERO、enspired 都是 Stochastic MILP 核心
  - 学术基准：RWTH M5Use（Celi Cortés 2024），+196% vs 单 FCR
- 当前 Regime V3（用户自创）已达 62-73% Oracle capture，但没有测过 MILP 对比
- 如果 MILP 不显著更优，**继续 Regime V3 是对的**；不然应切换

### 为什么用 Bootstrap 场景（而不是 VAE）
- 复用用户现有 Regime 分类器 = 信息等价 = 对比洁净（隔离 "MILP vs DP" 信号，排除 "场景质量差异"）
- 后续可升级到 Quantile（Modification #5 已准备）或 VAE（Route F，长期）

### 为什么用 HiGHS 不用 Gurobi
- Gurobi 商业试用 license **2 天审核**，太慢
- HiGHS 开源免费，已在 Pyomo + scipy 里原生支持
- 速度慢 3-5 倍，但 K=200 单 MILP 23s，可接受
- 用户同时在申请 Gurobi 和 COPT

### 为什么不停止实验重跑（solve_day_dual 修复后）
- **正在跑的 MILP 实验不依赖 solve_day_dual**（用的是 Pyomo `build_two_stage_lp`）
- 停止损失 14+h 算力 + 丢 K 现象数据
- 修复只影响 `scripts/24_dual_settlement_backtest.py`（Oracle 对比），重跑那一个只要 10 秒

---

## 🐛 已知问题

1. **Mac #2 比 Mac #1 慢 4 倍**（46s/day vs 11s/day）
   - 可能：后台进程、温度节流、内存压力
   - 影响：Mac #2 跑完要 +30h
   - 调查：`ssh jiang@192.168.0.21 'top -l 1 | head -20'`

2. **sklearn KMeans 报 RuntimeWarning: divide by zero in matmul**
   - 大量 warning 污染日志
   - 不影响结果（KMeans 仍然收敛）
   - TODO：用 `np.errstate(divide="ignore")` 抑制

3. **`.venv` 损坏**（Python 路径指向不存在的目录）
   - 影响：不能用 `source .venv/bin/activate`
   - 绕过：用系统 `python3` + `pip3 install --user`（已装 pyomo, highspy）
   - 要修：`python3 -m venv --clear .venv`

---

## 💡 接手 checklist（明天早上）

```bash
# 1. 先看监控状态
tail -50 /Users/jjj/Desktop/工作/电力交易/energy-storage-rl/runs/monitor.log
cat /Users/jjj/Desktop/工作/电力交易/energy-storage-rl/runs/ALERT.log 2>/dev/null

# 2. 看 Mac #1 是否完成
ps -p 61875 > /dev/null && echo "仍在跑" || echo "完成"
ls /Users/jjj/Desktop/工作/电力交易/energy-storage-rl/runs/milp_experiments/shandong_*.pkl | wc -l

# 3. 看 Mac #2 是否完成
ssh jiang@192.168.0.21 'ps -p 38438 > /dev/null && echo "仍在跑" || echo "完成"; ls ~/energy-storage-rl/runs/milp_experiments/*.pkl | wc -l'

# 4. 如果 Mac #1 完成，跑 K-mystery 调查
cd /Users/jjj/Desktop/工作/电力交易/energy-storage-rl
PYTHONPATH=. python3 scripts/30_k_mystery_investigation.py 2>&1 | tee runs/k_mystery/run.log

# 5. 当两台都完成，rsync 回来：
rsync -avz jiang@192.168.0.21:~/energy-storage-rl/runs/milp_experiments/ runs/milp_experiments/

# 6. 下一步：写分析脚本 scripts/29_milp_analyze.py（P1 任务）
```

---

## 用户偏好（从对话历史提取）

- 用户是门外汉出身但学习极快（一天从"什么是电"问到"Route C MILP"）
- 偏好**中文 + 技术术语不避讳**
- 喜欢**直接的数字和对比**（vs Regime V3、vs LightGBM、vs Oracle）
- 愿意投入时间做**长期正确**的技术决策，不仅仅"能用就行"
- **时间成本敏感**（不怕跑久，但不浪费精力）
- 有**两台 Mac M4 + Windows（未用）**
- 即将融资，需要 pitch 材料里有"最差/平均/最好"情景的**可信数字**

---

## 🛑 2026-04-21 下午勘误 / 作废声明（新接手 session 追加）

**作废**：本文档"📊 已知初步结果" 段（第 63-92 行，含"🚨 重大反直觉发现"）**读错了自己的数据**，结论不成立。

### 1. K-mystery 并不存在

原文档写的 "baseline ¥2,834 万" 其实是 **`shandong_deg_50`**（降解成本调到 ¥50，电池不愿循环）的数字，不是真正的 baseline。真正的 `shandong_baseline`（K=200, dev=10%, deg=¥2）来自 `runs/milp_experiments/summary.csv`：

| 配置 | 原文档"数字" | summary.csv 实际 | 单调性 |
|---|---|---|---|
| K_050 (dev=10%, deg=¥2) | ¥3,903 万 | ¥3,903 万 | ↑ |
| K_100 | ¥4,054 万 | ¥4,054 万 | ↑ |
| **baseline (K=200)** | ❌ ¥2,834 万 | ✅ **¥4,180 万** | ↑ |
| deg_50 (K=200, deg=¥50) | — | ¥2,833 万 | — |

K 随样本量单调递增（Monte Carlo 正常收敛）。`scripts/30_k_mystery_investigation.py` **不要跑，白烧 CPU**。

### 2. 真正该关注的结论：MILP 全线输 Regime V3（山东）

| 方法 | 山东年收入 | Capture vs Oracle |
|---|---|---|
| **Regime V3（生产）** | **¥5,381 万** | **64.8%** |
| MILP dev_00（最好配置） | ¥4,664 万 | 56.1%  |
| MILP baseline (dev=10%) | ¥4,180 万 | 50.3%  |
| MILP dev_20 | ¥2,778 万 | 33.4%  |
| MILP dev_50 | ¥−1,600 万 | 负值 |

**Route C v0.1 在山东未达预期。继续堆 AGC/TVF/报价曲线前，必须先定位根因。**

### 3. 实验已于 2026-04-21 11:45 停止

- Mac #1 PID 61875/61877/63726 已 kill（K_500 意义不大，单调性已确认）
- Mac #2 PID 38438 已 kill（Mac#2 慢 4 倍根因：用 miniforge3 Python + OpenBLAS，而 Mac#1 用系统 Python + Apple Accelerate；xcode CLT 未装）
- Crontab `monitor_experiments.sh` 已移除
- 数据归档：`runs/milp_experiments_v0.1/`（30 个文件）

### 4. 当前诊断计划（替代原 P1/P2）

**主嫌疑**（按概率排）：
- H1 场景权重均匀 1/K 丢失 regime 信号
- H2 Bootstrap 用单日 RT（带噪声），Regime V3 用 regime profile（类内均值，信噪比高）
- H3 Two-stage 结构让 DAM 过度保守（dev_00 最优证实此点）
- H4 Simulation 结算用 actual×dam_price，MILP objective 用 p_dam×dam_forecast（语义错位）

**Jensen 直觉**：`max_a f(a, price)` 对 price 凸 → `E[max f] ≥ max f(E)`。MILP 理论应 ≥ Regime V3，现在 ≤，说明实现有偏差。

**诊断实验序列**：
- **D1**（首要）：场景生成器消融，4 个变体
  - A1 = 现状复刻
  - **A2 = 12 regime profiles + regime_probs 权重**（"Regime V3 同源"）
  - A3 = bootstrap + regime_probs 权重
  - A4 = 同 regime 多日均值 + regime_probs
- D2：单阶段确定性 LP 对照（验证 H3）
- D3：结算语义审计（10 典型日 log）
- D4：跨省重跑（本地 Mac#1，~10h）

**阅读队列（边跑实验边读）**：
- Conejo, Carrion, Morales 2010《Decision Making Under Uncertainty in Electricity Markets》Ch.6-7
- Morales et al. 2014 IEEE TPS "Bidding with stepwise offering curves"
- Liu, Li, Zhao 2025 arXiv:2511.15629 (GPU-DP, ISO-NE 99% capture) — Thomas Lee 新作

---

**完。问题联系 JJJ（邮箱见团队联系方式文档）。**
