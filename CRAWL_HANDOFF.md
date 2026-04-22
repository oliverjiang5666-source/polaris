# 电力数据爬虫交接文档

## 目标
自建数据源，不依赖任何第三方。爬取中国5个省的电力现货市场15分钟出清电价+辅助服务数据，加上免费天气数据。

## 需要爬的目标

### 1. 现货出清电价（核心，每天爬）

| 省份 | 网址 | 电网 | 现货状态 | 出清周期 | 历史深度 |
|------|------|------|---------|---------|---------|
| 山西 | https://pmos.sx.sgcc.com.cn | 国网 | 正式运行2023底 | **5分钟**（288点/天） | 最长，2021.4起 |
| 山东 | https://pmos.sd.sgcc.com.cn | 国网 | 正式运行2024.6 | 15分钟（96点/天） | 2024.6起 |
| 广东 | https://pm.gd.csg.cn | 南网 | 正式运行2023底 | 15分钟 | 2023底起 |
| 甘肃 | https://pmos.gs.sgcc.com.cn | 国网 | 正式运行 | 15分钟 | 待探测 |
| 蒙西 | 内蒙古电力交易中心（网址待确认） | 蒙西电网 | 正式运行2025.2 | 15分钟 | 2025.2起 |

**爬什么：**
- 日前市场出清电价（每日发布，次日执行）
- 实时市场出清电价（每15分钟/5分钟更新）
- 每条数据：时间戳、时段、出清价格（元/MWh）、节点/区域

### 2. 辅助服务数据（每周/月爬）

各省交易中心在"市场信息披露"或"交易公告"板块会发布：
- 调频服务出清价格和结算报告（周度/月度）
- 备用服务价格
- 辅助服务费用分摊情况

这些通常是PDF或Excel附件，需要下载+解析。

重点参考值：
- 山西二次调频：报价上限15元/MW，性能系数最高2
- 山东调频：报量报价方式

### 3. 天气数据（免费，已有方案）

**Open-Meteo API，完全免费，不需要爬虫：**
```
# 示例：拿山西太原的逐小时天气
https://api.open-meteo.com/v1/archive?latitude=37.87&longitude=112.55&start_date=2021-01-01&end_date=2025-12-31&hourly=temperature_2m,wind_speed_10m,shortwave_radiation,direct_radiation

# 各省坐标
山西太原: 37.87, 112.55
山东济南: 36.65, 116.99
广东广州: 23.13, 113.26
甘肃兰州: 36.06, 103.83
内蒙呼和浩特: 40.84, 111.75
广西南宁: 22.82, 108.32（中广核项目）
```

变量说明：
- temperature_2m: 气温（影响空调负荷→影响电价）
- wind_speed_10m: 风速（影响风电出力→影响电价）
- shortwave_radiation: 总辐射（影响光伏出力→影响电价）
- direct_radiation: 直接辐射（更精准的光伏估算）

### 4. 负荷和新能源出力

各省交易中心有时会在市场报告中附带发布，但不稳定。优先从以下途径：
- 交易中心的"系统运行信息"板块
- 月度市场报告PDF中的图表（需要OCR或手动提取）
- 部分省份的电力调度网站

## 爬虫技术要求

### 探测阶段（优先做这个）
开VPN访问上述5个网站，搞清楚：
1. 数据在哪个页面/板块
2. 是直接HTML表格、还是JS动态加载、还是文件下载（PDF/Excel/CSV）
3. 需不需要登录/注册
4. 有没有反爬（验证码、频率限制）
5. 历史数据能追溯到什么时候

### 爬虫实现
```
data/
├── crawlers/
│   ├── base.py          # 基类：重试、频率限制、存储接口
│   ├── shanxi.py        # 山西（国网体系，可能和山东/甘肃类似）
│   ├── shandong.py      # 山东
│   ├── guangdong.py     # 广东（南网体系，不同于国网）
│   ├── gansu.py         # 甘肃
│   └── mengxi.py        # 蒙西
├── parsers/
│   ├── pdf_parser.py    # 解析辅助服务PDF报告
│   └── excel_parser.py  # 解析Excel附件
└── scheduler.py         # 定时任务：每天爬电价，每周爬辅助服务
```

### 存储格式
统一存为parquet，一个省一个文件：
```
data/raw/
├── spot_price_shanxi.parquet    # 列：timestamp, period, price_yuan_mwh, market(dam/rtm)
├── spot_price_shandong.parquet
├── spot_price_guangdong.parquet
├── ancillary_shanxi.parquet     # 列：timestamp, service_type, price, settlement
├── weather_shanxi.parquet       # 列：timestamp, temperature, wind_speed, radiation
└── ...
```

### 注意事项
- 国网系统的网站（山西/山东/甘肃）大概率结构相似，一套爬虫改参数就行
- 南网（广东）的网站结构不同，需要单独写
- **需要开VPN访问**——这些网站可能从新加坡/海外IP也能访问（不确定是否像ERCOT那样封海外），需要测试
- 爬取频率：电价每天爬一次就行（日前价格一天只出一次），不需要实时爬
- 要做好容错：网站改版、临时维护、数据格式变化都要能自动跳过+告警

## 已有的可复用代码

- `data/storage.py` — parquet存储，追加去重模式
- `data/fetch_ercot.py` — ERCOT数据拉取（可参考结构）
- funding-rate-rl项目的 `data/collectors/base.py` — 异步HTTP采集基类

## 优先级

```
P0: 山西（历史最长，5分钟粒度，最有价值）
P0: 广东（南网，中广核项目区域）
P1: 山东（第二大独立储能市场）
P2: 甘肃、蒙西（有数据就爬，没有就先不管）
```

## 第一步
开VPN后，用浏览器打开这5个网站，截图给Claude看，让它分析数据在哪、怎么抓。
