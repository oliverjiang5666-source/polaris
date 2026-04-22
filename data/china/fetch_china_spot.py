"""
中国电力现货市场数据采集工具

数据源优先级：
  1. 电力交易中心公开API（山东/广东，需注册）
  2. 电查查 dianchacha.cn（商业平台，部分免费）
  3. 北极星电力新闻网 bjx.com.cn（月度报告爬取）
  4. 泛能网 etpage.fanneng.com（月度汇总）

市场结构（与ERCOT对比）：
  ERCOT:  RTM 15min + DAM 1h, 单一结算
  山东:   日前(1h) + 实时(15min), 双结算, 报量报价
  广东:   日前(1h) + 实时(15min), 节点电价, 南方区域市场

目标数据字段：
  - timestamp: 时间戳（15min或1h粒度）
  - da_price:  日前出清价格 (元/MWh)
  - rt_price:  实时出清价格 (元/MWh)
  - province:  省份
  - node:      节点（山东统一结算点/广东节点电价）
"""

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent
SD_DIR = DATA_DIR / "shandong"
GD_DIR = DATA_DIR / "guangdong"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# ============================================================
# 数据源 1: 山东电力交易中心 API
# 网址: https://pmos.sd.sgcc.com.cn
# 需要注册市场主体账号，获取API token
# ============================================================

class ShandongSpotAPI:
    """
    山东电力交易平台 API

    注册流程：
    1. 访问 https://pmos.sd.sgcc.com.cn/pmos/index/login.jsp
    2. 注册市场主体账号（需企业资质）
    3. 获取 API token
    4. 设置环境变量 SD_POWER_TOKEN

    备选：通过泛能网数据接口（已对接山东交易中心）
    """

    BASE_URL = "https://pmos.sd.sgcc.com.cn/pmos"

    def __init__(self):
        self.token = os.getenv("SD_POWER_TOKEN", "")
        if not self.token:
            logger.warning(
                "SD_POWER_TOKEN not set. "
                "Register at https://pmos.sd.sgcc.com.cn or use alternative sources."
            )

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def fetch_da_prices(self, date: str) -> pd.DataFrame:
        """获取日前出清价格 (需要API token)"""
        if not self.token:
            raise ValueError("SD_POWER_TOKEN required. See class docstring for setup.")
        # API endpoint待确认（需登录后抓包获取实际endpoint）
        # 预留接口结构
        raise NotImplementedError(
            "Actual API endpoint needs to be captured from the SPA. "
            "Use browser DevTools Network tab after login to find the real endpoint."
        )


# ============================================================
# 数据源 2: 广东电力交易中心 API
# 网址: https://pm.gd.csg.cn
# 南方电网体系，与国网体系不同
# ============================================================

class GuangdongSpotAPI:
    """
    广东电力交易中心 API（南方电网）

    注册流程：
    1. 访问 https://pm.gd.csg.cn/views/index.html
    2. 注册市场主体账号
    3. 获取 API token
    4. 设置环境变量 GD_POWER_TOKEN
    """

    BASE_URL = "https://pm.gd.csg.cn"

    def __init__(self):
        self.token = os.getenv("GD_POWER_TOKEN", "")
        if not self.token:
            logger.warning(
                "GD_POWER_TOKEN not set. "
                "Register at https://pm.gd.csg.cn or use alternative sources."
            )

    def fetch_da_prices(self, date: str) -> pd.DataFrame:
        """获取日前出清价格"""
        if not self.token:
            raise ValueError("GD_POWER_TOKEN required.")
        raise NotImplementedError("Capture actual endpoint from browser DevTools.")


# ============================================================
# 数据源 3: 北极星电力新闻网 月度报告数据
# 最可靠的公开数据源 - 从月度运行报告中提取
# ============================================================

class BJXReportScraper:
    """
    从北极星电力新闻网(bjx.com.cn)爬取月度运行报告

    报告通常包含：
    - 月度分时段平均电价
    - 日前/实时出清均价
    - 峰谷价差统计
    - 负电价时段统计
    """

    SEARCH_URL = "https://news.bjx.com.cn/search.asp"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
    }

    def search_reports(self, province: str, year: int) -> list[dict]:
        """搜索月度运行报告"""
        keywords = f"{province}电力市场 月度运行 {year}"
        logger.info(f"Searching BJX for: {keywords}")

        try:
            r = requests.get(
                self.SEARCH_URL,
                params={"q": keywords},
                headers=self.HEADERS,
                timeout=30,
            )
            r.encoding = "utf-8"
            # BJX search results parsing
            # 实际需要用BeautifulSoup解析搜索结果页
            logger.info(f"BJX search returned status {r.status_code}")
            return []  # placeholder
        except Exception as e:
            logger.error(f"BJX search failed: {e}")
            return []


# ============================================================
# 数据源 4: 已知公开数据汇编
# 从各种公开报告中整理的历史数据
# ============================================================

# 山东2023-2024年核心参数（来源：北极星、CNESA、能源局报告）
SHANDONG_MONTHLY_STATS = {
    # (year, month): {da_avg, rt_avg, peak_avg, valley_avg, spread}
    # 单位: 元/MWh
    # 来源: 山东2024年市场总体运行情况报告（北极星电力新闻网）
    2024: {
        "da_avg_annual": 316.21,   # 日前出清均价
        "rt_avg_annual": 309.52,   # 实时出清均价
        "da_volume_gwh": 3334.97,  # 日前出清电量(亿kWh -> GWh)
        "rt_volume_gwh": 4201.48,  # 用电侧日前出清电量
        "settlement_avg": 453.51,  # 发电侧总结算均价
    },
    2023: {
        "settlement_avg": 468.16,  # 发电侧总结算均价
        "volume_gwh": 3255.45,     # 上网电量
    },
}

# 广东已知参数
GUANGDONG_MONTHLY_STATS = {
    # 来源: 广东电力市场半年报
    2024: {
        "storage_midterm_volume_gwh": 210,  # 储能中长期交易电量
        "note": "现货平均价差约0.6元/kWh（来自行业报告）",
    },
}


# ============================================================
# 合成数据生成器（基于真实统计参数）
# 在获取真实逐时数据前，用于初步模型验证
# ============================================================

def generate_synthetic_shandong(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    基于山东市场真实统计参数生成合成现货价格数据

    参数来源：
    - 年均日前价: 316.21 元/MWh (2024)
    - 年均实时价: 309.52 元/MWh (2024)
    - 峰谷价差: ~600 元/MWh（极端），常态200-400
    - 负电价: 存在，2023年曾出现连续21小时负电价

    价格模式特征（山东）：
    - 深谷: 02:00-06:00（新能源出力大+用电低谷）
    - 早峰: 08:00-11:00
    - 午谷: 12:00-14:00（光伏出力高峰）
    - 晚峰: 17:00-21:00（光伏退出+用电高峰）
    - 季节性: 夏季(7-8月)价差最大，春秋价差较小
    """
    import numpy as np

    np.random.seed(42)

    idx = pd.date_range(start=start_date, end=end_date, freq=freq, tz="Asia/Shanghai")
    n = len(idx)

    hours = idx.hour
    months = idx.month
    weekdays = idx.weekday  # 0=Mon, 6=Sun

    # --- 日内形状（山东特征：双谷双峰）---
    hourly_shape = np.array([
        #  0     1     2     3     4     5     6     7
        -0.3, -0.4, -0.5, -0.5, -0.4, -0.3, -0.1,  0.1,
        #  8     9    10    11    12    13    14    15
         0.4,  0.5,  0.4,  0.3,  0.0, -0.1,  0.0,  0.1,
        # 16    17    18    19    20    21    22    23
         0.3,  0.6,  0.7,  0.6,  0.4,  0.2,  0.0, -0.2,
    ])

    # --- 季节性调整 ---
    seasonal_factor = np.array([
        #  1月   2月   3月   4月   5月   6月
         0.9,  0.8,  0.85, 0.9,  0.95, 1.1,
        #  7月   8月   9月  10月  11月  12月
         1.2,  1.2,  1.0,  0.9,  0.85, 0.9,
    ])

    # --- 基础价格 ---
    base_da = 316.21  # 元/MWh (2024年均)
    base_rt = 309.52

    # 对2023年调高基准
    year_adj = np.where(idx.year == 2023, 468.16 / 453.51, 1.0)

    # 构建日前价格
    da_prices = np.zeros(n)
    for i in range(n):
        h = hours[i]
        m = months[i] - 1

        shape = hourly_shape[h]
        season = seasonal_factor[m]
        weekend = 0.85 if weekdays[i] >= 5 else 1.0

        # 价格 = 基准 * (1 + 日内形状*振幅) * 季节 * 周末 * 年度调整
        amplitude = 0.6  # 振幅系数
        noise = np.random.normal(0, 0.08)

        da_prices[i] = (
            base_da
            * (1 + shape * amplitude + noise)
            * season
            * weekend
            * year_adj[i]
        )

    # 实时价格 = 日前 + 随机偏差（更大波动）
    rt_noise = np.random.normal(0, 30, n)  # 实时波动更大
    rt_prices = da_prices + rt_noise

    # 注入负电价事件（山东特色）
    # 约1-2%的时段出现负电价（主要在凌晨+春秋光伏大发季节）
    neg_mask = (
        (hours >= 2) & (hours <= 5)
        & (months >= 3) & (months <= 5)
        & (np.random.random(n) < 0.15)
    )
    rt_prices[neg_mask] = np.random.uniform(-100, -10, neg_mask.sum())
    da_prices[neg_mask] = np.random.uniform(-50, 0, neg_mask.sum())

    # 注入极端高价事件（夏季尖峰）
    spike_mask = (
        (hours >= 17) & (hours <= 20)
        & (months >= 7) & (months <= 8)
        & (np.random.random(n) < 0.05)
    )
    rt_prices[spike_mask] = np.random.uniform(600, 1000, spike_mask.sum())

    # clip到合理范围
    da_prices = np.clip(da_prices, -200, 1500)
    rt_prices = np.clip(rt_prices, -500, 1500)

    df = pd.DataFrame({
        "timestamp": idx,
        "da_price": np.round(da_prices, 2),
        "rt_price": np.round(rt_prices, 2),
        "province": "shandong",
        "node": "SD_UNIFIED",  # 山东统一结算点
    })

    logger.info(
        f"Generated Shandong synthetic data: {len(df)} rows, "
        f"DA mean={df.da_price.mean():.1f}, RT mean={df.rt_price.mean():.1f}, "
        f"Spread(P90-P10)={df.rt_price.quantile(0.9) - df.rt_price.quantile(0.1):.1f}"
    )

    return df


def generate_synthetic_guangdong(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    freq: str = "1h",
) -> pd.DataFrame:
    """
    基于广东市场真实统计参数生成合成现货价格数据

    广东 vs 山东差异：
    - 广东用电基数更大，工业负荷占比高
    - 广东是节点电价（非统一结算点）
    - 广东与南方区域市场联动
    - 广东夏季空调负荷极大，夏季价差比山东更大
    - 光伏占比低于山东，午谷效应弱于山东
    """
    import numpy as np

    np.random.seed(123)

    idx = pd.date_range(start=start_date, end=end_date, freq=freq, tz="Asia/Shanghai")
    n = len(idx)

    hours = idx.hour
    months = idx.month
    weekdays = idx.weekday

    # 广东日内形状：午谷不如山东深（光伏占比低）
    hourly_shape = np.array([
        -0.3, -0.4, -0.5, -0.5, -0.4, -0.3, -0.1,  0.1,
         0.3,  0.5,  0.5,  0.4,  0.2,  0.1,  0.2,  0.3,
         0.4,  0.5,  0.6,  0.5,  0.4,  0.2,  0.0, -0.2,
    ])

    seasonal_factor = np.array([
        0.85, 0.8,  0.9,  1.0,  1.05, 1.15,
        1.3,  1.3,  1.1,  1.0,  0.9,  0.85,
    ])

    base_price = 350.0  # 广东电价整体高于山东

    da_prices = np.zeros(n)
    for i in range(n):
        h = hours[i]
        m = months[i] - 1

        shape = hourly_shape[h]
        season = seasonal_factor[m]
        weekend = 0.82 if weekdays[i] >= 5 else 1.0
        noise = np.random.normal(0, 0.07)

        da_prices[i] = base_price * (1 + shape * 0.55 + noise) * season * weekend

    rt_noise = np.random.normal(0, 25, n)
    rt_prices = da_prices + rt_noise

    # 广东负电价较少
    neg_mask = (
        (hours >= 3) & (hours <= 5)
        & (months >= 3) & (months <= 4)
        & (np.random.random(n) < 0.03)
    )
    rt_prices[neg_mask] = np.random.uniform(-50, -5, neg_mask.sum())

    # 夏季尖峰更猛
    spike_mask = (
        (hours >= 14) & (hours <= 19)
        & (months >= 6) & (months <= 9)
        & (np.random.random(n) < 0.04)
    )
    rt_prices[spike_mask] = np.random.uniform(700, 1200, spike_mask.sum())

    da_prices = np.clip(da_prices, -200, 1500)
    rt_prices = np.clip(rt_prices, -500, 1500)

    df = pd.DataFrame({
        "timestamp": idx,
        "da_price": np.round(da_prices, 2),
        "rt_price": np.round(rt_prices, 2),
        "province": "guangdong",
        "node": "GD_UNIFIED",  # 简化处理，实际是节点电价
    })

    logger.info(
        f"Generated Guangdong synthetic data: {len(df)} rows, "
        f"DA mean={df.da_price.mean():.1f}, RT mean={df.rt_price.mean():.1f}, "
        f"Spread(P90-P10)={df.rt_price.quantile(0.9) - df.rt_price.quantile(0.1):.1f}"
    )

    return df


# ============================================================
# 数据导出
# ============================================================

def save_all(overwrite: bool = False):
    """生成并保存所有合成数据"""

    for province, gen_fn, save_dir in [
        ("shandong", generate_synthetic_shandong, SD_DIR),
        ("guangdong", generate_synthetic_guangdong, GD_DIR),
    ]:
        out_path = save_dir / f"{province}_spot_synthetic_2023_2024.parquet"
        csv_path = save_dir / f"{province}_spot_synthetic_2023_2024.csv"

        if out_path.exists() and not overwrite:
            logger.info(f"Skipping {province}: {out_path} exists (use overwrite=True)")
            continue

        df = gen_fn()
        df.to_parquet(out_path, index=False)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {province}: {out_path} ({len(df)} rows)")

    # 合并输出
    sd = pd.read_parquet(SD_DIR / "shandong_spot_synthetic_2023_2024.parquet")
    gd = pd.read_parquet(GD_DIR / "guangdong_spot_synthetic_2023_2024.parquet")
    combined = pd.concat([sd, gd], ignore_index=True)
    combined_path = PROCESSED_DIR / "china_spot_combined_2023_2024.parquet"
    combined.to_parquet(combined_path, index=False)
    logger.info(f"Combined: {combined_path} ({len(combined)} rows)")


def print_summary():
    """打印数据概要"""
    for province in ["shandong", "guangdong"]:
        p_dir = DATA_DIR / province
        files = list(p_dir.glob("*.parquet"))
        if not files:
            print(f"\n{province}: No data files found")
            continue

        for f in files:
            df = pd.read_parquet(f)
            print(f"\n{'='*60}")
            print(f"{province.upper()} - {f.name}")
            print(f"{'='*60}")
            print(f"Rows:        {len(df):,}")
            print(f"Date range:  {df.timestamp.min()} → {df.timestamp.max()}")
            print(f"DA price:    mean={df.da_price.mean():.1f}, "
                  f"min={df.da_price.min():.1f}, max={df.da_price.max():.1f}")
            print(f"RT price:    mean={df.rt_price.mean():.1f}, "
                  f"min={df.rt_price.min():.1f}, max={df.rt_price.max():.1f}")

            # 峰谷价差
            spread = df.rt_price.quantile(0.9) - df.rt_price.quantile(0.1)
            print(f"Spread(P90-P10): {spread:.1f} 元/MWh")

            # 负电价比例
            neg_pct = (df.rt_price < 0).mean() * 100
            print(f"Negative RT:     {neg_pct:.1f}%")


if __name__ == "__main__":
    logger.info("Generating synthetic China spot market data...")
    save_all(overwrite=True)
    print_summary()
