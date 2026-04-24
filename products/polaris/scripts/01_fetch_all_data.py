"""
拉取ERCOT全量训练数据（不需要VPN）

需要先注册EIA API key（免费，30秒）：
  https://www.eia.gov/opendata/register.php
  拿到key后替换下面的 EIA_API_KEY

用法：
  cd ~/Desktop/energy-storage-rl
  .venv/bin/python scripts/01_fetch_all_data.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import pandas as pd
from loguru import logger

RAW = Path(__file__).parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# ==========================================
# 替换成你自己的key（免费注册）
# https://www.eia.gov/opendata/register.php
EIA_API_KEY = "DEMO_KEY"  # ← 换成你的
# ==========================================


def fetch_eia(endpoint: str, facets: dict, col_name: str) -> pd.DataFrame:
    """分页拉取EIA数据"""
    all_rows = []
    offset = 0
    base = f"https://api.eia.gov/v2/electricity/rto/{endpoint}/data/"

    while True:
        params = {
            "api_key": EIA_API_KEY,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "start": "2021-01-01",
            "end": "2026-02-01",
            "length": 5000,
            "offset": offset,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
        }
        params.update(facets)

        for attempt in range(3):
            try:
                r = requests.get(base, params=params, timeout=30)
                if r.status_code == 429:
                    logger.warning(f"  Rate limited, waiting 5s...")
                    time.sleep(5)
                    continue
                if r.status_code != 200:
                    logger.error(f"  {col_name} error {r.status_code} at offset {offset}")
                    return pd.DataFrame()
                break
            except Exception as e:
                logger.warning(f"  Retry {attempt+1}: {e}")
                time.sleep(3)
        else:
            break

        rows = r.json().get("response", {}).get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        logger.info(f"  {col_name}: {len(all_rows):,} rows...")
        if len(rows) < 5000:
            break
        offset += 5000
        time.sleep(1)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["period"])
    df[col_name] = pd.to_numeric(df["value"], errors="coerce")
    return df[["timestamp", col_name]].sort_values("timestamp").reset_index(drop=True)


def fetch_weather():
    """Texas天气（Open-Meteo，完全免费）"""
    logger.info("=== 天气数据（Open-Meteo）===")
    path = RAW / "weather_texas.parquet"
    if path.exists():
        logger.info(f"  已存在，跳过")
        return

    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": 31.97, "longitude": -99.90,
        "start_date": "2021-01-01", "end_date": "2026-01-31",
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,shortwave_radiation,direct_radiation",
        "timezone": "America/Chicago",
    }, timeout=60)

    if r.status_code == 200:
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(h["time"]),
            "temperature_c": h["temperature_2m"],
            "wind_speed_10m": h["wind_speed_10m"],
            "wind_speed_100m": h["wind_speed_100m"],
            "solar_radiation": h["shortwave_radiation"],
            "direct_radiation": h["direct_radiation"],
        })
        df.to_parquet(path, index=False)
        logger.info(f"  Saved: {len(df):,} rows")
    else:
        logger.error(f"  Failed: {r.status_code}")


def main():
    print("\n🔌 ERCOT 全量训练数据拉取\n")

    # 1. 天气（不需要EIA key）
    fetch_weather()

    if EIA_API_KEY == "DEMO_KEY":
        print("\n⚠️  你还在用DEMO_KEY，会被速率限制！")
        print("   去 https://www.eia.gov/opendata/register.php 免费注册")
        print("   拿到key后替换脚本里的 EIA_API_KEY\n")

    # 2. 负荷
    logger.info("=== ERCOT 负荷（Demand）===")
    demand = fetch_eia("region-data", {"facets[type][]": "D"}, "demand_mwh")
    if not demand.empty:
        demand.to_parquet(RAW / "ercot_demand.parquet", index=False)
        logger.info(f"  Saved: {len(demand):,} rows, {demand['timestamp'].min()} ~ {demand['timestamp'].max()}")

    # 3. 风电
    logger.info("=== ERCOT 风电（Wind）===")
    wind = fetch_eia("fuel-type-data", {"facets[fueltype][]": "WND"}, "wind_mwh")
    if not wind.empty:
        wind.to_parquet(RAW / "ercot_wind.parquet", index=False)
        logger.info(f"  Saved: {len(wind):,} rows, {wind['timestamp'].min()} ~ {wind['timestamp'].max()}")

    # 4. 光伏
    logger.info("=== ERCOT 光伏（Solar）===")
    solar = fetch_eia("fuel-type-data", {"facets[fueltype][]": "SUN"}, "solar_mwh")
    if not solar.empty:
        solar.to_parquet(RAW / "ercot_solar.parquet", index=False)
        logger.info(f"  Saved: {len(solar):,} rows, {solar['timestamp'].min()} ~ {solar['timestamp'].max()}")

    # 5. 总发电
    logger.info("=== ERCOT 总发电（Generation）===")
    gen = fetch_eia("region-data", {"facets[type][]": "NG"}, "generation_mwh")
    if not gen.empty:
        gen.to_parquet(RAW / "ercot_generation.parquet", index=False)
        logger.info(f"  Saved: {len(gen):,} rows, {gen['timestamp'].min()} ~ {gen['timestamp'].max()}")

    # 汇总
    print("\n=== 数据汇总 ===")
    for f in sorted(RAW.glob("*.parquet")):
        df = pd.read_parquet(f)
        print(f"  {f.name:35s} {len(df):>10,} rows")


if __name__ == "__main__":
    main()
