"""
Fetch historical weather data from Open-Meteo API (free, no key needed).

Downloads hourly: temperature_2m, wind_speed_10m, shortwave_radiation, direct_radiation
Then resamples to 15-min and merges into province parquet files.

Usage:
    PYTHONPATH=. python3 scripts/13_fetch_weather.py
"""
from __future__ import annotations

import time
import json
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

PROCESSED_DIR = Path("data/china/processed")

# Province coordinates (representative city for load-weighted weather)
PROVINCES = {
    "shandong": {"lat": 36.65, "lon": 116.99, "name": "济南"},
    "shanxi":   {"lat": 37.87, "lon": 112.55, "name": "太原"},
    "guangdong": {"lat": 23.13, "lon": 113.26, "name": "广州"},
    "gansu":    {"lat": 36.06, "lon": 103.83, "name": "兰州"},
}

WEATHER_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_radiation",
]


def fetch_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly weather from Open-Meteo archive API."""
    vars_str = ",".join(WEATHER_VARS)
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        f"&hourly={vars_str}"
        f"&timezone=Asia/Shanghai"
    )
    logger.info(f"  Fetching: {url[:120]}...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())

    if "hourly" not in data:
        raise ValueError(f"API error: {data.get('reason', data)}")

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        **{var: hourly[var] for var in WEATHER_VARS}
    })
    df = df.set_index("timestamp")
    logger.info(f"  Got {len(df)} hourly rows: {df.index[0]} → {df.index[-1]}")
    return df


def resample_15min(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Resample hourly weather to 15-min by interpolation."""
    # Create 15-min index
    idx_15m = pd.date_range(df_hourly.index[0], df_hourly.index[-1], freq="15min")
    df_15m = df_hourly.reindex(idx_15m)
    # Interpolate (linear for temperature/wind, forward-fill for radiation)
    df_15m["temperature_2m"] = df_15m["temperature_2m"].interpolate(method="linear")
    df_15m["wind_speed_10m"] = df_15m["wind_speed_10m"].interpolate(method="linear")
    df_15m["shortwave_radiation"] = df_15m["shortwave_radiation"].interpolate(method="linear").clip(lower=0)
    df_15m["direct_radiation"] = df_15m["direct_radiation"].interpolate(method="linear").clip(lower=0)
    return df_15m


def merge_weather_to_oracle(province: str, df_weather_15m: pd.DataFrame):
    """Merge weather into the oracle parquet file."""
    oracle_path = PROCESSED_DIR / f"{province}_oracle.parquet"
    df = pd.read_parquet(oracle_path)
    logger.info(f"  Oracle: {len(df)} rows, {df.index[0]} → {df.index[-1]}")

    # Drop existing weather columns if any
    for col in WEATHER_VARS + ["temperature"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Merge on index (timestamp)
    df = df.join(df_weather_15m, how="left")

    # Fill NaN at edges
    for var in WEATHER_VARS:
        if var in df.columns:
            nan_pct = df[var].isna().mean() * 100
            if nan_pct > 0:
                df[var] = df[var].ffill().bfill().fillna(0)
                logger.info(f"    {var}: filled {nan_pct:.1f}% NaN")

    # Add temperature alias for features.py compatibility
    if "temperature_2m" in df.columns:
        df["temperature"] = df["temperature_2m"]

    # Save
    df.to_parquet(oracle_path)
    logger.info(f"  Saved: {oracle_path} ({len(df)} rows, {len(df.columns)} cols)")

    # Verify
    for var in WEATHER_VARS:
        if var in df.columns:
            nz = (df[var] != 0).sum() / len(df) * 100
            logger.info(f"    {var}: non-zero={nz:.0f}%, range=[{df[var].min():.1f}, {df[var].max():.1f}]")


def main():
    for province, info in PROVINCES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  {province.upper()} ({info['name']}) — lat={info['lat']}, lon={info['lon']}")
        logger.info(f"{'='*60}")

        # Determine date range from oracle file
        oracle_path = PROCESSED_DIR / f"{province}_oracle.parquet"
        if not oracle_path.exists():
            logger.warning(f"  Skipping {province}: oracle file not found")
            continue

        df_oracle = pd.read_parquet(oracle_path, columns=["rt_price"])
        start = df_oracle.index[0].strftime("%Y-%m-%d")
        end = df_oracle.index[-1].strftime("%Y-%m-%d")
        logger.info(f"  Date range: {start} → {end}")

        # Open-Meteo limits to ~2 years per request, so chunk if needed
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        chunks = []
        chunk_start = start_dt
        while chunk_start < end_dt:
            chunk_end = min(chunk_start + pd.DateOffset(years=2) - pd.DateOffset(days=1), end_dt)
            try:
                df_chunk = fetch_weather(
                    info["lat"], info["lon"],
                    chunk_start.strftime("%Y-%m-%d"),
                    chunk_end.strftime("%Y-%m-%d"),
                )
                chunks.append(df_chunk)
                logger.info(f"  Chunk OK: {chunk_start.date()} → {chunk_end.date()}, {len(df_chunk)} rows")
            except Exception as e:
                logger.error(f"  Fetch error: {e}")
            chunk_start = chunk_end + pd.DateOffset(days=1)
            time.sleep(1)  # be nice to the API

        if not chunks:
            logger.error(f"  No weather data for {province}")
            continue

        df_weather = pd.concat(chunks)
        df_weather = df_weather[~df_weather.index.duplicated(keep="first")]
        logger.info(f"  Total weather: {len(df_weather)} hourly rows")

        # Resample to 15-min
        df_15m = resample_15min(df_weather)
        logger.info(f"  Resampled: {len(df_15m)} 15-min rows")

        # Merge into oracle
        merge_weather_to_oracle(province, df_15m)

        # Save raw weather as backup
        weather_path = PROCESSED_DIR / f"{province}_weather.parquet"
        df_weather.to_parquet(weather_path)
        logger.info(f"  Raw weather saved: {weather_path}")


if __name__ == "__main__":
    main()
