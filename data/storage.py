"""
Parquet存储 — 复用funding-rate-rl的模式

追加写入+去重，支持增量更新。
"""
from pathlib import Path
import pandas as pd
from loguru import logger

RAW_DIR = Path(__file__).parent / "raw"


def save(df: pd.DataFrame, name: str) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"

    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df], ignore_index=True)
        if "timestamp" in df.columns:
            df = df.drop_duplicates(subset=["timestamp"], keep="last")
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = df.drop_duplicates(keep="last").reset_index(drop=True)

    df.to_parquet(path, index=False)
    logger.info(f"Saved {path.name}: {len(df):,} rows")
    return len(df)


def load(name: str) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()
