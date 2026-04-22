"""DuckDB → parquet/xlsx 导出，对接现有ingest.py"""

from __future__ import annotations
from pathlib import Path
from loguru import logger

from crawlers.storage.db import db
from crawlers.config.provinces import PROVINCES
from crawlers.config.settings import settings


def export_all_parquet(output_dir: Path = None):
    """导出所有省份数据为parquet，格式与现有pipeline兼容"""
    output_dir = output_dir or settings.project_root / "data" / "china" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    db.export_all_parquet(output_dir)
    logger.info("All provinces exported to {}", output_dir)


def export_province_parquet(province_name: str, output_dir: Path = None):
    output_dir = output_dir or settings.project_root / "data" / "china" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = PROVINCES.get(province_name)
    if not spec:
        raise ValueError(f"Unknown province: {province_name}")
    out_path = output_dir / f"{province_name}_crawled.parquet"
    db.export_parquet(spec.name_cn, out_path)
    return out_path
