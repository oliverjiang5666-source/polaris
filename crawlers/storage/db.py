"""DuckDB 存储层 — upsert / query / backup"""

from __future__ import annotations
import duckdb
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from loguru import logger

from crawlers.config.settings import settings
from crawlers.sources.base import RawRecord


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS spot_data (
    indicator   VARCHAR NOT NULL,
    province    VARCHAR NOT NULL,
    timestamp   TIMESTAMP NOT NULL,
    value       DOUBLE NOT NULL,
    unit        VARCHAR NOT NULL,
    source      VARCHAR NOT NULL,
    fetched_at  TIMESTAMP NOT NULL,
    PRIMARY KEY (indicator, province, timestamp)
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_province_ts
ON spot_data (province, timestamp);
"""


class CrawlerDB:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or settings.db_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_INDEX)
            logger.info("Connected to DuckDB: {}", self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def upsert(self, records: list[RawRecord]):
        if not records:
            return
        conn = self.connect()
        df = pd.DataFrame([r.to_dict() for r in records])
        # DuckDB INSERT OR REPLACE
        conn.execute("""
            INSERT OR REPLACE INTO spot_data
            SELECT * FROM df
        """)
        logger.info("Upserted {} records", len(records))

    def query_province(
        self, province: str, start: date = None, end: date = None
    ) -> pd.DataFrame:
        conn = self.connect()
        sql = "SELECT * FROM spot_data WHERE province = ?"
        params = [province]
        if start:
            sql += " AND timestamp >= ?"
            params.append(datetime.combine(start, datetime.min.time()))
        if end:
            sql += " AND timestamp <= ?"
            params.append(datetime.combine(end, datetime.max.time()))
        sql += " ORDER BY timestamp, indicator"
        return conn.execute(sql, params).fetchdf()

    def get_last_timestamp(self, province: str) -> datetime | None:
        conn = self.connect()
        result = conn.execute(
            "SELECT MAX(timestamp) FROM spot_data WHERE province = ?",
            [province],
        ).fetchone()
        return result[0] if result and result[0] else None

    def get_stats(self) -> pd.DataFrame:
        conn = self.connect()
        return conn.execute("""
            SELECT
                province,
                COUNT(*) as record_count,
                COUNT(DISTINCT indicator) as indicator_count,
                MIN(timestamp) as earliest,
                MAX(timestamp) as latest,
                MAX(fetched_at) as last_crawled
            FROM spot_data
            GROUP BY province
            ORDER BY province
        """).fetchdf()

    def export_parquet(self, province: str, output_path: Path):
        conn = self.connect()
        conn.execute(
            f"COPY (SELECT * FROM spot_data WHERE province = ? ORDER BY timestamp) "
            f"TO '{output_path}' (FORMAT PARQUET)",
            [province],
        )
        logger.info("Exported {} → {}", province, output_path)

    def export_all_parquet(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        provinces = conn.execute(
            "SELECT DISTINCT province FROM spot_data"
        ).fetchdf()["province"].tolist()
        for prov in provinces:
            out = output_dir / f"{prov}_crawled.parquet"
            self.export_parquet(prov, out)

    def total_records(self) -> int:
        conn = self.connect()
        return conn.execute("SELECT COUNT(*) FROM spot_data").fetchone()[0]


# 全局实例
db = CrawlerDB()
