"""数据源适配层 — 抽象基类 + RawRecord"""

from __future__ import annotations
import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from loguru import logger

from crawlers.config.provinces import ProvinceSpec
from crawlers.config.settings import settings


@dataclass
class RawRecord:
    """统一中间格式，与现有xlsx 7列结构对齐"""
    indicator: str       # "日前价格" | "实时价格" | "负荷" | ...
    province: str        # "山东省"
    timestamp: datetime  # 精确到分钟
    value: float
    unit: str            # "元/兆瓦时" | "MW"
    source: str          # "dianchacha" | "sgcc_pmos"
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "indicator": self.indicator,
            "province": self.province,
            "timestamp": self.timestamp,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }


class SourceAdapter(ABC):
    """所有数据源适配器的基类"""

    name: str = "base"  # 子类覆盖

    @abstractmethod
    async def fetch_day(
        self, province: ProvinceSpec, target_date: date
    ) -> list[RawRecord]:
        """抓取指定省份指定日期的全部指标数据"""
        ...

    @abstractmethod
    async def check_availability(self, province: ProvinceSpec) -> bool:
        """检查该数据源对该省份是否可用"""
        ...

    async def fetch_range(
        self,
        province: ProvinceSpec,
        start: date,
        end: date,
        on_day_done: callable = None,
    ) -> list[RawRecord]:
        """
        按天循环抓取日期范围。
        内置频率控制、重试、断点续传。
        """
        all_records = []
        current = start
        day_count = 0

        while current <= end:
            try:
                records = await self._fetch_with_retry(province, current)
                all_records.extend(records)
                day_count += 1

                if on_day_done:
                    on_day_done(current, len(records))

                logger.info(
                    "[{}] {} {} → {}条记录",
                    self.name, province.name_cn, current, len(records),
                )

            except Exception as e:
                logger.error(
                    "[{}] {} {} 失败: {}",
                    self.name, province.name_cn, current, e,
                )

            # 频率控制
            delay = random.uniform(
                settings.min_request_interval,
                settings.max_request_interval,
            )
            await asyncio.sleep(delay)

            # 批次冷却
            if day_count % settings.burst_limit == 0:
                logger.info("Burst cooldown {}s...", settings.burst_cooldown)
                await asyncio.sleep(settings.burst_cooldown)

            current += timedelta(days=1)

        logger.info(
            "[{}] {} 范围抓取完成: {}~{}, 共{}天{}条",
            self.name, province.name_cn, start, end, day_count, len(all_records),
        )
        return all_records

    async def _fetch_with_retry(
        self, province: ProvinceSpec, target_date: date
    ) -> list[RawRecord]:
        last_error = None
        for attempt in range(1, settings.max_retries + 1):
            try:
                return await self.fetch_day(province, target_date)
            except Exception as e:
                last_error = e
                if attempt < settings.max_retries:
                    wait = settings.retry_delay * attempt
                    logger.warning(
                        "[{}] 第{}次重试 ({}/{}), 等待{}s: {}",
                        self.name, attempt, attempt, settings.max_retries, wait, e,
                    )
                    await asyncio.sleep(wait)
        raise last_error
