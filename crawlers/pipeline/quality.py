"""数据质量校验"""

from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from loguru import logger

from crawlers.sources.base import RawRecord
from crawlers.config.provinces import ProvinceSpec


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str = "warning"  # "warning" | "error"


@dataclass
class QualityReport:
    province: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    @property
    def errors(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "error"]

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"{self.province}: {passed}/{total} checks passed, "
            f"{len(self.errors)} errors, {len(self.warnings)} warnings"
        )


class QualityChecker:
    # 价格合理范围 (元/MWh)
    PRICE_MIN = -500
    PRICE_MAX = 2000
    # 负荷合理范围 (MW)
    LOAD_MAX = 200000
    # 相邻时间点最大价格变化
    PRICE_MAX_JUMP = 800
    # 每天应有的时间点数
    POINTS_PER_DAY_15MIN = 96
    POINTS_PER_DAY_5MIN = 288

    def check(self, records: list[RawRecord], spec: ProvinceSpec) -> QualityReport:
        report = QualityReport(province=spec.name_cn)

        if not records:
            report.checks.append(CheckResult(
                "non_empty", False, "No records", severity="error"
            ))
            return report

        report.checks.append(CheckResult("non_empty", True, f"{len(records)} records"))

        df = pd.DataFrame([r.to_dict() for r in records])

        report.checks.append(self._check_completeness(df, spec))
        report.checks.append(self._check_price_range(df))
        report.checks.append(self._check_load_range(df))
        report.checks.append(self._check_nulls(df))
        report.checks.append(self._check_duplicates(df))

        return report

    def _check_completeness(self, df: pd.DataFrame, spec: ProvinceSpec) -> CheckResult:
        """检查每天是否有足够的时间点"""
        df_ts = df.drop_duplicates(subset=["timestamp", "indicator"])
        dates = pd.to_datetime(df_ts["timestamp"]).dt.date.unique()
        expected = (
            self.POINTS_PER_DAY_5MIN if spec.settlement_interval == 5
            else self.POINTS_PER_DAY_15MIN
        )
        # 检查每个指标每天的完整性
        indicators = df_ts["indicator"].unique()
        incomplete_days = 0
        for ind in indicators[:3]:  # 抽查前3个指标
            sub = df_ts[df_ts["indicator"] == ind]
            for d in dates:
                day_count = (pd.to_datetime(sub["timestamp"]).dt.date == d).sum()
                if day_count < expected * 0.9:  # 允许10%缺失
                    incomplete_days += 1
        total_checks = min(len(indicators), 3) * len(dates)
        ratio = 1 - incomplete_days / max(total_checks, 1)
        passed = ratio > 0.85
        return CheckResult(
            "completeness", passed,
            f"完整性 {ratio:.1%} ({incomplete_days}/{total_checks} 不完整天)",
            severity="warning",
        )

    def _check_price_range(self, df: pd.DataFrame) -> CheckResult:
        price_indicators = ["日前价格", "实时价格"]
        price_df = df[df["indicator"].isin(price_indicators)]
        if price_df.empty:
            return CheckResult("price_range", True, "No price data to check")
        out_of_range = (
            (price_df["value"] < self.PRICE_MIN) | (price_df["value"] > self.PRICE_MAX)
        ).sum()
        total = len(price_df)
        passed = out_of_range / max(total, 1) < 0.01
        return CheckResult(
            "price_range", passed,
            f"价格异常 {out_of_range}/{total} ({out_of_range/max(total,1):.2%})",
            severity="warning" if out_of_range < 10 else "error",
        )

    def _check_load_range(self, df: pd.DataFrame) -> CheckResult:
        load_df = df[df["indicator"] == "负荷"]
        if load_df.empty:
            return CheckResult("load_range", True, "No load data to check")
        negative = (load_df["value"] < 0).sum()
        over_max = (load_df["value"] > self.LOAD_MAX).sum()
        bad = negative + over_max
        passed = bad == 0
        return CheckResult(
            "load_range", passed,
            f"负荷异常: {negative}个负值, {over_max}个超上限",
            severity="warning",
        )

    def _check_nulls(self, df: pd.DataFrame) -> CheckResult:
        nulls = df["value"].isna().sum()
        total = len(df)
        ratio = nulls / max(total, 1)
        return CheckResult(
            "nulls", ratio < 0.05,
            f"空值 {nulls}/{total} ({ratio:.2%})",
            severity="warning" if ratio < 0.1 else "error",
        )

    def _check_duplicates(self, df: pd.DataFrame) -> CheckResult:
        dupes = df.duplicated(subset=["indicator", "province", "timestamp"]).sum()
        return CheckResult(
            "duplicates", dupes == 0,
            f"重复记录 {dupes}",
            severity="warning",
        )
