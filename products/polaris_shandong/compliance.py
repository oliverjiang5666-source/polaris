"""
Polaris 山东 · Bid Curve Compliance
====================================

根据 §7.2.8 验证和修正独立新型储能 bid curve。

规则摘要:
  - 充电 / 放电各最多 5 段
  - 每段区间长度 ≥ 2 MW
  - 充电第 1 段起点 = -额定充电功率，最后一段终点 = 0
  - 放电第 1 段起点 = 0，最后一段终点 = +额定放电功率
  - 两段衔接：下段起点 = 上段终点
  - 价格随出力增加单调非递减
  - 每段价在 [申报上下限] 内
  - 最小连续时长 ≥ 15 分钟整数倍

接口参照 Logan compliance.py:
  ComplianceRules.from_yaml(path) → rules
  validate(bid, rules) → ValidationResult
  enforce(bid, rules) → (fixed_bid, log)
"""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from products.polaris_shandong.bid_curve import StorageBidCurve, StorageBidSegment


@dataclass
class ComplianceRules:
    # 段数
    n_segments_max_per_side: int = 5

    # 每段最小长度
    segment_min_mw: float = 2.0

    # 精度
    price_precision: float = 0.01   # 元/MWh
    quantity_precision: float = 0.001

    # 价格限值
    bid_price_lower: float | None = None
    bid_price_upper: float | None = None

    # 单调性
    monotonic: str = "non_decreasing"

    # 最小连续时长 (分钟)
    min_continuous_minutes: int = 15
    step_minutes: int = 15

    # 电站装机
    rated_charge_power_mw: float = 200.0
    rated_discharge_power_mw: float = 200.0

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        rated_charge_power_mw: float | None = None,
        rated_discharge_power_mw: float | None = None,
    ) -> "ComplianceRules":
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        bc = y["bid_curve"]
        pl = y.get("price_limits", {})
        sp = y.get("storage_params", {}).get("required", {})
        defaults = y.get("plant_defaults", {})

        r_chg = (
            rated_charge_power_mw
            or sp.get("rated_charge_power_mw")
            or defaults.get("rated_charge_power_mw", 200.0)
        )
        r_dis = (
            rated_discharge_power_mw
            or sp.get("rated_discharge_power_mw")
            or defaults.get("rated_discharge_power_mw", 200.0)
        )

        return cls(
            n_segments_max_per_side=int(bc["charge"]["n_segments_max"]),
            segment_min_mw=float(bc["charge"]["segment_min_mw"]),
            price_precision=float(bc.get("price_precision_yuan_per_mwh", 0.01)),
            bid_price_lower=pl.get("bid_price_lower"),
            bid_price_upper=pl.get("bid_price_upper"),
            monotonic=bc["charge"].get("price_monotonic", "non_decreasing"),
            min_continuous_minutes=int(sp.get("min_continuous_minutes", 15)),
            step_minutes=15,
            rated_charge_power_mw=float(r_chg),
            rated_discharge_power_mw=float(r_dis),
        )


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate(bid: StorageBidCurve, rules: ComplianceRules) -> ValidationResult:
    """按 §7.2.8 校验充放两侧的 bid curve"""
    errors: list[str] = []

    # 充电侧
    errors.extend(_validate_side(
        bid.charge_segments,
        side="charge",
        rated_power=-rules.rated_charge_power_mw,   # magnitude 负
        expected_first_start=-rules.rated_charge_power_mw,
        expected_last_end=0.0,
        rules=rules,
    ))

    # 放电侧
    errors.extend(_validate_side(
        bid.discharge_segments,
        side="discharge",
        rated_power=+rules.rated_discharge_power_mw,
        expected_first_start=0.0,
        expected_last_end=+rules.rated_discharge_power_mw,
        rules=rules,
    ))

    # SoC 必报项
    if bid.soc_min_pct >= bid.soc_max_pct:
        errors.append(f"SoC 上下限倒置: min={bid.soc_min_pct}, max={bid.soc_max_pct}")
    if not (0 <= bid.soc_min_pct < bid.soc_max_pct <= 100):
        errors.append(f"SoC 超范围: [{bid.soc_min_pct}, {bid.soc_max_pct}] 应 ⊂ [0, 100]")

    # 充放电转换效率 1% - 99% (§7.2.8 第 4 项)
    if not (1 <= bid.round_trip_efficiency_pct <= 99):
        errors.append(f"RTE {bid.round_trip_efficiency_pct}% 超范围 [1, 99]")

    # 最小连续时长必须 15 分钟整数倍
    if bid.min_continuous_minutes % 15 != 0:
        errors.append(f"min_continuous_minutes {bid.min_continuous_minutes} 非 15 的倍数")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def _validate_side(
    segments: list[StorageBidSegment],
    side: str,
    rated_power: float,
    expected_first_start: float,
    expected_last_end: float,
    rules: ComplianceRules,
) -> list[str]:
    errors: list[str] = []
    n = len(segments)

    if n > rules.n_segments_max_per_side:
        errors.append(f"{side} 段数 {n} > {rules.n_segments_max_per_side}")
        return errors

    if n == 0:
        errors.append(f"{side} 段数为 0，需至少 1 段覆盖 [{expected_first_start}, {expected_last_end}]")
        return errors

    # 首段起点
    if abs(segments[0].start_mw - expected_first_start) > 1e-3:
        errors.append(
            f"{side} 首段起点 {segments[0].start_mw} ≠ 期望 {expected_first_start}"
        )

    # 末段终点
    if abs(segments[-1].end_mw - expected_last_end) > 1e-3:
        errors.append(
            f"{side} 末段终点 {segments[-1].end_mw} ≠ 期望 {expected_last_end}"
        )

    # 段间衔接
    for i in range(1, n):
        if abs(segments[i].start_mw - segments[i - 1].end_mw) > 1e-3:
            errors.append(
                f"{side} 段 {i+1} 起点 {segments[i].start_mw} ≠ 段 {i} 终点 {segments[i-1].end_mw}"
            )

    # 每段 ≥ 2 MW
    for i, seg in enumerate(segments):
        if seg.width_mw < rules.segment_min_mw - 1e-6:
            errors.append(
                f"{side} 段 {i+1} 宽 {seg.width_mw:.2f} MW < 最小 {rules.segment_min_mw} MW"
            )

    # 价格单调非递减
    if rules.monotonic == "non_decreasing":
        for i in range(1, n):
            if segments[i].price_yuan_mwh < segments[i - 1].price_yuan_mwh - 1e-6:
                errors.append(
                    f"{side} 段 {i+1} 价 {segments[i].price_yuan_mwh} < 段 {i} 价 {segments[i-1].price_yuan_mwh}（需单调非递减）"
                )
                break

    # 价格在上下限内
    if rules.bid_price_lower is not None:
        for i, seg in enumerate(segments):
            if seg.price_yuan_mwh < rules.bid_price_lower - 1e-6:
                errors.append(
                    f"{side} 段 {i+1} 价 {seg.price_yuan_mwh} < 下限 {rules.bid_price_lower}"
                )
                break

    if rules.bid_price_upper is not None:
        for i, seg in enumerate(segments):
            if seg.price_yuan_mwh > rules.bid_price_upper + 1e-6:
                errors.append(
                    f"{side} 段 {i+1} 价 {seg.price_yuan_mwh} > 上限 {rules.bid_price_upper}"
                )
                break

    return errors


def enforce(
    bid: StorageBidCurve,
    rules: ComplianceRules,
) -> tuple[StorageBidCurve | None, list[str]]:
    """
    尝试把 bid 修正为合规版本。
    操作:
      1. 首末端点对齐到 ±rated
      2. 段间衔接（后段起点 = 前段终点）
      3. 过小段合并或扩容
      4. 价格单调化
      5. 价格 clip 到上下限
    """
    log: list[str] = []

    fixed_charge = _enforce_side(
        bid.charge_segments,
        side="charge",
        expected_first_start=-rules.rated_charge_power_mw,
        expected_last_end=0.0,
        rules=rules,
        log=log,
    )

    fixed_discharge = _enforce_side(
        bid.discharge_segments,
        side="discharge",
        expected_first_start=0.0,
        expected_last_end=+rules.rated_discharge_power_mw,
        rules=rules,
        log=log,
    )

    # SoC 修正
    soc_min = max(0.0, min(bid.soc_min_pct, 100.0))
    soc_max = max(soc_min + 1, min(bid.soc_max_pct, 100.0))
    if soc_min != bid.soc_min_pct or soc_max != bid.soc_max_pct:
        log.append(f"SoC 范围修正 [{bid.soc_min_pct}, {bid.soc_max_pct}] → [{soc_min}, {soc_max}]")

    # RTE 修正
    rte = max(1.0, min(bid.round_trip_efficiency_pct, 99.0))
    if rte != bid.round_trip_efficiency_pct:
        log.append(f"RTE {bid.round_trip_efficiency_pct}% → {rte}%")

    # 最小连续时长 round 到 15 的倍数
    mc = max(15, (bid.min_continuous_minutes // 15) * 15)
    if mc != bid.min_continuous_minutes:
        log.append(f"min_continuous_minutes {bid.min_continuous_minutes} → {mc}")

    new_bid = StorageBidCurve(
        charge_segments=fixed_charge,
        discharge_segments=fixed_discharge,
        rated_charge_power_mw=rules.rated_charge_power_mw,
        rated_discharge_power_mw=rules.rated_discharge_power_mw,
        soc_min_pct=soc_min,
        soc_max_pct=soc_max,
        round_trip_efficiency_pct=rte,
        min_continuous_minutes=mc,
        min_standby_minutes=bid.min_standby_minutes,
        terminal_soc_target_pct=bid.terminal_soc_target_pct,
        initial_soc_pct=bid.initial_soc_pct,
        da_charge_upper_96=bid.da_charge_upper_96,
        da_discharge_upper_96=bid.da_discharge_upper_96,
        rationale=bid.rationale + (f" [enforce: {'; '.join(log)}]" if log else ""),
    )
    return new_bid, log


def _enforce_side(
    segments: list[StorageBidSegment],
    side: str,
    expected_first_start: float,
    expected_last_end: float,
    rules: ComplianceRules,
    log: list[str],
) -> list[StorageBidSegment]:
    """单侧修正"""
    if not segments:
        # 没有段 → 生成 1 段全宽（价格设为中性，等后续凸化覆盖）
        log.append(f"{side} 无段，生成默认 1 段")
        mid_price = 0.0
        if rules.bid_price_lower is not None and rules.bid_price_upper is not None:
            mid_price = 0.5 * (rules.bid_price_lower + rules.bid_price_upper)
        return [StorageBidSegment(expected_first_start, expected_last_end, mid_price)]

    # 首段起点对齐
    if abs(segments[0].start_mw - expected_first_start) > 1e-3:
        log.append(f"{side} 首段起点 {segments[0].start_mw} → {expected_first_start}")
        segments[0] = StorageBidSegment(
            expected_first_start, segments[0].end_mw, segments[0].price_yuan_mwh
        )

    # 末段终点对齐
    if abs(segments[-1].end_mw - expected_last_end) > 1e-3:
        log.append(f"{side} 末段终点 {segments[-1].end_mw} → {expected_last_end}")
        segments[-1] = StorageBidSegment(
            segments[-1].start_mw, expected_last_end, segments[-1].price_yuan_mwh
        )

    # 段间衔接强制
    for i in range(1, len(segments)):
        if abs(segments[i].start_mw - segments[i - 1].end_mw) > 1e-3:
            segments[i] = StorageBidSegment(
                segments[i - 1].end_mw, segments[i].end_mw, segments[i].price_yuan_mwh
            )
            log.append(f"{side} 段 {i+1} 起点对齐到段 {i} 终点 {segments[i-1].end_mw}")

    # 过小段处理：合并相邻段
    i = 0
    while i < len(segments):
        if segments[i].width_mw < rules.segment_min_mw - 1e-6:
            if i + 1 < len(segments):
                # 合并到下一段
                segments[i + 1] = StorageBidSegment(
                    segments[i].start_mw,
                    segments[i + 1].end_mw,
                    segments[i + 1].price_yuan_mwh,
                )
                log.append(f"{side} 段 {i+1} 过小({segments[i].width_mw:.2f} MW) 合并到段 {i+2}")
                segments.pop(i)
            elif i > 0:
                # 末段过小 → 合并到前一段
                segments[i - 1] = StorageBidSegment(
                    segments[i - 1].start_mw,
                    segments[i].end_mw,
                    segments[i - 1].price_yuan_mwh,
                )
                log.append(f"{side} 末段过小，合并到段 {i}")
                segments.pop(i)
                i -= 1
        i += 1

    # 段数超限：保留首末 + 中间按宽度保留最大 N-2
    if len(segments) > rules.n_segments_max_per_side:
        # 简化：保留首 + 末 + 中间 top-(N-2) 宽段
        widths = [(i, s.width_mw) for i, s in enumerate(segments)]
        widths.sort(key=lambda x: -x[1])
        keep_indices = set([0, len(segments) - 1])
        for idx, _ in widths:
            if len(keep_indices) >= rules.n_segments_max_per_side:
                break
            keep_indices.add(idx)
        new_segs = [segments[i] for i in sorted(keep_indices)]
        # 重新衔接
        for k in range(1, len(new_segs)):
            new_segs[k] = StorageBidSegment(
                new_segs[k - 1].end_mw, new_segs[k].end_mw, new_segs[k].price_yuan_mwh
            )
        segments = new_segs
        log.append(f"{side} 段数超限，裁剪到 {rules.n_segments_max_per_side}")

    # 价格单调非递减
    if rules.monotonic == "non_decreasing":
        for i in range(1, len(segments)):
            if segments[i].price_yuan_mwh < segments[i - 1].price_yuan_mwh:
                segments[i] = StorageBidSegment(
                    segments[i].start_mw,
                    segments[i].end_mw,
                    segments[i - 1].price_yuan_mwh + rules.price_precision,
                )
                log.append(f"{side} 段 {i+1} 价强制单调: {segments[i].price_yuan_mwh}")

    # 价格 round + clip
    for i, seg in enumerate(segments):
        p = round(seg.price_yuan_mwh / rules.price_precision) * rules.price_precision
        if rules.bid_price_lower is not None:
            p = max(p, rules.bid_price_lower)
        if rules.bid_price_upper is not None:
            p = min(p, rules.bid_price_upper)
        if abs(p - seg.price_yuan_mwh) > rules.price_precision:
            segments[i] = StorageBidSegment(seg.start_mw, seg.end_mw, p)

    return segments
