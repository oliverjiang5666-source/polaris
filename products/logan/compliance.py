"""
Logan · Bid Curve Compliance
==============================

把 bid curve 强制对齐到省级规则：段数 / 每段量 / 精度 / 单调性 / 总量上限。

设计（从 gansu.yaml 读参数）：
  - min_steps / max_steps
  - min_step_fraction_of_capacity
  - price_precision_yuan_per_mwh
  - quantity_precision_mw
  - monotonic: "non_decreasing"
  - cannot_over_bid
  - bid_price_lower / upper

接口：
  - ComplianceRules.from_yaml(path, capacity_mw) → 加载规则
  - validate(bid) → (is_valid, errors)
  - round_prices(prices) → np.ndarray
  - round_quantities(quantities) → np.ndarray
  - enforce(bid) → 尝试修正（若可，round + clip；否则返回 None + 理由）
"""
from __future__ import annotations

import math
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from products.logan.bid_curve_generator import BidStep, HourlyBid


# ============================================================
# Config from YAML
# ============================================================

@dataclass
class ComplianceRules:
    # 段数约束
    min_steps: int = 3
    max_steps: int = 10

    # 每段最小占装机比例
    min_step_fraction_of_capacity: float = 0.10

    # 精度
    price_precision: float = 10.0          # 元/MWh
    quantity_precision: float = 0.001       # MW

    # 单调性: "non_decreasing" or "non_increasing"
    monotonic: str = "non_decreasing"

    # 覆盖范围: "full" (Σq = capacity) or "up_to_forecast"
    coverage: str = "full"

    # 价格上下限
    bid_price_lower: float | None = None
    bid_price_upper: float | None = None

    # 超报约束
    cannot_over_bid: bool = True

    # 电站容量（每场站注入）
    capacity_mw: float = 100.0

    @classmethod
    def from_yaml(cls, path: str | Path, capacity_mw: float) -> "ComplianceRules":
        with open(path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        b = y["bid_curve"]
        p = y.get("price_limits", {})
        o = y.get("renewable_output", {})
        return cls(
            min_steps=int(b.get("min_steps", 3)),
            max_steps=int(b.get("max_steps", 10)),
            min_step_fraction_of_capacity=float(b.get("min_step_fraction_of_capacity", 0.10)),
            price_precision=float(b.get("price_precision_yuan_per_mwh", 10.0)),
            quantity_precision=float(b.get("quantity_precision_mw", 0.001)),
            monotonic=b.get("monotonic", "non_decreasing"),
            coverage=b.get("coverage", "full"),
            bid_price_lower=(
                float(p["bid_price_lower"]) if p.get("bid_price_lower") is not None else None
            ),
            bid_price_upper=(
                float(p["bid_price_upper"]) if p.get("bid_price_upper") is not None else None
            ),
            cannot_over_bid=bool(o.get("cannot_over_bid", True)),
            capacity_mw=float(capacity_mw),
        )

    @property
    def min_step_mw(self) -> float:
        return self.min_step_fraction_of_capacity * self.capacity_mw


# ============================================================
# Rounding helpers
# ============================================================

def round_to_multiple(x: float, step: float) -> float:
    """Round x to nearest multiple of step"""
    if step <= 0:
        return x
    return round(x / step) * step


def round_price_array(prices: np.ndarray, precision: float) -> np.ndarray:
    return np.round(prices / precision) * precision


def round_quantity_array(quantities: np.ndarray, precision: float) -> np.ndarray:
    return np.round(quantities / precision) * precision


# ============================================================
# Validation
# ============================================================

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def validate_daily(daily_bid, rules: ComplianceRules) -> ValidationResult:
    """
    校验 DailyBid（全天一条曲线）是否合规。

    甘肃 V3.2 附件2 第 32 条：
        - 段数 ∈ [min_steps, max_steps]
        - 每段 ≥ 10% 装机容量
        - 单调非递减（价随量升）
        - 价格精度 10 元/MWh
        - 量精度 0.001 MW
        - Σ q_k ≤ capacity（通常要求 == capacity 覆盖全区间）
    """
    errors: list[str] = []
    steps = daily_bid.steps
    n = len(steps)

    if n < rules.min_steps:
        errors.append(f"段数 {n} < 最少 {rules.min_steps}")
    if n > rules.max_steps:
        errors.append(f"段数 {n} > 最多 {rules.max_steps}")

    # 每段量
    for i, s in enumerate(steps):
        if s.quantity_mw < rules.min_step_mw - 1e-6:
            errors.append(
                f"段 {i+1} 量 {s.quantity_mw:.3f} MW < 最小 {rules.min_step_mw:.3f} MW"
            )

    # 单调非递减（价）
    if rules.monotonic == "non_decreasing":
        for i in range(1, n):
            if steps[i].price_yuan_mwh < steps[i - 1].price_yuan_mwh - 1e-6:
                errors.append(f"段 {i+1} 价 {steps[i].price_yuan_mwh} < 前一段 {steps[i-1].price_yuan_mwh}")
                break

    # 价格精度
    for i, s in enumerate(steps):
        if abs(s.price_yuan_mwh - round_to_multiple(s.price_yuan_mwh, rules.price_precision)) > 1e-6:
            errors.append(f"段 {i+1} 价 {s.price_yuan_mwh} 未 round 到 {rules.price_precision} 精度")

    # 价格上下限
    if rules.bid_price_lower is not None:
        for i, s in enumerate(steps):
            if s.price_yuan_mwh < rules.bid_price_lower - 1e-6:
                errors.append(f"段 {i+1} 价 < 下限 {rules.bid_price_lower}")
                break
    if rules.bid_price_upper is not None:
        for i, s in enumerate(steps):
            if s.price_yuan_mwh > rules.bid_price_upper + 1e-6:
                errors.append(f"段 {i+1} 价 > 上限 {rules.bid_price_upper}")
                break

    # 总量 vs 装机
    total = sum(s.quantity_mw for s in steps)
    if total > rules.capacity_mw + 1e-3:
        errors.append(f"总量 {total:.3f} MW > 装机 {rules.capacity_mw} MW")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


def validate(bid: HourlyBid, rules: ComplianceRules, forecast_mw: float | None = None) -> ValidationResult:
    """
    Check bid against rules. Returns ValidationResult with details.
    forecast_mw: D 日该小时功率预测（for cannot_over_bid check）
    """
    errors: list[str] = []
    steps = bid.steps
    n = len(steps)

    if n == 0:
        # 不报也 OK（功率预测为 0 时）
        return ValidationResult(True)

    # 段数
    if n < rules.min_steps:
        errors.append(f"段数 {n} < 最少 {rules.min_steps}")
    if n > rules.max_steps:
        errors.append(f"段数 {n} > 最多 {rules.max_steps}")

    # 每段量
    for i, s in enumerate(steps):
        if s.quantity_mw < rules.min_step_mw - 1e-6:
            errors.append(
                f"第 {i+1} 段量 {s.quantity_mw:.3f} MW < 最小 {rules.min_step_mw:.3f} MW "
                f"(10% × capacity {rules.capacity_mw})"
            )

    # 单调性
    if rules.monotonic == "non_decreasing":
        for i in range(1, n):
            if steps[i].price_yuan_mwh < steps[i - 1].price_yuan_mwh - 1e-6:
                errors.append(f"第 {i+1} 段价 {steps[i].price_yuan_mwh} < 第 {i} 段价 {steps[i-1].price_yuan_mwh} (要求非递减)")
                break
    elif rules.monotonic == "non_increasing":
        for i in range(1, n):
            if steps[i].price_yuan_mwh > steps[i - 1].price_yuan_mwh + 1e-6:
                errors.append(f"非递增违规")
                break

    # 价格精度
    for i, s in enumerate(steps):
        if abs(s.price_yuan_mwh - round_to_multiple(s.price_yuan_mwh, rules.price_precision)) > 1e-6:
            errors.append(f"第 {i+1} 段价 {s.price_yuan_mwh} 未 round 到 {rules.price_precision} 精度")

    # 价格范围
    if rules.bid_price_lower is not None:
        for i, s in enumerate(steps):
            if s.price_yuan_mwh < rules.bid_price_lower - 1e-6:
                errors.append(f"第 {i+1} 段价 {s.price_yuan_mwh} < 下限 {rules.bid_price_lower}")
                break
    if rules.bid_price_upper is not None:
        for i, s in enumerate(steps):
            if s.price_yuan_mwh > rules.bid_price_upper + 1e-6:
                errors.append(f"第 {i+1} 段价 {s.price_yuan_mwh} > 上限 {rules.bid_price_upper}")
                break

    # 总量 vs 功率预测（不能超报）
    total = sum(s.quantity_mw for s in steps)
    if rules.cannot_over_bid and forecast_mw is not None:
        if total > forecast_mw + 1e-3:
            errors.append(f"总申报量 {total:.3f} MW > 功率预测 {forecast_mw:.3f} MW (不可超报)")

    # 总量 vs 容量
    if total > rules.capacity_mw + 1e-3:
        errors.append(f"总申报量 {total:.3f} MW > 装机 {rules.capacity_mw} MW")

    return ValidationResult(ok=(len(errors) == 0), errors=errors)


# ============================================================
# Enforcement (round + clip + repair)
# ============================================================

def enforce(
    bid: HourlyBid,
    rules: ComplianceRules,
    forecast_mw: float | None = None,
) -> tuple[HourlyBid | None, list[str]]:
    """
    尝试把 bid 修正为合规版本。
    返回 (修正后 bid, 修正日志)。若无法修正（如段数不足），返回 (None, reason)。

    操作：
    1. Round 价 到精度
    2. Round 量 到精度
    3. Clip 价格到范围
    4. Clip 总量到 forecast（若 cannot_over_bid）
    5. 移除过小段（< 10% 装机），把量合并到相邻段
    """
    log: list[str] = []
    if not bid.steps:
        return bid, []

    # Step 1: 排序 + round 价
    steps_sorted = sorted(bid.steps, key=lambda s: s.price_yuan_mwh)
    prices = np.array([s.price_yuan_mwh for s in steps_sorted], dtype=np.float64)
    quantities = np.array([s.quantity_mw for s in steps_sorted], dtype=np.float64)

    prices_r = round_price_array(prices, rules.price_precision)
    # 去重（round 后可能有重复）- 合并重复价格的量
    unique_prices, inverse = np.unique(prices_r, return_inverse=True)
    merged_qty = np.zeros_like(unique_prices, dtype=np.float64)
    for i, idx in enumerate(inverse):
        merged_qty[idx] += quantities[i]
    prices_r = unique_prices
    quantities = merged_qty

    # Step 2: Clip 价到范围
    if rules.bid_price_lower is not None:
        prices_r = np.maximum(prices_r, rules.bid_price_lower)
    if rules.bid_price_upper is not None:
        prices_r = np.minimum(prices_r, rules.bid_price_upper)

    # 再次 round + 去重（clip 后可能又出现等价点）
    prices_r = round_price_array(prices_r, rules.price_precision)
    unique_prices, inverse = np.unique(prices_r, return_inverse=True)
    merged_qty = np.zeros_like(unique_prices, dtype=np.float64)
    for i, idx in enumerate(inverse):
        merged_qty[idx] += quantities[i]
    prices_r = unique_prices
    quantities = merged_qty

    # Step 3: 过滤过小段，合并到最近段
    min_step_mw = rules.min_step_mw
    while True:
        too_small_idx = np.where((quantities > 0) & (quantities < min_step_mw - 1e-6))[0]
        if len(too_small_idx) == 0:
            break
        # 取最小的合并到相邻
        i = int(too_small_idx[0])
        if len(quantities) == 1:
            # 只剩一段且过小 → 直接设为 min_step_mw（如果 forecast 允许）
            if quantities[0] < min_step_mw and forecast_mw is not None and forecast_mw >= min_step_mw:
                quantities[0] = min_step_mw
                log.append(f"单段量 {bid.steps[0].quantity_mw:.3f} → {min_step_mw:.3f} (补齐到最小段)")
            else:
                # 无法补齐，放弃
                return None, ["无法满足最小段约束"]
            break
        # 合并到下一段（如果 i 是最后一段，合到前一段）
        if i == len(quantities) - 1:
            merge_to = i - 1
        else:
            merge_to = i + 1
        quantities[merge_to] += quantities[i]
        quantities[i] = 0.0
        # 删除零段
        mask = quantities > 1e-9
        quantities = quantities[mask]
        prices_r = prices_r[mask]
        log.append(f"段 {i+1} 量 < 10% 最小段，已合并")

    # Step 4: 控制总量（不超报）
    total = quantities.sum()
    if rules.cannot_over_bid and forecast_mw is not None and total > forecast_mw + 1e-3:
        # 按比例缩减
        scale = forecast_mw / total
        quantities = quantities * scale
        log.append(f"总量 {total:.3f} > 预测 {forecast_mw:.3f}，按比例缩减 {scale:.3f}x")
        # 可能使段变小 - 再检查 / round
        quantities = round_quantity_array(quantities, rules.quantity_precision)

    # Step 5: round 量
    quantities = round_quantity_array(quantities, rules.quantity_precision)

    # Step 6: 段数约束
    n = len(quantities)
    if n < rules.min_steps:
        # 段数不足：把最大段拆分
        while len(quantities) < rules.min_steps:
            largest_i = int(np.argmax(quantities))
            if quantities[largest_i] < 2 * min_step_mw:
                # 无法拆分
                return None, [f"段数不足 {len(quantities)} < {rules.min_steps}，且无法拆分"]
            # 拆半
            half = quantities[largest_i] / 2
            if half < min_step_mw:
                half = min_step_mw
            # 新价格 = 相邻价格中点
            if largest_i == len(prices_r) - 1:
                new_price = prices_r[largest_i] + rules.price_precision
            else:
                mid = (prices_r[largest_i] + prices_r[largest_i + 1]) / 2
                new_price = round_to_multiple(mid, rules.price_precision)
                if new_price <= prices_r[largest_i]:
                    new_price = prices_r[largest_i] + rules.price_precision
                if new_price >= prices_r[largest_i + 1]:
                    new_price = prices_r[largest_i + 1] - rules.price_precision
            if new_price <= prices_r[largest_i]:
                # 无法找到合规新价
                return None, [f"无法插入新段（价格精度限制）"]
            # 插入
            quantities[largest_i] -= half
            quantities = np.insert(quantities, largest_i + 1, half)
            prices_r = np.insert(prices_r, largest_i + 1, new_price)
            log.append(f"段数不足，在 {prices_r[largest_i]} 后插入价 {new_price} 段")
    elif n > rules.max_steps:
        return None, [f"段数 {n} > {rules.max_steps}"]

    # Final: 构造 HourlyBid
    new_steps = [
        BidStep(quantity_mw=float(q), price_yuan_mwh=float(p))
        for q, p in zip(quantities, prices_r)
        if q > 1e-9
    ]

    new_bid = HourlyBid(
        hour=bid.hour,
        steps=new_steps,
        intended_quantity=sum(s.quantity_mw for s in new_steps),
        power_forecast=bid.power_forecast,
        offset_ratio=bid.offset_ratio,
        rationale=bid.rationale + (f" [compliance: {'; '.join(log)}]" if log else ""),
    )
    return new_bid, log
