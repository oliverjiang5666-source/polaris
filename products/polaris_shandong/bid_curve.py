"""
Polaris 山东 · Storage Bid Curve
================================

独立新型储能电站的日前 bid curve 数据结构（§7.2.8）:

    充电曲线: 最多 5 段
      第 1 段起点 = -额定充电功率（负值）
      最后一段终点 = 0
      每段 ≥ 2 MW，单调非递减

    放电曲线: 最多 5 段
      第 1 段起点 = 0
      最后一段终点 = +额定放电功率（正值）
      每段 ≥ 2 MW，单调非递减

    曲线共用 96 时段（封存报价，§9.2.1 日内/实时用日前封存申报出清）

和 Logan DailyBid 的区别：
    Logan: 全天一条曲线，3-10 段，只放电（新能源）
    Polaris 山东: 充+放两条曲线，各 5 段，两侧都有
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class StorageBidSegment:
    """单段储能 bid（充电段 quantity 为负，放电段为正）"""
    start_mw: float          # 区间起点
    end_mw: float            # 区间终点
    price_yuan_mwh: float    # 该段报价

    @property
    def width_mw(self) -> float:
        return abs(self.end_mw - self.start_mw)


@dataclass
class StorageBidCurve:
    """
    一天封存的储能 bid curve（充+放）。
    在日前/日内/实时市场复用（封存报价）。
    """
    charge_segments: list[StorageBidSegment] = field(default_factory=list)
    discharge_segments: list[StorageBidSegment] = field(default_factory=list)

    # 必报参数
    rated_charge_power_mw: float = 0.0       # 额定充电功率（>0 magnitude）
    rated_discharge_power_mw: float = 0.0
    soc_min_pct: float = 5.0
    soc_max_pct: float = 95.0
    round_trip_efficiency_pct: float = 90.0
    min_continuous_minutes: int = 15
    min_standby_minutes: int = 15

    # 可选
    terminal_soc_target_pct: float | None = None  # 期望终态 SoC
    initial_soc_pct: float = 50.0

    # 96 点充放电计划上限（运行日充电出力上限 + 放电出力上限，§7.2.8 第 2 项）
    da_charge_upper_96: np.ndarray = field(
        default_factory=lambda: np.zeros(96)
    )
    da_discharge_upper_96: np.ndarray = field(
        default_factory=lambda: np.zeros(96)
    )

    rationale: str = ""

    @property
    def n_charge_segments(self) -> int:
        return len(self.charge_segments)

    @property
    def n_discharge_segments(self) -> int:
        return len(self.discharge_segments)

    def cleared_power(self, lmp: float, forecast_upper_mw: float = np.inf) -> float:
        """
        给定该时段 LMP 和该时段允许的充电/放电上限，返回出清功率（正=放电、负=充电）。

        逻辑（§8.4.4 SCED 逻辑的简化版）:
          - 看放电曲线: 价格 ≤ LMP 的段累加 q → 出清放电量
          - 看充电曲线: 价格 ≥ LMP 的段累加 |q| → 出清充电量（储能愿意在价高时放弃充电，价低时充更多）
          - 两者不可同时激活（物理约束），取哪个取决于 LMP 相对 bid curve 的位置
        """
        # 放电侧：累加所有报价 ≤ LMP 的段
        discharge = 0.0
        for seg in self.discharge_segments:
            # 每段报价定义为 "该段的微增电量价格"，当 LMP ≥ seg.price 整段中标
            if seg.price_yuan_mwh <= lmp:
                discharge += seg.width_mw

        # 充电侧：储能愿意在 LMP ≤ seg.price（充电报价是"愿付价"）时充电
        # 但储能充电报价单调非递减意味着：|start|（最负/最大充电功率）对应报价最低
        # 约定：充电报价 p^C_k 是"第 k 段充电一单位电量愿意付的最高价"
        # LMP ≤ p^C → 充电该段被激活
        charge = 0.0
        for seg in self.charge_segments:
            if seg.price_yuan_mwh >= lmp:
                charge += seg.width_mw  # 正数 magnitude

        # 物理：储能不能同时充放（除非带 round trip 损耗的极端情况）
        net = discharge - charge

        # Forecast / 运行日上限截断（§7.2.8 第 2 项：运行日充放电出力上限）
        # 放电截到 forecast upper
        if net > 0:
            net = min(net, forecast_upper_mw)
        elif net < 0:
            net = max(net, -forecast_upper_mw)

        return net

    def cleared_series_96(
        self,
        lmp_96: np.ndarray,
        charge_upper_96: np.ndarray | None = None,
        discharge_upper_96: np.ndarray | None = None,
        soc_aware: bool = False,
        capacity_mwh: float = 200.0,
        initial_soc_pct: float = 50.0,
        dt_hours: float = 0.25,
    ) -> np.ndarray:
        """
        返回 (96,) 每时段出清净功率（正=放电、负=充电）.

        soc_aware=False (默认):
            纯 LMP 触发, 不考虑 SoC — 用于快速验算, 可能物理不可行
        soc_aware=True:
            按时序扫 96 时段, 动态维护 SoC, 若 bid 清量超出 SoC 允许则 clip
            → 这才是 "bid curve 在 SCUC/SCED 下的真实可行 cleared"
        """
        if charge_upper_96 is None:
            charge_upper_96 = self.da_charge_upper_96
        if discharge_upper_96 is None:
            discharge_upper_96 = self.da_discharge_upper_96

        out = np.zeros(96)
        soc = initial_soc_pct / 100.0
        soc_min = self.soc_min_pct / 100.0
        soc_max = self.soc_max_pct / 100.0
        # 充放电效率 (sqrt RTE)
        eta = (self.round_trip_efficiency_pct / 100.0) ** 0.5

        for t in range(96):
            net = self.cleared_power(
                lmp=float(lmp_96[t]),
                forecast_upper_mw=float(max(charge_upper_96[t], discharge_upper_96[t])),
            )
            if net > 0:
                net = min(net, float(discharge_upper_96[t]))
            elif net < 0:
                net = max(net, -float(charge_upper_96[t]))

            if soc_aware:
                if net > 0:   # 放电
                    # SoC 下限约束: 最多放到 soc_min
                    max_discharge_energy = max(0, (soc - soc_min) * capacity_mwh) * eta  # MWh 能真正吐出
                    max_discharge_mw = max_discharge_energy / dt_hours
                    net = min(net, max_discharge_mw)
                    # 更新 SoC: 放电 net*dt MWh → 需要消耗 net*dt/eta MWh 储量
                    soc -= (net * dt_hours) / eta / capacity_mwh
                elif net < 0:  # 充电 (net < 0, |net| = 充电 MW)
                    max_charge_mwh_in = max(0, (soc_max - soc) * capacity_mwh) / eta
                    max_charge_mw = max_charge_mwh_in / dt_hours
                    if abs(net) > max_charge_mw:
                        net = -max_charge_mw
                    # 充电: -net*dt MWh 进来 → 储存增加 -net*dt*eta MWh
                    soc += (-net * dt_hours) * eta / capacity_mwh
                soc = float(np.clip(soc, soc_min, soc_max))

            out[t] = net
        return out


def build_from_tensor_dp_plan(
    power_96: np.ndarray,                  # 96 点 Tensor DP 出的 power（正=放电、负=充电）
    lmp_96: np.ndarray,                    # 96 点 LMP (预期值或历史)
    rated_charge_power: float,
    rated_discharge_power: float,
    n_segments_each_side: int = 5,
    price_monotone_epsilon: float = 0.01,
    method: str = "convex_hull",           # "convex_hull" (严格) 或 "quantile" (旧启发式)
) -> tuple[list[StorageBidSegment], list[StorageBidSegment]]:
    """
    从 96 点 Tensor DP 计划转换成 5+5 段阶梯 bid（Lee-Sun 2025 §II-C）。

    两种算法:
      convex_hull: 严格 upper convex hull + Ramer-Douglas-Peucker 简化
                   保证 bid curve 在任意 LMP 下 cleared power 最接近 DP 最优
      quantile:    旧启发式分位数切 (保留做 baseline 对比)

    数学: 把 (|power|, LMP) 96 点对投到 2D 平面, 取 upper convex hull,
          取凸包上 n+1 个转折点作 bid curve 的 breakpoint.
    """
    if method == "quantile":
        return _build_quantile(
            power_96, lmp_96, rated_charge_power, rated_discharge_power,
            n_segments_each_side, price_monotone_epsilon,
        )

    # method = "convex_hull"
    discharge_mask = power_96 > 1e-6
    charge_mask = power_96 < -1e-6

    discharge_segments = _build_side_convex_hull(
        powers=power_96[discharge_mask],
        lmps=lmp_96[discharge_mask],
        rated_power=rated_discharge_power,
        n_segments=n_segments_each_side,
        sign=+1,
    )

    charge_segments = _build_side_convex_hull(
        powers=-power_96[charge_mask],          # 转 magnitude
        lmps=lmp_96[charge_mask],
        rated_power=rated_charge_power,
        n_segments=n_segments_each_side,
        sign=-1,
    )

    # 强制单调 (保险)
    _enforce_monotone(discharge_segments, price_monotone_epsilon)
    _enforce_monotone(charge_segments, price_monotone_epsilon)

    return charge_segments, discharge_segments


# ============================================================
# 严格 Upper Convex Hull (Lee-Sun 2025 §II-C)
# ============================================================

def _build_side_convex_hull(
    powers: np.ndarray,                     # >= 0 (magnitude)
    lmps: np.ndarray,
    rated_power: float,
    n_segments: int,
    sign: int,
) -> list[StorageBidSegment]:
    """
    算法:
      1. 按 |power| 升序排列 (|power|, LMP) 点集 (包括 (0, 0) 起点)
      2. 维护一个栈, 保证 slope 非递减 (upper convex hull 从低到高)
      3. 末端补齐到 rated_power
      4. Douglas-Peucker 简化到 n_segments + 1 顶点
      5. 输出 n_segments 段, 每段价取 envelope 上对应区间中点价
    """
    if len(powers) == 0:
        # 空侧: 报不可能被清的极端价
        if sign > 0:
            return [StorageBidSegment(0.0, rated_power, 100000.0)]
        else:
            return [StorageBidSegment(-rated_power, 0.0, -100000.0)]

    # Step 1: 排序
    abs_p = np.asarray(powers, dtype=np.float64)
    lmp = np.asarray(lmps, dtype=np.float64)
    order = np.argsort(abs_p)
    abs_p = abs_p[order]
    lmp = lmp[order]

    # Step 2: upper convex hull (凸包上沿)
    # 起点 (0, min_lmp - margin) 作为虚拟始点, 保证凸包从 0 开始
    hull: list[tuple[float, float]] = [(0.0, float(lmp.min()) - 1.0)]
    for p, l in zip(abs_p, lmp):
        # 如果点和最后一点同功率, 取较大 LMP (hull 上沿)
        if abs(p - hull[-1][0]) < 1e-6:
            hull[-1] = (hull[-1][0], max(hull[-1][1], float(l)))
            continue

        while len(hull) >= 2:
            (p1, l1), (p2, l2) = hull[-2], hull[-1]
            # 检查凸性: 从 p1 到 p2 的 slope 应 <= 从 p2 到 (p, l) 的 slope
            # 即 (l2 - l1) / (p2 - p1) <= (l - l2) / (p - p2)
            # 交叉乘: (l2 - l1) * (p - p2) <= (l - l2) * (p2 - p1)  [都 > 0]
            if (l2 - l1) * (p - p2) > (l - l2) * (p2 - p1) + 1e-9:
                # hull[-1] 不在 upper envelope 上, pop
                hull.pop()
            else:
                break
        hull.append((float(p), float(l)))

    # Step 3: 末端延伸到 rated_power
    if hull[-1][0] < rated_power - 1e-6:
        # 末段延伸, 价格保持不变 (保守, 防止超出历史价)
        hull.append((rated_power, hull[-1][1]))

    # 去掉虚拟起点 (0, min_lmp - 1) 如果没有实际 (0, ...) 点
    if hull[0][0] < 1e-6 and len(hull) > 1:
        # 第二个点起算
        pass  # 保留 (0, ...), 因为 bid curve 需要从 0 开始

    # Step 4: 简化到 n_segments + 1 个点
    if len(hull) > n_segments + 1:
        hull = _ramer_douglas_peucker_to_n(hull, n_segments + 1)

    # 如果凸包点不够, 补足到 n_segments + 1
    while len(hull) < n_segments + 1:
        # 在最宽段插入中点
        widths = [(hull[i+1][0] - hull[i][0], i) for i in range(len(hull) - 1)]
        _, max_i = max(widths, key=lambda x: x[0])
        mid_p = (hull[max_i][0] + hull[max_i + 1][0]) / 2
        mid_l = (hull[max_i][1] + hull[max_i + 1][1]) / 2
        hull.insert(max_i + 1, (mid_p, mid_l))

    # Step 5: 生成段
    segments = []
    for i in range(n_segments):
        p_start, l_start = hull[i]
        p_end, l_end = hull[i + 1]
        # 段内均价作报价 (也可取 l_start 为"触发价")
        seg_price = 0.5 * (l_start + l_end)

        if sign < 0:
            # 充电: 起点在负, 终点向 0
            segments.append(StorageBidSegment(-p_end, -p_start, seg_price))
        else:
            segments.append(StorageBidSegment(p_start, p_end, seg_price))

    if sign < 0:
        segments.reverse()   # 充电 bid 从 -P_max 到 0 排序

    # 确保首段起点 / 末段终点对齐 rated
    if segments:
        if sign > 0:
            segments[0].start_mw = 0.0
            segments[-1].end_mw = rated_power
        else:
            segments[0].start_mw = -rated_power
            segments[-1].end_mw = 0.0

    return segments


def _ramer_douglas_peucker_to_n(
    points: list[tuple[float, float]],
    target_n: int,
) -> list[tuple[float, float]]:
    """
    迭代移除垂直距离最小的点, 直到点数 = target_n.
    保留首末点.
    """
    pts = [list(p) for p in points]
    while len(pts) > target_n:
        min_dist = float("inf")
        min_idx = 1
        for i in range(1, len(pts) - 1):
            # pts[i] 到 line(pts[i-1], pts[i+1]) 的垂直距离
            x1, y1 = pts[i - 1]
            x2, y2 = pts[i + 1]
            x0, y0 = pts[i]
            dx, dy = x2 - x1, y2 - y1
            ln_sq = dx * dx + dy * dy
            if ln_sq < 1e-12:
                dist = 0.0
            else:
                # 投影到 [0, 1]
                t = ((x0 - x1) * dx + (y0 - y1) * dy) / ln_sq
                t = max(0.0, min(1.0, t))
                cx = x1 + t * dx
                cy = y1 + t * dy
                dist = ((x0 - cx) ** 2 + (y0 - cy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        pts.pop(min_idx)
    return [tuple(p) for p in pts]


def _enforce_monotone(segments: list[StorageBidSegment], epsilon: float):
    """保险: 强制价单调非递减"""
    for i in range(1, len(segments)):
        if segments[i].price_yuan_mwh < segments[i - 1].price_yuan_mwh:
            segments[i].price_yuan_mwh = segments[i - 1].price_yuan_mwh + epsilon


# ============================================================
# 旧启发式 (quantile): 保留做 baseline 对比
# ============================================================

def _build_quantile(
    power_96, lmp_96, rated_charge_power, rated_discharge_power,
    n_segments, price_monotone_epsilon,
):
    """旧分位数切法, 保留做 A/B 对比"""
    dt = 0.25
    discharge_mask = power_96 > 1e-6
    charge_mask = power_96 < -1e-6

    discharge_segments = _build_side_quantile(
        powers=power_96[discharge_mask],
        lmps=lmp_96[discharge_mask],
        rated_power=rated_discharge_power,
        n_segments=n_segments,
        sign=+1,
        price_monotone_epsilon=price_monotone_epsilon,
    )
    charge_segments = _build_side_quantile(
        powers=-power_96[charge_mask],
        lmps=lmp_96[charge_mask],
        rated_power=rated_charge_power,
        n_segments=n_segments,
        sign=-1,
        price_monotone_epsilon=price_monotone_epsilon,
    )
    return charge_segments, discharge_segments


def _build_side_quantile(
    powers, lmps, rated_power, n_segments, sign, price_monotone_epsilon,
):
    """单侧 bid curve 构造（5 段分位数切, 旧版）"""
    if len(powers) == 0:
        if sign > 0:
            return [StorageBidSegment(0.0, rated_power, 100000.0)]
        else:
            return [StorageBidSegment(-rated_power, 0.0, -100000.0)]
    order = np.argsort(lmps)
    sorted_lmps = lmps[order]
    sorted_powers = powers[order]
    cum_mwh = np.cumsum(sorted_powers) * 0.25
    total_mwh = cum_mwh[-1] if len(cum_mwh) > 0 else 0
    segments = []
    current_start = 0.0 if sign > 0 else -rated_power
    for k in range(n_segments):
        lo_cum = k * total_mwh / n_segments
        hi_cum = (k + 1) * total_mwh / n_segments if k < n_segments - 1 else total_mwh
        lo_idx = int(np.searchsorted(cum_mwh, lo_cum))
        hi_idx = int(np.searchsorted(cum_mwh, hi_cum))
        seg_lmps = sorted_lmps[lo_idx:max(hi_idx, lo_idx + 1)]
        seg_price = float(seg_lmps.mean()) if len(seg_lmps) > 0 else sorted_lmps.mean()
        width_mw = rated_power / n_segments
        seg_end = current_start + width_mw
        segments.append(StorageBidSegment(current_start, seg_end, seg_price))
        current_start = seg_end
    for i in range(1, len(segments)):
        if segments[i].price_yuan_mwh < segments[i-1].price_yuan_mwh:
            segments[i].price_yuan_mwh = segments[i-1].price_yuan_mwh + price_monotone_epsilon
    if sign > 0 and segments:
        segments[-1].end_mw = rated_power
    elif sign < 0 and segments:
        segments[-1].end_mw = 0.0
    return segments
