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
    ) -> np.ndarray:
        """返回 (96,) 每时段出清净功率（正=放电、负=充电）"""
        if charge_upper_96 is None:
            charge_upper_96 = self.da_charge_upper_96
        if discharge_upper_96 is None:
            discharge_upper_96 = self.da_discharge_upper_96

        out = np.zeros(96)
        for t in range(96):
            # 充电上限从 charge_upper（正 magnitude），放电上限从 discharge_upper
            # cleared_power 内部 net > 0 用 forecast_upper 截放电，net < 0 截充电
            # 这里需要分别传
            net = self.cleared_power(
                lmp=float(lmp_96[t]),
                forecast_upper_mw=float(max(charge_upper_96[t], discharge_upper_96[t])),
            )
            # 更精细的截断
            if net > 0:
                net = min(net, float(discharge_upper_96[t]))
            elif net < 0:
                net = max(net, -float(charge_upper_96[t]))
            out[t] = net
        return out


def build_from_tensor_dp_plan(
    power_96: np.ndarray,                  # 96 点 Tensor DP 出的 power（正=放电、负=充电）
    lmp_96: np.ndarray,                    # 96 点 LMP (预期值或历史)
    rated_charge_power: float,
    rated_discharge_power: float,
    n_segments_each_side: int = 5,
    price_monotone_epsilon: float = 0.01,
) -> tuple[list[StorageBidSegment], list[StorageBidSegment]]:
    """
    从 96 点 Tensor DP 计划转换成 5+5 段阶梯 bid（convexification, Lee-Sun §II-C 简化版）。

    策略:
      放电侧: 把 power > 0 的时段按 LMP 排序，分位数切成 K 段
              每段用该分位区间的 LMP 均值作为报价
              量 = 该段在 96 时段的累计 MWh / dt 换算平均 MW
      充电侧: power < 0 的时段类似处理（价格升序对应 LMP 降序，因为充电报价愿付价）

    这不是严格的"最优 bid curve"，而是 TensorDP 策略的 convex hull 近似。
    严格版本需要解 Lee-Sun §II-C 的凸化 LP，Phase 2 再优化。
    """
    dt = 0.25
    discharge_mask = power_96 > 1e-6
    charge_mask = power_96 < -1e-6

    discharge_segments = _build_side(
        powers=power_96[discharge_mask],
        lmps=lmp_96[discharge_mask],
        rated_power=rated_discharge_power,
        n_segments=n_segments_each_side,
        sign=+1,
        price_monotone_epsilon=price_monotone_epsilon,
    )

    charge_segments = _build_side(
        powers=-power_96[charge_mask],          # 转成 magnitude
        lmps=lmp_96[charge_mask],
        rated_power=rated_charge_power,
        n_segments=n_segments_each_side,
        sign=-1,
        price_monotone_epsilon=price_monotone_epsilon,
    )

    return charge_segments, discharge_segments


def _build_side(
    powers: np.ndarray,
    lmps: np.ndarray,
    rated_power: float,
    n_segments: int,
    sign: int,                             # +1=放电, -1=充电
    price_monotone_epsilon: float,
) -> list[StorageBidSegment]:
    """单侧 bid curve 构造（5 段分位数切）"""
    if len(powers) == 0:
        # 没有该侧操作，报一段全量高价（永不中标）
        if sign > 0:
            # 放电 0 → rated，高价
            return [StorageBidSegment(0.0, rated_power, 100000.0)]
        else:
            # 充电 -rated → 0，低价（愿付价低=不充电）
            return [StorageBidSegment(-rated_power, 0.0, -100000.0)]

    # 按 LMP 排序（放电按升序：低价先中标；充电按降序：愿付高价先中标 = LMP 低时）
    # 实际上两侧都按 LMP 升序切分位数，因为 bid curve 要求价单调非递减
    order = np.argsort(lmps)
    sorted_lmps = lmps[order]
    sorted_powers = powers[order]

    # 累计 MWh 分位数切 K 段
    cum_mwh = np.cumsum(sorted_powers) * 0.25   # dt=0.25h
    total_mwh = cum_mwh[-1] if len(cum_mwh) > 0 else 0

    segments = []
    current_start = 0.0 if sign > 0 else -rated_power

    for k in range(n_segments):
        # 每段量 = 总量 / K，但最后一段补齐到 rated_power
        q_target = total_mwh / n_segments if k < n_segments - 1 else (total_mwh - k * total_mwh / n_segments)
        # 找落在 (kQ_段, (k+1)Q_段] 的 LMP 均值
        lo_cum = k * total_mwh / n_segments
        hi_cum = (k + 1) * total_mwh / n_segments if k < n_segments - 1 else total_mwh

        lo_idx = int(np.searchsorted(cum_mwh, lo_cum))
        hi_idx = int(np.searchsorted(cum_mwh, hi_cum))
        seg_lmps = sorted_lmps[lo_idx:max(hi_idx, lo_idx + 1)]
        seg_price = float(seg_lmps.mean()) if len(seg_lmps) > 0 else sorted_lmps.mean()

        # 量换算成 MW 宽度（按占 rated_power 比例）
        width_mw = rated_power / n_segments

        if sign > 0:   # 放电：正向
            seg_end = current_start + width_mw
            segments.append(StorageBidSegment(current_start, seg_end, seg_price))
            current_start = seg_end
        else:          # 充电：负向（起点更负，终点向 0）
            seg_end = current_start + width_mw
            segments.append(StorageBidSegment(current_start, seg_end, seg_price))
            current_start = seg_end

    # 强制单调非递减（§7.2.8 报价曲线应随出力增加单调非递减）
    for i in range(1, len(segments)):
        if segments[i].price_yuan_mwh < segments[i-1].price_yuan_mwh:
            segments[i].price_yuan_mwh = segments[i-1].price_yuan_mwh + price_monotone_epsilon

    # 最后一段强制终点对齐 rated
    if sign > 0 and segments:
        segments[-1].end_mw = rated_power
    elif sign < 0 and segments:
        segments[-1].end_mw = 0.0

    return segments
