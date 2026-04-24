"""电池物理参数"""
from dataclasses import dataclass


@dataclass
class BatteryParams:
    capacity_mw: float = 200
    capacity_mwh: float = 400
    max_soc: float = 0.95
    min_soc: float = 0.05
    charge_efficiency: float = 0.9487     # sqrt(0.90)
    discharge_efficiency: float = 0.9487
    degradation_cost_per_cycle: float = 10_000  # $/cycle ($150/kWh × 400MWh / 6000 cycles)
    interval_hours: float = 0.25          # 15分钟

    @property
    def usable_mwh(self) -> float:
        return self.capacity_mwh * (self.max_soc - self.min_soc)

    @property
    def max_charge_mwh(self) -> float:
        """单个时段最大充电量MWh"""
        return self.capacity_mw * self.interval_hours

    @property
    def max_discharge_mwh(self) -> float:
        """单个时段最大放电量MWh"""
        return self.capacity_mw * self.interval_hours
