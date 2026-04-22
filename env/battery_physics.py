"""
电池物理模型升级 (Modification #4)

升级 3 个方面：
1. SoC 依赖的效率曲线（端点效率下降）
2. 温度修正（极端温度影响效率 3-15%）
3. SoH 动态追踪（循环累积 → 容量衰减）

对比原版 (config.py 里的 BatteryConfig)：
- 固定 RTE = 0.90 → 变量化的 RTE
- 额定容量 400 MWh → 实际可用 = 400 × SoH
- 不追踪老化 → 每步更新 SoH

引用：
- RWTH 2023 Applied Energy 121428: 实测 SoC-power 曲线
- RWTH 2024 Applied Energy 127340: 效率-负载曲线
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from config import BatteryConfig


@dataclass
class PhysicsConfig:
    """物理模型高级配置"""
    # SoC 依赖效率
    use_soc_efficiency: bool = True
    # 温度修正
    use_temp_correction: bool = True
    optimal_temp_c: float = 25.0
    # SoH 追踪
    track_soh: bool = True
    soh_initial: float = 1.0  # 初始健康度
    # 老化模型：损失率 = base_rate × (1 + DoD_multiplier × DoD^2)
    calendar_aging_per_day: float = 0.00008  # ~3%/year
    cycle_aging_per_full_cycle: float = 0.00008  # ~80% after 2500 full cycles
    dod_penalty_factor: float = 1.5  # DoD 高的循环老化更快


class PhysicsBattery:
    """
    封装物理建模的电池类。比 BatteryConfig 更丰富。
    """

    def __init__(self, base: BatteryConfig | None = None, physics: PhysicsConfig | None = None):
        self.base = base or BatteryConfig()
        self.physics = physics or PhysicsConfig()

        self.soh = self.physics.soh_initial
        self.cumulative_throughput_mwh = 0.0

    # ============ 效率 ============

    def soc_dependent_efficiency(self, soc: float) -> tuple[float, float]:
        """
        返回 (charge_eff, discharge_eff) 在当前 SoC 下。

        经验曲线（来自 RWTH 数据 + 简化）：
        - 中段 (20-80%)：full efficiency
        - 近端 (10-20%, 80-90%)：-2%
        - 极端 (<10%, >90%)：-4%
        """
        if not self.physics.use_soc_efficiency:
            return self.base.charge_efficiency, self.base.discharge_efficiency

        base_c = self.base.charge_efficiency
        base_d = self.base.discharge_efficiency

        if 0.20 <= soc <= 0.80:
            return base_c, base_d
        elif 0.10 <= soc <= 0.90:
            return base_c * 0.98, base_d * 0.98
        else:
            return base_c * 0.94, base_d * 0.94

    def temperature_correction(self, temp_c: float) -> float:
        """温度修正系数（乘到基础效率上）"""
        if not self.physics.use_temp_correction:
            return 1.0

        deviation = abs(temp_c - self.physics.optimal_temp_c)
        # 偏离每 5°C 降 1%，最多降 15%
        correction = 1.0 - min(deviation * 0.002, 0.15)
        return max(correction, 0.85)

    def effective_efficiency(self, soc: float, temp_c: float = 25.0) -> tuple[float, float]:
        """完整效率（SoC + 温度）"""
        soc_c, soc_d = self.soc_dependent_efficiency(soc)
        temp_cor = self.temperature_correction(temp_c)
        return soc_c * temp_cor, soc_d * temp_cor

    # ============ 容量 ============

    @property
    def effective_capacity_mwh(self) -> float:
        """实际可用容量 = 名义 × SoH"""
        return self.base.capacity_mwh * self.soh

    @property
    def effective_capacity_mw(self) -> float:
        """实际可用功率 = 名义 × SoH（简化：功率衰减与容量同步）"""
        return self.base.capacity_mw * self.soh

    # ============ 老化更新 ============

    def update_aging(self, energy_mwh: float, dt: float, dod: float = 0.5):
        """
        根据这一步的吞吐量更新 SoH。

        公式：
            calendar_loss = calendar_rate × dt
            cycle_loss = cycle_rate × |energy|/cap × (1 + dod_factor × dod^2)
            SoH -= (calendar_loss + cycle_loss)
        """
        if not self.physics.track_soh:
            return

        dt_days = dt / 24.0
        calendar_loss = self.physics.calendar_aging_per_day * dt_days

        cycle_stress = abs(energy_mwh) / self.base.capacity_mwh
        cycle_loss = (self.physics.cycle_aging_per_full_cycle * cycle_stress *
                     (1 + self.physics.dod_penalty_factor * dod ** 2))

        self.soh = max(self.soh - calendar_loss - cycle_loss, 0.70)
        self.cumulative_throughput_mwh += abs(energy_mwh)

    # ============ Step 函数 ============

    def step(
        self,
        power_mw: float,
        price: float,
        soc: float,
        temp_c: float = 25.0,
        degradation_cost_per_mwh: float = 2.0,
    ) -> tuple[float, float, float, dict]:
        """
        物理模型 step（升级版，替代 _step_battery）。
        """
        dt = self.base.interval_hours
        eff_cap_mw = self.effective_capacity_mw
        eff_cap_mwh = self.effective_capacity_mwh

        # 效率（动态）
        eta_c, eta_d = self.effective_efficiency(soc, temp_c)

        # 限功率（考虑 SoH）
        power_mw = float(np.clip(power_mw, -eff_cap_mw, eff_cap_mw))
        energy_mwh = power_mw * dt

        if energy_mwh > 0:  # 放电
            soc_change = -energy_mwh / eff_cap_mwh / eta_d
        elif energy_mwh < 0:  # 充电
            soc_change = -energy_mwh * eta_c / eff_cap_mwh
        else:
            soc_change = 0.0

        new_soc = soc + soc_change

        # 边界处理
        if new_soc > self.base.max_soc or new_soc < self.base.min_soc:
            if new_soc > self.base.max_soc:
                actual_sc = self.base.max_soc - soc
            else:
                actual_sc = self.base.min_soc - soc
            if energy_mwh > 0:
                energy_mwh = -actual_sc * eff_cap_mwh * eta_d
            elif energy_mwh < 0:
                energy_mwh = -actual_sc * eff_cap_mwh / eta_c
            else:
                energy_mwh = 0.0
            power_mw = energy_mwh / dt
            new_soc = soc + actual_sc

        # 收入
        revenue = energy_mwh * price
        degradation = abs(energy_mwh) * degradation_cost_per_mwh

        # 更新 SoH
        dod = abs(new_soc - soc)
        self.update_aging(energy_mwh, dt, dod)

        info = {
            "efficiency_charge": eta_c,
            "efficiency_discharge": eta_d,
            "soh": self.soh,
            "effective_capacity_mwh": eff_cap_mwh,
        }

        return new_soc, revenue - degradation, energy_mwh, info


if __name__ == "__main__":
    from loguru import logger

    # Test 1: 效率曲线
    pb = PhysicsBattery()
    logger.info("SoC 依赖效率曲线:")
    for soc in [0.05, 0.15, 0.50, 0.85, 0.95]:
        c, d = pb.effective_efficiency(soc, 25.0)
        rte = c * d
        logger.info(f"  SoC={soc}: charge={c:.4f}, discharge={d:.4f}, RTE={rte:.4f}")

    # Test 2: 温度影响
    logger.info("\n温度修正（SoC=0.5）:")
    for temp in [0, 15, 25, 35, 45]:
        c, d = pb.effective_efficiency(0.5, temp)
        rte = c * d
        logger.info(f"  T={temp}°C: RTE={rte:.4f}")

    # Test 3: 长期老化仿真
    logger.info("\n长期老化仿真（每天一次满充满放，5年）:")
    pb2 = PhysicsBattery()
    n_years = 5
    n_days = 365 * n_years
    for d in range(n_days):
        # 一次满充满放
        pb2.update_aging(400.0, 4.0, dod=0.9)  # 放电
        pb2.update_aging(400.0, 4.0, dod=0.9)  # 充电
    logger.info(f"  5 年后 SoH = {pb2.soh:.4f}")
    logger.info(f"  累积吞吐: {pb2.cumulative_throughput_mwh/1e6:.2f} GWh")
    logger.info(f"  等效循环数: {pb2.cumulative_throughput_mwh / (2 * pb2.base.capacity_mwh):.0f}")
