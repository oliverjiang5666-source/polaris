"""
4个Baseline策略

所有策略输入：当前observation（和RL agent看到的一样）
输出：action index (0-4)
"""
import numpy as np
from data.features import FEATURE_COLS


def tou_strategy(obs: np.ndarray, hour: float) -> int:
    """固定时段：凌晨充电，傍晚放电"""
    if 1 <= hour <= 5:     # 凌晨低谷 → 快充
        return 2
    elif 17 <= hour <= 21:  # 傍晚高峰 → 快放
        return 4
    return 0                # 其余等待


def threshold_strategy(obs: np.ndarray, price: float, ma_96: float,
                       charge_ratio: float = 0.7, discharge_ratio: float = 1.3) -> int:
    """价格阈值：低于24h均价×charge_ratio充，高于×discharge_ratio放"""
    if ma_96 <= 0:
        return 0
    ratio = price / ma_96
    slow_charge = charge_ratio + 0.15
    slow_discharge = discharge_ratio - 0.15
    if ratio < charge_ratio:
        return 2  # 快充
    elif ratio < slow_charge:
        return 1  # 慢充
    elif ratio > discharge_ratio:
        return 4  # 快放
    elif ratio > slow_discharge:
        return 3  # 慢放
    return 0


def intraday_strategy(prices_so_far: np.ndarray, current_price: float, soc: float) -> int:
    """日内策略：用已见到的价格（非后视）判断高低"""
    if len(prices_so_far) < 4:
        return 0

    # 用已见到的价格分布判断当前价格是高还是低
    p25 = np.percentile(prices_so_far, 25)
    p75 = np.percentile(prices_so_far, 75)

    if current_price <= p25 and soc < 0.85:
        return 2  # 快充（价格在低25%区间）
    elif current_price <= np.percentile(prices_so_far, 40) and soc < 0.7:
        return 1  # 慢充
    elif current_price >= p75 and soc > 0.15:
        return 4  # 快放（价格在高25%区间）
    elif current_price >= np.percentile(prices_so_far, 60) and soc > 0.3:
        return 3  # 慢放
    return 0


def dam_threshold(obs: np.ndarray, price: float, ma_96: float,
                  dam_position: float, soc: float,
                  charge_ratio: float = 0.7, discharge_ratio: float = 1.3) -> int:
    """DAM-aware Threshold：结合RTM价格比率 + DAM日内位置做决策。
    dam_position=0→今天DAM最低点，1→最高点。合法前瞻（DAM昨天公布）。
    """
    if ma_96 <= 0:
        return 0
    ratio = price / ma_96

    # DAM说当前是低价时段 + RTM也偏低 → 充电
    if dam_position < 0.25 and ratio < charge_ratio + 0.15:
        return 2 if dam_position < 0.15 else 1
    # DAM说当前是高价时段 + RTM也偏高 → 放电
    if dam_position > 0.75 and ratio > discharge_ratio - 0.15:
        return 4 if dam_position > 0.85 else 3
    # RTM单独触发（回退到标准Threshold）
    if ratio < charge_ratio:
        return 2
    elif ratio < charge_ratio + 0.15:
        return 1
    elif ratio > discharge_ratio:
        return 4
    elif ratio > discharge_ratio - 0.15:
        return 3
    return 0


def do_nothing(obs: np.ndarray) -> int:
    """不操作"""
    return 0


def hindsight_oracle(idx: int, prices: np.ndarray, soc: float) -> int:
    """Hindsight oracle：看未来24h价格，percentile择时充放。
    用percentile而非min-max position，确保充放时间窗口对称（各~6h/天）。
    仅用于生成BC训练数据，不能实盘使用。
    """
    price = prices[idx]
    n = len(prices)
    end = min(idx + 97, n)
    future = prices[idx + 1:end]
    if len(future) < 8:
        return 0

    # 极端价格立即响应
    if price > 500 and soc > 0.10:
        return 4  # 极高价→快放
    if price < -10 and soc < 0.90:
        return 2  # 负电价→快充

    future_max = float(np.max(future))
    future_min = float(np.min(future))
    spread = future_max - future_min
    if spread < 10:
        return 0  # 价差太小

    # 用percentile确保充放窗口对称（不受价格右偏影响）
    sorted_future = np.sort(future)
    percentile = float(np.searchsorted(sorted_future, price)) / len(future)

    # 价格在未来24h底部25%→充电
    if percentile < 0.15 and soc < 0.90:
        return 2  # 快充
    if percentile < 0.25 and soc < 0.80:
        return 1  # 慢充

    # 价格在未来24h顶部25%→放电
    if percentile > 0.85 and soc > 0.10:
        return 4  # 快放
    if percentile > 0.75 and soc > 0.20:
        return 3  # 慢放

    return 0
