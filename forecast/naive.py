"""
Naive price forecasters — baselines for MPC evaluation.
"""
from __future__ import annotations
import numpy as np


class PersistenceForecaster:
    """Forecast = current price repeated for all horizons."""

    def predict(self, prices: np.ndarray, idx: int, horizon: int = 96) -> np.ndarray:
        return np.full(horizon, prices[idx])


class YesterdaySameHourForecaster:
    """Forecast = yesterday's price at the same time of day."""

    def predict(self, prices: np.ndarray, idx: int, horizon: int = 96) -> np.ndarray:
        forecast = np.zeros(horizon)
        for h in range(horizon):
            past_idx = idx + h + 1 - 96  # same time yesterday
            if 0 <= past_idx < len(prices):
                forecast[h] = prices[past_idx]
            else:
                forecast[h] = prices[idx]  # fallback to current
        return forecast


class DAForecaster:
    """Use day-ahead prices as forecast (for same-day steps)."""

    def __init__(self, da_prices: np.ndarray):
        self.da_prices = da_prices

    def predict(self, prices: np.ndarray, idx: int, horizon: int = 96) -> np.ndarray:
        forecast = np.zeros(horizon)
        for h in range(horizon):
            target_idx = idx + h + 1
            if target_idx < len(self.da_prices):
                da = self.da_prices[target_idx]
                if da != 0 and not np.isnan(da):
                    forecast[h] = da
                else:
                    forecast[h] = prices[idx]
            else:
                forecast[h] = prices[idx]
        return forecast
