"""
LightGBM multi-horizon price forecaster.

Trains one model per forecast horizon. At prediction time,
models at selected horizons produce point forecasts, and
intermediate horizons are linearly interpolated.
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb
from loguru import logger


DEFAULT_HORIZONS = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96]


class LGBMForecaster:

    def __init__(
        self,
        horizons: list[int] | None = None,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        verbose: int = -1,
    ):
        self.horizons = horizons or DEFAULT_HORIZONS
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            verbose=verbose,
            random_state=42,
            n_jobs=-1,
        )
        self.models: dict[int, lgb.LGBMRegressor] = {}
        self._full_prices: np.ndarray | None = None  # set externally for lag lookup

    def fit(
        self,
        features: np.ndarray,
        prices: np.ndarray,
    ):
        """
        Train one LightGBM model per horizon (vectorized).

        Args:
            features: (N, D) feature matrix (31 features)
            prices: (N,) price vector
        """
        n = len(prices)
        n_base = features.shape[1]

        for h in self.horizons:
            # Valid indices: need price at t+h and lag_96 at t+h-96
            max_t = n - h
            if max_t <= 96:
                logger.warning(f"  Horizon {h}: not enough data, skipping")
                continue

            idx = np.arange(max_t)

            # Build feature matrix: base features + price_lag_96 + target_hour_sin/cos
            X = np.zeros((len(idx), n_base + 3), dtype=np.float32)
            X[:, :n_base] = features[idx]

            # price_lag_96: yesterday's price at the TARGET step
            lag96_idx = idx + h - 96
            valid_lag = lag96_idx >= 0
            X[valid_lag, n_base] = prices[lag96_idx[valid_lag]]
            X[~valid_lag, n_base] = prices[idx[~valid_lag]]

            # Target hour encoding
            target_step_in_day = (idx + h) % 96
            target_hour = target_step_in_day / 4.0
            X[:, n_base + 1] = np.sin(2 * np.pi * target_hour / 24)
            X[:, n_base + 2] = np.cos(2 * np.pi * target_hour / 24)

            # Target
            Y = prices[idx + h]

            model = lgb.LGBMRegressor(**self.params)
            model.fit(X, Y)
            self.models[h] = model

            # Report training RMSE
            train_pred = model.predict(X)
            rmse = np.sqrt(np.mean((train_pred - Y) ** 2))
            logger.info(f"  h={h:3d} ({h*0.25:5.1f}h): RMSE={rmse:.1f}, n={len(Y)}")

        logger.info(f"  Trained {len(self.models)} horizon models")

    def predict(self, features_t: np.ndarray, idx: int, horizon: int = 96) -> np.ndarray:
        """
        Predict next `horizon` prices from features at time t.

        Args:
            features_t: (D,) feature vector at time t
            idx: current index in the FULL price array (for lag computation)
            horizon: number of steps to predict

        Returns:
            (horizon,) predicted prices
        """
        n_base = len(features_t)
        forecasts_at_horizons = {}

        for h in sorted(self.models.keys()):
            if h > horizon:
                break

            x = np.zeros((1, n_base + 3), dtype=np.float32)
            x[0, :n_base] = features_t

            # price_lag_96 for target step
            lag96_idx = idx + h - 96
            if self._full_prices is not None and 0 <= lag96_idx < len(self._full_prices):
                x[0, n_base] = self._full_prices[lag96_idx]
            elif self._full_prices is not None and idx < len(self._full_prices):
                x[0, n_base] = self._full_prices[idx]

            # target hour encoding
            target_step = (idx + h) % 96
            target_hour = target_step / 4.0
            x[0, n_base + 1] = np.sin(2 * np.pi * target_hour / 24)
            x[0, n_base + 2] = np.cos(2 * np.pi * target_hour / 24)

            pred = self.models[h].predict(x)[0]
            forecasts_at_horizons[h] = pred

        # Interpolate to full horizon
        forecast = np.zeros(horizon)
        sorted_h = sorted(forecasts_at_horizons.keys())

        if not sorted_h:
            # No models available, return current price as fallback
            if self._full_prices is not None and idx < len(self._full_prices):
                return np.full(horizon, self._full_prices[idx])
            return forecast

        for step in range(1, horizon + 1):
            if step in forecasts_at_horizons:
                forecast[step - 1] = forecasts_at_horizons[step]
            else:
                lower_h = max([h for h in sorted_h if h <= step], default=sorted_h[0])
                upper_h = min([h for h in sorted_h if h >= step], default=sorted_h[-1])
                if lower_h == upper_h:
                    forecast[step - 1] = forecasts_at_horizons[lower_h]
                else:
                    alpha = (step - lower_h) / (upper_h - lower_h)
                    forecast[step - 1] = (
                        (1 - alpha) * forecasts_at_horizons[lower_h]
                        + alpha * forecasts_at_horizons[upper_h]
                    )

        return forecast
