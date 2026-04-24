"""
Net Load Forecaster
===================

预测 D 日 96 点净负荷（= 负荷 − 风电 − 光伏），作为 DA 价预测的底层驱动因子。

第一性原理：
  净负荷 = load(t) − wind(t) − solar(t)
  受以下因素驱动：
    - 时段（小时 sin/cos）→ 决定日内负荷曲线形态
    - 日历（工作日 / 周末 / 节假日）→ 工业负荷强相关
    - 季节 → 空调 / 采暖
    - 天气（温度、风速、光照）→ 负荷 + 新能源出力
    - 短期趋势（昨天同时段、上周同时段）→ 自回归

设计选择：
  单模型 + 小时作为特征，而不是 96 个独立模型。理由：
    - 小时间强相关，单模型能共享信息
    - 样本少时（1-2 年）单模型更稳
    - LightGBM 对 tabular 友好

不做：
  - 神经网络（数据量不够 + 额外复杂度无回报）
  - 预测太长的 horizon（> 48h 信息稀薄）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger

from core.calendar_features import add_calendar_features, CALENDAR_COLS


def _lazy_lgb():
    import lightgbm as lgb
    return lgb


STEPS_PER_DAY = 96


@dataclass
class NetLoadConfig:
    lag_days: tuple[int, ...] = (1, 2, 7)       # 用昨天 / 前天 / 上周同时段
    use_weather: bool = True
    n_estimators: int = 400
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8


class NetLoadForecaster:
    """
    Predict net_load[t] given D-1 information set.

    Fit:
        f = NetLoadForecaster()
        f.fit(df_train)   # df has load_mw, wind_mw, solar_mw, calendar, weather

    Predict one day:
        net_load_96 = f.predict_day(target_day_idx=d, df=df)
    """

    def __init__(self, config: NetLoadConfig | None = None):
        self.config = config or NetLoadConfig()

        self.model = None
        self.feature_columns: list[str] | None = None
        self.train_rmse: float | None = None

    @staticmethod
    def _ensure_net_load(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "net_load" not in df.columns:
            wind = df["wind_mw"].fillna(0) if "wind_mw" in df.columns else 0
            solar = df["solar_mw"].fillna(0) if "solar_mw" in df.columns else 0
            df["net_load"] = df["load_mw"].fillna(0) - wind - solar
        return df

    def _build_feature_matrix(
        self,
        df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        构建 (X, y)，其中 y 是 net_load[t]，X 是该 t 的特征：
          - 日历
          - lag_{d,h}: net_load at t - d*96
          - weather at t
        """
        df = self._ensure_net_load(df)
        if "hour" not in df.columns:
            df = add_calendar_features(df)

        # Lags
        for d in self.config.lag_days:
            shift = d * STEPS_PER_DAY
            df[f"net_load_lag_{d}d"] = df["net_load"].shift(shift)

        # 7-day rolling mean (same hour-of-day)
        df["net_load_rolling_7d"] = df["net_load"].shift(STEPS_PER_DAY).rolling(7 * STEPS_PER_DAY).mean()

        weather_cols = []
        if self.config.use_weather:
            for c in ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]:
                if c in df.columns:
                    weather_cols.append(c)

        feat_cols = (
            CALENDAR_COLS
            + [f"net_load_lag_{d}d" for d in self.config.lag_days]
            + ["net_load_rolling_7d"]
            + weather_cols
        )
        feat_cols = [c for c in feat_cols if c in df.columns]

        # Drop rows with NaN lag (burn-in)
        valid = df[feat_cols + ["net_load"]].dropna()
        X = valid[feat_cols]
        y = valid["net_load"]
        return X, y

    def fit(self, df: pd.DataFrame) -> "NetLoadForecaster":
        X, y = self._build_feature_matrix(df)
        self.feature_columns = list(X.columns)
        logger.info(f"NetLoadForecaster.fit: {len(X):,} rows, {len(self.feature_columns)} features")

        lgb = _lazy_lgb()
        self.model = lgb.LGBMRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            verbose=-1,
            random_state=42,
        )
        self.model.fit(X.values, y.values)
        y_pred = self.model.predict(X.values)
        self.train_rmse = float(np.sqrt(np.mean((y.values - y_pred) ** 2)))
        logger.info(f"  Train RMSE: {self.train_rmse:.1f} MW")
        return self

    def predict_day(
        self,
        target_day_idx: int,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """
        预测 target_day_idx 日（0-indexed）的 96 点净负荷。
        依赖 df 里 target_day_idx 日之前的 lag 数据 + target_day 日的日历/天气。

        注意：天气预报应该已经 merge 到 df 里（调用方保证）。
        """
        assert self.model is not None
        df = self._ensure_net_load(df)
        if "hour" not in df.columns:
            df = add_calendar_features(df)

        # 给 df 先加 lags（基于到 target_day - 1 的数据，自然避免 leakage）
        for d in self.config.lag_days:
            shift = d * STEPS_PER_DAY
            col = f"net_load_lag_{d}d"
            if col not in df.columns:
                df[col] = df["net_load"].shift(shift)
        if "net_load_rolling_7d" not in df.columns:
            df["net_load_rolling_7d"] = (
                df["net_load"].shift(STEPS_PER_DAY).rolling(7 * STEPS_PER_DAY).mean()
            )

        start = target_day_idx * STEPS_PER_DAY
        end = start + STEPS_PER_DAY
        if end > len(df):
            raise ValueError(f"target_day_idx {target_day_idx} 超出 df 长度")

        X_day = df[self.feature_columns].iloc[start:end].fillna(0).values
        return self.model.predict(X_day)

    def predict_range(
        self,
        start_day: int,
        end_day: int,
        df: pd.DataFrame,
    ) -> np.ndarray:
        """批量预测多天，返回 (n_days * 96,)"""
        out = []
        for d in range(start_day, end_day):
            out.append(self.predict_day(d, df))
        return np.concatenate(out)
