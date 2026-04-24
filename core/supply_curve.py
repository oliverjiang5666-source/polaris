"""
Supply Curve (DA price = f(net load))
======================================

第一性原理：
    DA 出清价 = SCUC 问题的对偶价 = 边际机组报价
    机组报价在短期内结构稳定 → 在给定机组池下，DA 价 ≈ 净负荷的单调函数

实现（两层）：
  Layer 1: 按 (season, hour_bucket) 分桶的 Isotonic Regression
           保证单调性，非参数，fit 快。
  Layer 2: 残差 LightGBM
           学"节假日 / 机组检修 / 跨省送电 / 极端天气"等结构外因素
  Layer 3 (optional): 残差分位数回归
           输出 DA 价的 P05/P25/P50/P75/P95

对比其他方法：
  纯 LGBM  :  没有单调性约束 → 净负荷 outlier 时会预测反方向
  NN       :  样本不够（陕甘宁可能 1-2 年），过拟合风险
  这个方法 :  结构最贴合物理，样本效率高，可解释
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.isotonic import IsotonicRegression
from loguru import logger


# lightgbm 按需导入（让单元测试 / sanity check 不强依赖）
def _lazy_lgb():
    import lightgbm as lgb
    return lgb


# ============================================================
# Config & Key schema
# ============================================================

@dataclass
class SupplyCurveConfig:
    seasonal: bool = True          # 按季节分桶
    time_of_day_split: bool = True # 按时段分桶 (4 档)
    residual_model: bool = True    # 是否加残差 GBM
    quantile_regression: bool = True
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    residual_n_estimators: int = 200
    residual_max_depth: int = 5


def _bucket_key(season: int, hour_bucket: int, cfg: SupplyCurveConfig) -> tuple:
    """把 (season, hour_bucket) 压成单一 key；如果 cfg 不分，就退化到 0"""
    s = season if cfg.seasonal else 0
    h = hour_bucket if cfg.time_of_day_split else 0
    return (int(s), int(h))


# ============================================================
# Main class
# ============================================================

class SupplyCurve:
    """
    DA = IsotonicByBucket(net_load) + ResidualGBM(features) + noise

    Fit:
        sc = SupplyCurve()
        sc.fit(df_train)             # df has net_load, da_price, calendar cols

    Predict point:
        sc.predict(net_load, season, hour_bucket, extra_features)

    Predict quantile:
        sc.predict_quantile(net_load, season, hour_bucket, q=0.9, extra)
    """

    def __init__(self, config: SupplyCurveConfig | None = None):
        self.config = config or SupplyCurveConfig()

        # bucket_key -> IsotonicRegression
        self.iso_by_key: dict[tuple, IsotonicRegression] = {}
        # bucket_key -> (x_min, x_max) for extrapolation bookkeeping
        self.iso_bounds: dict[tuple, tuple[float, float]] = {}

        # Residual models
        self.residual_features: list[str] | None = None
        self.residual_median_model = None
        self.residual_quantile_models: dict[float, object] = {}

        # Fallback
        self._global_iso: IsotonicRegression | None = None
        self._global_mean_da: float = 0.0

        # Fit stats
        self.n_train_rows: int = 0
        self.residual_rmse: float | None = None

    # ============================================================
    # Data prep
    # ============================================================

    @staticmethod
    def _ensure_net_load(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "net_load" not in df.columns:
            wind = df["wind_mw"].fillna(0) if "wind_mw" in df.columns else 0
            solar = df["solar_mw"].fillna(0) if "solar_mw" in df.columns else 0
            df["net_load"] = df["load_mw"].fillna(0) - wind - solar
        return df

    @staticmethod
    def _default_residual_features(df: pd.DataFrame) -> list[str]:
        candidates = [
            "hour", "hour_sin", "hour_cos",
            "weekday", "weekday_sin", "weekday_cos",
            "month", "month_sin", "month_cos",
            "season", "hour_bucket",
            "is_weekend", "is_holiday", "is_workday",
            "net_load",
            "temperature_2m", "wind_speed_10m", "shortwave_radiation",
            "tie_line_mw", "hydro_mw",
        ]
        return [c for c in candidates if c in df.columns]

    # ============================================================
    # Fit
    # ============================================================

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "da_price",
        min_samples_per_bucket: int = 50,
    ) -> "SupplyCurve":
        """
        df: 15 分钟粒度，index=timestamp，必须含：
            load_mw, wind_mw, solar_mw (or net_load),
            da_price (或 target_col),
            season, hour_bucket (调用 add_calendar_features 加上)
        """
        df = self._ensure_net_load(df)

        # 过滤有效样本
        mask = df[target_col].notna() & df["net_load"].notna()
        df_fit = df.loc[mask].copy()
        self.n_train_rows = len(df_fit)
        logger.info(f"SupplyCurve.fit: {self.n_train_rows:,} valid rows")

        # -------- Layer 1: Isotonic by bucket --------
        cfg = self.config
        keys_seen = set()
        for (season, hour_bucket), sub in df_fit.groupby(["season", "hour_bucket"]):
            key = _bucket_key(int(season), int(hour_bucket), cfg)
            keys_seen.add(key)
            # 如果 cfg 禁用某个维度，多个 bucket 会合并到同一 key，下面会 refit（稍浪费，但简单）

        if cfg.seasonal or cfg.time_of_day_split:
            groupers = []
            if cfg.seasonal:
                groupers.append("season")
            if cfg.time_of_day_split:
                groupers.append("hour_bucket")

            for bucket_vals, sub in df_fit.groupby(groupers):
                if not isinstance(bucket_vals, tuple):
                    bucket_vals = (bucket_vals,)
                season = int(bucket_vals[groupers.index("season")]) if cfg.seasonal else 0
                hour_bucket = int(bucket_vals[groupers.index("hour_bucket")]) if cfg.time_of_day_split else 0
                key = _bucket_key(season, hour_bucket, cfg)

                if len(sub) < min_samples_per_bucket:
                    continue
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                iso.fit(sub["net_load"].values, sub[target_col].values)
                self.iso_by_key[key] = iso
                self.iso_bounds[key] = (
                    float(sub["net_load"].min()),
                    float(sub["net_load"].max()),
                )
        else:
            # 全局一条曲线
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(df_fit["net_load"].values, df_fit[target_col].values)
            self.iso_by_key[(0, 0)] = iso
            self.iso_bounds[(0, 0)] = (
                float(df_fit["net_load"].min()),
                float(df_fit["net_load"].max()),
            )

        # 全局 fallback（某些 (season, hb) 样本不够时用）
        self._global_iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        self._global_iso.fit(df_fit["net_load"].values, df_fit[target_col].values)
        self._global_mean_da = float(df_fit[target_col].mean())

        logger.info(f"  Isotonic fitted for {len(self.iso_by_key)} buckets")

        # -------- Layer 2: Residual model --------
        if cfg.residual_model:
            df_fit["da_base"] = self._predict_base_array(df_fit)
            df_fit["residual"] = df_fit[target_col] - df_fit["da_base"]

            self.residual_features = self._default_residual_features(df_fit)
            X = df_fit[self.residual_features].fillna(0).values
            y = df_fit["residual"].values

            lgb = _lazy_lgb()
            self.residual_median_model = lgb.LGBMRegressor(
                n_estimators=cfg.residual_n_estimators,
                max_depth=cfg.residual_max_depth,
                learning_rate=0.05,
                subsample=0.8,
                verbose=-1,
                random_state=42,
            )
            self.residual_median_model.fit(X, y)
            y_pred = self.residual_median_model.predict(X)
            self.residual_rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
            logger.info(f"  Residual RMSE (train): {self.residual_rmse:.2f} 元/MWh")

            # 分位数模型
            if cfg.quantile_regression:
                for q in cfg.quantiles:
                    if q == 0.5:
                        continue  # 用 median model 代替
                    model = lgb.LGBMRegressor(
                        objective="quantile",
                        alpha=q,
                        n_estimators=cfg.residual_n_estimators,
                        max_depth=cfg.residual_max_depth,
                        learning_rate=0.05,
                        subsample=0.8,
                        verbose=-1,
                        random_state=42,
                    )
                    model.fit(X, y)
                    self.residual_quantile_models[q] = model
                logger.info(f"  Quantile models fitted for {list(self.residual_quantile_models.keys())}")

        return self

    # ============================================================
    # Prediction helpers
    # ============================================================

    def _predict_base_scalar(self, net_load: float, season: int, hour_bucket: int) -> float:
        key = _bucket_key(season, hour_bucket, self.config)
        iso = self.iso_by_key.get(key, self._global_iso)
        if iso is None:
            return self._global_mean_da
        return float(iso.predict([net_load])[0])

    def _predict_base_array(self, df: pd.DataFrame) -> np.ndarray:
        """对整个 df 按 (season, hour_bucket) 批量预测 isotonic base"""
        out = np.full(len(df), self._global_mean_da, dtype=np.float64)
        for key, iso in self.iso_by_key.items():
            season_k, hour_k = key
            mask = np.ones(len(df), dtype=bool)
            if self.config.seasonal:
                mask &= df["season"].values == season_k
            if self.config.time_of_day_split:
                mask &= df["hour_bucket"].values == hour_k
            if mask.any():
                out[mask] = iso.predict(df.loc[mask, "net_load"].values)
        return out

    # ============================================================
    # Public prediction API
    # ============================================================

    def predict(
        self,
        net_load: float | np.ndarray,
        season: int | np.ndarray,
        hour_bucket: int | np.ndarray,
        extra_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """
        点预测 DA 价。

        - net_load, season, hour_bucket 可以是标量或 1-D array（需等长）
        - extra_df：如果给了残差模型，这里提供匹配的特征 DataFrame
        """
        net_load = np.atleast_1d(net_load).astype(np.float64)
        season = np.atleast_1d(season).astype(np.int64)
        hour_bucket = np.atleast_1d(hour_bucket).astype(np.int64)
        n = len(net_load)

        base = np.zeros(n)
        for i in range(n):
            base[i] = self._predict_base_scalar(net_load[i], int(season[i]), int(hour_bucket[i]))

        if self.residual_median_model is None or extra_df is None:
            return base

        # 准备残差特征
        X = self._prepare_extra_features(extra_df, net_load, season, hour_bucket)
        residual = self.residual_median_model.predict(X)
        return base + residual

    def predict_quantile(
        self,
        net_load: float | np.ndarray,
        season: int | np.ndarray,
        hour_bucket: int | np.ndarray,
        quantile: float,
        extra_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """分位数预测 DA 价"""
        base = self.predict(net_load, season, hour_bucket, extra_df=None)

        if quantile == 0.5 or self.residual_median_model is None:
            # 退化到中位数 + 默认残差
            if self.residual_median_model is not None and extra_df is not None:
                X = self._prepare_extra_features(extra_df, np.atleast_1d(net_load), np.atleast_1d(season), np.atleast_1d(hour_bucket))
                return base + self.residual_median_model.predict(X)
            return base

        if quantile not in self.residual_quantile_models:
            raise ValueError(f"quantile {quantile} not in trained quantiles {list(self.residual_quantile_models.keys())}")

        if extra_df is None:
            # 没有 extra_df 时，只能用 base
            return base

        X = self._prepare_extra_features(extra_df, np.atleast_1d(net_load), np.atleast_1d(season), np.atleast_1d(hour_bucket))
        residual = self.residual_quantile_models[quantile].predict(X)
        return base + residual

    def _prepare_extra_features(
        self,
        extra_df: pd.DataFrame,
        net_load: np.ndarray,
        season: np.ndarray,
        hour_bucket: np.ndarray,
    ) -> np.ndarray:
        """把 extra_df 对齐到所需特征列，缺的补 0"""
        if self.residual_features is None:
            return np.zeros((len(net_load), 0))

        extra = extra_df.copy().reset_index(drop=True)
        if "net_load" not in extra.columns:
            extra["net_load"] = net_load
        if "season" not in extra.columns:
            extra["season"] = season
        if "hour_bucket" not in extra.columns:
            extra["hour_bucket"] = hour_bucket

        for col in self.residual_features:
            if col not in extra.columns:
                extra[col] = 0
        return extra[self.residual_features].fillna(0).values

    # ============================================================
    # 诊断
    # ============================================================

    def describe(self) -> dict:
        return {
            "n_buckets": len(self.iso_by_key),
            "n_train_rows": self.n_train_rows,
            "residual_rmse": self.residual_rmse,
            "has_quantile": bool(self.residual_quantile_models),
            "quantiles_available": list(self.residual_quantile_models.keys()),
        }
