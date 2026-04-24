"""
Regime Classifier (Logan 独立版)
=================================

把每一天的 96 点 RT 价格 shape 归到 K 类（默认 K=12）。
然后训练一个分类器：用 D-1 及之前的特征预测 D 日属于哪类。

设计选择（第一性原理）：
  - Shape 而非 level：每天 z-score 标准化后聚类，避免高/低价日被分到不同类
  - GBM 分类器：比 NN 样本效率高，对 tabular 特征天然友好
  - 输出概率：所有下游 head 都需要 P(regime=k) 做条件推理

和 optimization/milp/scenario_generator.py 里的 RegimeClassifier 功能类似，
但接口独立 —— Logan 不依赖 ProvinceData，直接吃 DataFrame。

典型流程：
    clf = RegimeClassifier(n_regimes=12)
    clf.fit(df_train)             # df_train 必须含 rt_price + 日历/天气特征
    probs = clf.predict_proba(d_features)  # (n_regimes,)
    profiles = clf.regime_profiles()       # (n_regimes, 96) 每类的典型 RT 曲线
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from loguru import logger

from core.calendar_features import add_calendar_features


STEPS_PER_DAY = 96
DEFAULT_N_REGIMES = 12


@dataclass
class RegimeFitResult:
    n_regimes: int
    n_train_days: int
    train_labels: np.ndarray           # (n_train_days,)
    regime_profiles: np.ndarray        # (n_regimes, 96) 每类 RT 均值
    regime_counts: np.ndarray          # (n_regimes,) 每类样本数
    feature_columns: list[str]


class RegimeClassifier:
    """
    K-means on daily RT shape + GBM classifier from D-1 features.
    """

    def __init__(self, n_regimes: int = DEFAULT_N_REGIMES, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state

        self.kmeans: KMeans | None = None
        self.gbm: GradientBoostingClassifier | None = None
        self.feature_columns: list[str] | None = None
        self._train_labels: np.ndarray | None = None
        self._regime_profiles: np.ndarray | None = None  # (K, 96) in raw RT units
        self._regime_counts: np.ndarray | None = None
        self._rt_price_mean: float | None = None
        self._rt_price_std: float | None = None

    # ============================================================
    # Shape extraction
    # ============================================================

    @staticmethod
    def _daily_shape(day_prices: np.ndarray) -> np.ndarray:
        """一天 96 点价格规整到零均值单方差（shape-only）"""
        m = day_prices.mean()
        s = max(day_prices.std(), 1.0)
        return (day_prices - m) / s

    @staticmethod
    def _reshape_to_days(rt_prices: np.ndarray) -> np.ndarray:
        """(N,) → (N // 96, 96)，末尾不足一天的丢弃"""
        n_days = len(rt_prices) // STEPS_PER_DAY
        return rt_prices[: n_days * STEPS_PER_DAY].reshape(n_days, STEPS_PER_DAY)

    # ============================================================
    # Feature engineering for D-1 → D regime prediction
    # ============================================================

    def _build_day_features(
        self,
        day_idx: int,
        rt_days: np.ndarray,             # (n_days, 96)
        da_days: np.ndarray | None,      # (n_days, 96) or None
        calendar_df: pd.DataFrame,       # 按 96 步对齐的 calendar
        weather_df: pd.DataFrame | None, # optional
        labels_so_far: np.ndarray | None,
    ) -> dict:
        """
        为 "预测 day_idx 日属于哪类" 构建特征。
        输入信息集：day_idx - 1 及之前的数据，加 day_idx 日的日历 + 天气预报。

        feature 列尽量覆盖：
          - 最近一天的价格统计
          - 最近一周的价格趋势
          - 日历（D 日是什么类型）
          - D 日天气预报（如有）
        """
        f = {}
        d = day_idx - 1  # 截止到 D-1 的数据可用
        if d < 0:
            return None  # 第一天没有历史

        today_rt = rt_days[d]
        f["y_mean"] = float(today_rt.mean())
        f["y_std"] = float(today_rt.std())
        f["y_min"] = float(today_rt.min())
        f["y_max"] = float(today_rt.max())
        f["y_range"] = f["y_max"] - f["y_min"]
        f["y_peak_hour"] = int(np.argmax(today_rt)) // 4  # 峰值小时
        f["y_trough_hour"] = int(np.argmin(today_rt)) // 4

        # 六段均值
        for i, nm in enumerate(["night", "morn", "mid", "aftn", "eve", "late"]):
            f[f"y_{nm}_mean"] = float(today_rt[i * 16 : (i + 1) * 16].mean())
        f["y_morn_vs_eve"] = f["y_morn_mean"] - f["y_eve_mean"]

        # 近 7 天趋势
        if d >= 6:
            week = rt_days[d - 6 : d + 1]
            f["wk_mean"] = float(week.mean())
            f["wk_std"] = float(week.std())
            f["wk_trend"] = float(today_rt.mean() - rt_days[d - 6].mean())
        else:
            f["wk_mean"] = f["y_mean"]
            f["wk_std"] = f["y_std"]
            f["wk_trend"] = 0.0

        # 昨日 regime（如果训过）
        if labels_so_far is not None and d < len(labels_so_far):
            f["yest_reg"] = int(labels_so_far[d])
        else:
            f["yest_reg"] = -1

        # D 日 DA 曲线特征（D-1 晚间已出清，完全可知）
        if da_days is not None and day_idx < len(da_days):
            da_tomorrow = da_days[day_idx]
            if not np.isnan(da_tomorrow).all():
                f["da_mean"] = float(np.nanmean(da_tomorrow))
                f["da_std"] = float(np.nanstd(da_tomorrow))
                f["da_range"] = float(np.nanmax(da_tomorrow) - np.nanmin(da_tomorrow))
                f["da_peak_hour"] = int(np.nanargmax(da_tomorrow)) // 4

        # 日历：D 日的
        tgt_start = day_idx * STEPS_PER_DAY
        if tgt_start < len(calendar_df):
            row = calendar_df.iloc[tgt_start]
            for col in ["weekday", "month", "season", "is_weekend", "is_holiday"]:
                if col in calendar_df.columns:
                    f[f"cal_{col}"] = float(row[col])

        # 天气预报：D 日的均值/峰值
        if weather_df is not None:
            ts = day_idx * STEPS_PER_DAY
            te = (day_idx + 1) * STEPS_PER_DAY
            if te <= len(weather_df):
                for col in ["temperature_2m", "wind_speed_10m", "shortwave_radiation"]:
                    if col in weather_df.columns:
                        v = weather_df[col].iloc[ts:te].fillna(0).values
                        if len(v) > 0:
                            f[f"wx_{col}_m"] = float(v.mean())
                            f[f"wx_{col}_x"] = float(v.max())

        return f

    # ============================================================
    # Fit
    # ============================================================

    def fit(
        self,
        df: pd.DataFrame,
        rt_col: str = "rt_price",
        da_col: str = "da_price",
        weather_cols: tuple[str, ...] = ("temperature_2m", "wind_speed_10m", "shortwave_radiation"),
    ) -> RegimeFitResult:
        """
        df: 15 分钟粒度，index=timestamp，必须含 rt_price；最好含 da_price + 天气。
        """
        df = df.copy()
        if not any(col in df.columns for col in ("hour",)):
            df = add_calendar_features(df)

        # 按 96 步 reshape
        rt_values = df[rt_col].fillna(0).values.astype(np.float64)
        rt_days = self._reshape_to_days(rt_values)
        n_days = len(rt_days)

        da_days = None
        if da_col in df.columns:
            da_values = df[da_col].values.astype(np.float64)
            da_days = self._reshape_to_days(da_values)

        weather_df = df[list(weather_cols)] if all(c in df.columns for c in weather_cols) else None

        self._rt_price_mean = float(rt_values.mean())
        self._rt_price_std = float(rt_values.std())

        # Step 1: Shape clustering
        shapes = np.vstack([self._daily_shape(d) for d in rt_days])
        self.kmeans = KMeans(
            n_clusters=self.n_regimes,
            n_init=20,
            random_state=self.random_state,
        )
        self.kmeans.fit(shapes)
        labels = self.kmeans.predict(shapes)

        # Step 2: Compute regime profiles (in raw RT units)
        profiles = np.zeros((self.n_regimes, STEPS_PER_DAY))
        counts = np.zeros(self.n_regimes, dtype=np.int64)
        for k in range(self.n_regimes):
            mask = labels == k
            counts[k] = int(mask.sum())
            if mask.sum() > 0:
                profiles[k] = rt_days[mask].mean(axis=0)
            else:
                profiles[k] = rt_days.mean(axis=0)

        self._train_labels = labels
        self._regime_profiles = profiles
        self._regime_counts = counts

        # Step 3: D-1 → D classifier
        X_rows, y_rows = [], []
        for d in range(1, n_days - 1):
            feat = self._build_day_features(d + 1, rt_days, da_days, df, weather_df, labels)
            if feat is None:
                continue
            X_rows.append(feat)
            y_rows.append(labels[d + 1])

        Xdf = pd.DataFrame(X_rows).fillna(0)
        y_arr = np.array(y_rows)
        self.feature_columns = Xdf.columns.tolist()

        self.gbm = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state,
        )
        self.gbm.fit(Xdf, y_arr)

        train_acc = self.gbm.score(Xdf, y_arr)
        logger.info(
            f"RegimeClassifier fit: n_days={n_days}, n_regimes={self.n_regimes}, "
            f"train_acc={train_acc:.3f}, "
            f"regime_counts={counts.tolist()}"
        )

        return RegimeFitResult(
            n_regimes=self.n_regimes,
            n_train_days=n_days,
            train_labels=labels,
            regime_profiles=profiles,
            regime_counts=counts,
            feature_columns=list(self.feature_columns),
        )

    # ============================================================
    # Predict
    # ============================================================

    def predict_proba(
        self,
        day_idx: int,
        df: pd.DataFrame,
        rt_col: str = "rt_price",
        da_col: str = "da_price",
        weather_cols: tuple[str, ...] = ("temperature_2m", "wind_speed_10m", "shortwave_radiation"),
        labels_so_far: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        预测第 day_idx 天（0-indexed）属于每个 regime 的概率。
        df 需要包含到 day_idx-1 为止的历史数据 + day_idx 日的日历/天气。
        """
        assert self.gbm is not None, "Classifier not fitted"
        df = df.copy()
        if "hour" not in df.columns:
            df = add_calendar_features(df)

        rt_values = df[rt_col].fillna(0).values.astype(np.float64)
        rt_days = self._reshape_to_days(rt_values)
        da_days = None
        if da_col in df.columns:
            da_values = df[da_col].values.astype(np.float64)
            da_days = self._reshape_to_days(da_values)
        weather_df = df[list(weather_cols)] if all(c in df.columns for c in weather_cols) else None

        feat = self._build_day_features(day_idx, rt_days, da_days, df, weather_df, labels_so_far)
        if feat is None:
            # fallback: uniform over all regimes
            return np.ones(self.n_regimes) / self.n_regimes

        X = pd.DataFrame([feat])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns].fillna(0)

        probs_partial = self.gbm.predict_proba(X)[0]
        full = np.zeros(self.n_regimes)
        for i, c in enumerate(self.gbm.classes_):
            full[c] = probs_partial[i]
        return full

    def regime_profiles(self) -> np.ndarray:
        """(n_regimes, 96) 每类 RT 典型曲线（原始量纲）"""
        assert self._regime_profiles is not None
        return self._regime_profiles.copy()

    @property
    def train_labels(self) -> np.ndarray:
        assert self._train_labels is not None
        return self._train_labels

    def relabel_all_days(
        self,
        df: pd.DataFrame,
        rt_col: str = "rt_price",
    ) -> np.ndarray:
        """对 df 里的每一天给出 regime label（用训好的 kmeans）。"""
        assert self.kmeans is not None
        rt_values = df[rt_col].fillna(0).values.astype(np.float64)
        rt_days = self._reshape_to_days(rt_values)
        shapes = np.vstack([self._daily_shape(d) for d in rt_days])
        return self.kmeans.predict(shapes)
