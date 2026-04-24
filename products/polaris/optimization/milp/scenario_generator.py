"""
场景生成器：K=200 条未来价格曲线

三种方法：
  A) Regime Bootstrap  — 用分类器预测 regime 概率 → 采样 K 条历史曲线
  B) Quantile Regression — LGBM 分位数回归 → 构造 K 条分位场景
  C) VAE (未实现)         — 条件变分自编码器 → 采样 K 条生成场景

方法 A 和 Regime V3 同信息量，可做最洁净对照实验。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from loguru import logger

from optimization.milp.data_loader import ProvinceData


N_REGIMES = 12


@dataclass
class ScenarioSet:
    """K 个场景的容器"""
    rt_scenarios: np.ndarray      # [K, 96] RT 价格场景
    dam_forecast: np.ndarray      # [96] DAM 预期价（取场景 DAM 的均值作为近似）
    dam_scenarios: np.ndarray     # [K, 96] DAM 价格场景
    weights: np.ndarray           # [K] 权重（通常 1/K）
    method: str                   # "bootstrap" / "quantile" / "vae"
    meta: dict                    # 额外信息（如 regime_probs）


class RegimeClassifier:
    """Regime 分类器（KMeans + GBM），复用 scripts/22_regime_v3 的逻辑"""

    def __init__(self, n_regimes: int = N_REGIMES):
        self.n_regimes = n_regimes
        self.kmeans: KMeans | None = None
        self.gbm: GradientBoostingClassifier | None = None
        self.feature_columns: list[str] | None = None

    def _daily_shape(self, day_prices: np.ndarray) -> np.ndarray:
        """把一天 96 点价格规整化为形态（减均值除标准差）"""
        m = day_prices.mean()
        s = max(day_prices.std(), 1.0)
        return (day_prices - m) / s

    def _cluster_shapes(self, rt_days: np.ndarray) -> np.ndarray:
        """对训练期所有天聚类，返回每天的 regime label"""
        shapes = np.vstack([self._daily_shape(d) for d in rt_days])
        self.kmeans = KMeans(n_clusters=self.n_regimes, n_init=20, random_state=42)
        self.kmeans.fit(shapes)
        return self.kmeans.predict(shapes)

    def _predict_regime(self, rt_days: np.ndarray) -> np.ndarray:
        """对新天数预测 regime"""
        assert self.kmeans is not None
        shapes = np.vstack([self._daily_shape(d) for d in rt_days])
        return self.kmeans.predict(shapes)

    def _build_features_for_day(
        self,
        pm_rt: np.ndarray,
        pm_dam: np.ndarray,
        df: pd.DataFrame,
        d: int,
        all_labels: np.ndarray | None,
    ) -> dict:
        """为预测 D+1 构建 D 及之前的特征（复用 22 脚本的特征工程）"""
        f = {}
        t = pm_rt[d]
        f["price_mean"], f["price_std"] = t.mean(), t.std()
        f["price_range"] = t.max() - t.min()
        f["price_min"], f["price_max"] = t.min(), t.max()
        f["price_skew"] = float(pd.Series(t).skew())

        for i, nm in enumerate(["night", "morn", "mid", "aftn", "eve", "late"]):
            f[f"{nm}_mean"] = t[i * 16:(i + 1) * 16].mean()
        f["morn_vs_eve"] = t[16:32].mean() - t[64:80].mean()

        if d >= 2:
            y = pm_rt[d - 1]
            f["y_mean"], f["y_std"], f["y_range"] = y.mean(), y.std(), y.max() - y.min()
            f["dod_change"] = t.mean() - y.mean()
        else:
            f["y_mean"], f["y_std"], f["y_range"], f["dod_change"] = t.mean(), t.std(), 0, 0

        if d >= 7:
            w = pm_rt[d - 6:d + 1]
            f["wk_mean"], f["wk_std"], f["wk_trend"] = w.mean(), w.std(), pm_rt[d].mean() - pm_rt[d - 6].mean()
        else:
            f["wk_mean"], f["wk_std"], f["wk_trend"] = t.mean(), t.std(), 0

        if all_labels is not None:
            if d < len(all_labels):
                f["today_reg"] = all_labels[d]
            if d >= 1:
                f["yest_reg"] = all_labels[d - 1]

        # DAM 日内特征
        if d < len(pm_dam):
            td = pm_dam[d]
            f["dam_mean"], f["dam_std"] = td.mean(), td.std()
            f["dam_range"] = td.max() - td.min()
            f["dam_rt_spread"] = td.mean() - t.mean()

        # 外生特征（如果有）
        for col in ["load_norm", "renewable_penetration", "wind_ratio",
                    "solar_ratio", "net_load_norm", "temperature_norm"]:
            if col in df.columns:
                v = df[col].fillna(0).values[d * 96:(d + 1) * 96]
                if len(v) > 0:
                    f[f"{col}_m"], f[f"{col}_x"] = v.mean(), v.max()

        for col in ["temperature_norm", "wind_speed_norm", "solar_radiation_norm"]:
            if col in df.columns:
                ts, te = (d + 1) * 96, (d + 2) * 96
                if te <= len(df):
                    v = df[col].fillna(0).values[ts:te]
                    if len(v) > 0:
                        f[f"tw_{col}_m"], f[f"tw_{col}_x"] = v.mean(), v.max()

        ti = (d + 1) * 96
        if ti < len(df):
            dt = df.index[ti]
            f["tw_wd"], f["tw_mo"] = dt.weekday(), dt.month
            f["tw_we"] = 1.0 if dt.weekday() >= 5 else 0.0

        return f

    def fit(
        self,
        province_data: ProvinceData,
        train_day_end: int,
    ):
        """在 [0, train_day_end) 天训练聚类+分类器"""
        train_rt = province_data.rt_prices[:train_day_end]
        train_dam = province_data.dam_prices[:train_day_end]
        df = province_data.df

        # Step 1: 聚类
        labels = self._cluster_shapes(train_rt)

        # Step 2: 建 D-1 特征预测 D 的 regime
        Xt, yt = [], []
        for d in range(7, train_day_end - 1):
            feat = self._build_features_for_day(train_rt, train_dam, df, d, labels)
            Xt.append(feat)
            yt.append(labels[d + 1])

        Xdf = pd.DataFrame(Xt)
        ya = np.array(yt)
        self.feature_columns = Xdf.columns.tolist()

        # Step 3: 训 GBM
        self.gbm = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42)
        self.gbm.fit(Xdf, ya)

        self._train_labels = labels
        self._train_end = train_day_end

    def predict_regime_probs(
        self,
        province_data: ProvinceData,
        target_day: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        为第 target_day 天预测 regime 概率。
        用 target_day - 1 及之前的数据。
        """
        assert self.gbm is not None, "Classifier not fitted"
        d = target_day - 1
        feat = self._build_features_for_day(
            province_data.rt_prices, province_data.dam_prices,
            province_data.df, d, self._extend_labels(province_data, d))
        Xp = pd.DataFrame([feat])
        for col in self.feature_columns:
            if col not in Xp.columns:
                Xp[col] = 0
        Xp = Xp[self.feature_columns]
        probs = self.gbm.predict_proba(Xp)[0]
        full_probs = np.zeros(self.n_regimes)
        for i, c in enumerate(self.gbm.classes_):
            full_probs[c] = probs[i]
        return full_probs, self.gbm.classes_

    def _extend_labels(self, province_data: ProvinceData, d: int) -> np.ndarray:
        """为当前所有天（包括测试期）预测 label"""
        all_days_rt = province_data.rt_prices[:d + 1]
        return self._predict_regime(all_days_rt)

    @property
    def train_labels(self) -> np.ndarray:
        return self._train_labels


def generate_scenarios_bootstrap(
    classifier: RegimeClassifier,
    province_data: ProvinceData,
    target_day: int,
    K: int = 200,
    rng: np.random.Generator | None = None,
) -> ScenarioSet:
    """
    Bootstrap 场景生成：
      1. 预测 target_day 的 regime 概率
      2. 对每个场景 k：按概率抽一个 regime，从训练期该 regime 的历史天中随机抽一天
      3. 用该天的 (RT, DAM) 作为场景 k
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    regime_probs, classes = classifier.predict_regime_probs(province_data, target_day)

    # 归一化概率
    regime_probs = regime_probs / regime_probs.sum()

    # 各 regime 在训练集里对应的天
    train_labels = classifier.train_labels
    train_rt = province_data.rt_prices[:len(train_labels)]
    train_dam = province_data.dam_prices[:len(train_labels)]

    rt_scenarios = np.zeros((K, 96))
    dam_scenarios = np.zeros((K, 96))

    for k in range(K):
        regime_k = rng.choice(len(regime_probs), p=regime_probs)
        candidate_days = np.where(train_labels == regime_k)[0]
        if len(candidate_days) == 0:
            candidate_days = np.arange(len(train_labels))
        day_idx = rng.choice(candidate_days)
        rt_scenarios[k] = train_rt[day_idx]
        dam_scenarios[k] = train_dam[day_idx]

    # DAM 预测：取场景 DAM 的均值（简化近似）
    dam_forecast = dam_scenarios.mean(axis=0)

    weights = np.ones(K) / K

    return ScenarioSet(
        rt_scenarios=rt_scenarios,
        dam_forecast=dam_forecast,
        dam_scenarios=dam_scenarios,
        weights=weights,
        method="bootstrap",
        meta={"regime_probs": regime_probs},
    )


def generate_scenarios_quantile(
    province_data: ProvinceData,
    target_day: int,
    K: int = 200,
    feature_lag_days: int = 7,
    rng: np.random.Generator | None = None,
) -> ScenarioSet:
    """
    Quantile Regression 场景：
      1. 对每个小时（0-95），用 LightGBM 分位数回归预测 P5/P25/P50/P75/P95
      2. 对每个场景 k：抽一个分位 u ~ U(0,1)，所有小时用同一个 u 重构一条完整曲线

    注：u 在不同小时共享 → 捕捉日内相关性的粗近似
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("需要 lightgbm: pip install lightgbm")

    if rng is None:
        rng = np.random.default_rng(seed=42)

    # 构建简单特征：价格 lag + 时段 + 星期
    train_end = target_day - feature_lag_days
    if train_end < 100:
        raise ValueError(f"训练数据不足：target_day={target_day}")

    # 每个小时训练一个模型（96 个），每个输出 5 个分位
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    hour_forecasts = np.zeros((96, 5))

    # 简化版：用过去 7 天同时段价格作特征
    for h in range(96):
        X_train = []
        y_train = []
        for d in range(feature_lag_days, train_end):
            lag_features = [province_data.rt_prices[d - k, h] for k in range(1, feature_lag_days + 1)]
            X_train.append(lag_features)
            y_train.append(province_data.rt_prices[d, h])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # 构建测试特征
        X_test = np.array([province_data.rt_prices[target_day - k, h] for k in range(1, feature_lag_days + 1)])

        for qi, q in enumerate(quantiles):
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=q, n_estimators=50, max_depth=5,
                learning_rate=0.05, verbose=-1)
            model.fit(X_train, y_train)
            pred = model.predict([X_test])[0]
            hour_forecasts[h, qi] = pred

    # 采样 K 条场景
    rt_scenarios = np.zeros((K, 96))
    for k in range(K):
        u = rng.uniform(0, 1)
        # 按分位水平重构曲线（线性插值）
        for h in range(96):
            # 在 [P5, P25, P50, P75, P95] 上线性插值
            rt_scenarios[k, h] = np.interp(u, [0.05, 0.25, 0.5, 0.75, 0.95], hour_forecasts[h])

    # DAM 简化：用场景 RT 的均值作为代理
    dam_forecast = hour_forecasts[:, 2]  # P50
    dam_scenarios = np.tile(dam_forecast, (K, 1))

    weights = np.ones(K) / K

    return ScenarioSet(
        rt_scenarios=rt_scenarios,
        dam_forecast=dam_forecast,
        dam_scenarios=dam_scenarios,
        weights=weights,
        method="quantile",
        meta={"hour_quantiles": hour_forecasts},
    )


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    from optimization.milp.data_loader import load_province, split_walkforward

    logger.info("=== 测试 Scenario Generator ===\n")

    data = load_province("shandong")
    quarters = split_walkforward(data)
    train_end = quarters[0][0]

    logger.info(f"训练分类器 (train_end={train_end})...")
    clf = RegimeClassifier(n_regimes=12)
    clf.fit(data, train_end)
    logger.info(f"分类器训练完毕，特征数={len(clf.feature_columns)}")

    # 测试 Bootstrap 场景
    logger.info("\n测试 Bootstrap 场景（K=20）...")
    target = quarters[0][0] + 5
    scen = generate_scenarios_bootstrap(clf, data, target, K=20)
    logger.info(f"  RT scenarios: shape={scen.rt_scenarios.shape}")
    logger.info(f"  RT mean: {scen.rt_scenarios.mean():.1f}")
    logger.info(f"  RT std:  {scen.rt_scenarios.std():.1f}")
    logger.info(f"  DAM forecast range: {scen.dam_forecast.min():.1f} ~ {scen.dam_forecast.max():.1f}")
    logger.info(f"  Regime probs: {scen.meta['regime_probs']}")

    # 实际 target 天的价格对比
    actual_rt = data.rt_prices[target]
    logger.info(f"  实际 RT range:  {actual_rt.min():.1f} ~ {actual_rt.max():.1f}")
    logger.info(f"  场景覆盖率（actual 在 P5-P95 区间内的比例）：")
    rt_p5 = np.percentile(scen.rt_scenarios, 5, axis=0)
    rt_p95 = np.percentile(scen.rt_scenarios, 95, axis=0)
    coverage = ((actual_rt >= rt_p5) & (actual_rt <= rt_p95)).mean()
    logger.info(f"  {coverage * 100:.1f}%  (理想 90%)")
