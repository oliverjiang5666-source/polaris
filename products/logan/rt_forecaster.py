"""
Logan · Head 2: RT Price Forecaster (Kalman state-space)
=========================================================

实时 RT 价预测（未来 H 步，默认 16 步 = 4 小时）。

第一性原理：
    RT[t] = DA[t] + x[t]             # DA 是零误差锚
    x[t] = phi · x[t-1] + β·u[t] + ε  # x 是残差，有均值回归

其中：
    u[t] = 实时供需偏差的驱动特征
         = [Δload_dev, Δrenewable_dev, reserve_margin]
    ε ~ N(0, Q)  # 过程噪声（可异方差）

观测方程：
    y[t] = RT[t] - DA[t]  = x[t] + v,  v ~ N(0, R)  # 理论上 v=0 但数据对齐误差导致 R>0

Kalman filter:
    更新方程标准的 Kalman 一步：
        predict:  x_t|t-1 = phi·x_{t-1}|t-1 + β·u_t
                  P_t|t-1 = phi²·P_{t-1}|t-1 + Q
        update:   K = P_t|t-1 / (P_t|t-1 + R)
                  x_t|t = x_t|t-1 + K·(y_t - x_t|t-1)
                  P_t|t = (1 - K)·P_t|t-1

为什么 Kalman 结构最优：
  1. 对均值回归过程有 built-in bias（AR 项）
  2. Online update 一行代码，实时场景必备
  3. 天然输出不确定性（P_t|t）
  4. 参数少（phi, β, Q, R）→ 样本效率高

用法：
    fcst = RTForecaster()
    fcst.fit(df_train)         # 估 phi, β, Q, R
    # 上线后每 15 分钟：
    fcst.update(observed_rt, da, net_load_deviation)
    rt_next_16 = fcst.predict(horizon=16, future_da=..., future_dev=...)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class RTForecasterConfig:
    horizon: int = 16              # 预测步数（默认 4 小时）
    phi_init: float = 0.7          # AR 系数初值
    use_drivers: bool = True       # 是否用净负荷偏差 drivers
    min_phi: float = 0.05
    max_phi: float = 0.99


@dataclass
class KalmanState:
    x_post: float = 0.0            # 当前后验状态
    P_post: float = 100.0          # 当前后验方差


class RTForecaster:
    """
    RT - DA 残差的 AR + drivers 状态空间模型。

    参数估计方法：OLS 对 fit 样本做一次回归，不做完整 EM（够用、稳、快）。
    """

    def __init__(self, config: RTForecasterConfig | None = None):
        self.config = config or RTForecasterConfig()

        # Parameters (fitted)
        self.phi: float = self.config.phi_init
        self.beta: np.ndarray | None = None        # shape (n_drivers,)
        self.driver_cols: list[str] | None = None
        self.Q: float = 1.0                        # process noise var
        self.R: float = 1.0                        # observation noise var

        # Online state
        self.state = KalmanState()
        self._fitted = False
        self.fit_rmse: float | None = None

    @staticmethod
    def _compute_drivers(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算"实时供需偏差"相关的 drivers。
        需要 df 已有：load_mw, wind_mw, solar_mw；且最好有 load_forecast, wind_forecast 等
        但客户不一定给 forecast 列，这里用 lag 自回归作 proxy。

        Drivers:
            load_delta:       load - load_lag_1d（vs 昨日同时段）
            renewable_delta:  (wind+solar) - (wind_lag+solar_lag)
            net_load_delta:   load_delta - renewable_delta
        """
        d = pd.DataFrame(index=df.index)
        STEPS = 96

        load = df["load_mw"].fillna(0) if "load_mw" in df.columns else pd.Series(0, index=df.index)
        wind = df["wind_mw"].fillna(0) if "wind_mw" in df.columns else pd.Series(0, index=df.index)
        solar = df["solar_mw"].fillna(0) if "solar_mw" in df.columns else pd.Series(0, index=df.index)

        d["load_delta"] = load - load.shift(STEPS)
        d["renewable_delta"] = (wind + solar) - (wind.shift(STEPS) + solar.shift(STEPS))
        d["net_load_delta"] = d["load_delta"] - d["renewable_delta"]

        return d.fillna(0)

    def fit(
        self,
        df: pd.DataFrame,
        rt_col: str = "rt_price",
        da_col: str = "da_price",
    ) -> "RTForecaster":
        """
        估计 phi, beta, Q, R。
        方法：
          1. y = RT - DA（残差）
          2. OLS: y[t] = phi·y[t-1] + β·u[t] + eps
          3. Q = var(eps), R 设小（假设 RT、DA 观测精确，对齐误差 << 过程噪声）
        """
        y_all = (df[rt_col] - df[da_col]).values.astype(np.float64)
        drivers = self._compute_drivers(df)
        if self.config.use_drivers:
            self.driver_cols = ["load_delta", "renewable_delta"]  # net_load_delta 近似线性组合，去掉避免共线
            U = drivers[self.driver_cols].values
        else:
            self.driver_cols = []
            U = np.zeros((len(df), 0))

        # 丢弃 NaN（主要来自 DA 缺失）
        valid = ~(np.isnan(y_all) | np.any(np.isnan(U), axis=1))
        valid[0] = False  # 需要 y[t-1]
        y = y_all[valid]
        y_lag = y_all[np.roll(valid, -1)][: len(y)]  # y[t-1]
        # 更简单更稳的取法：
        idx = np.where(valid)[0]
        idx_prev = idx - 1
        mask2 = (idx_prev >= 0) & ~np.isnan(y_all[idx_prev])
        idx = idx[mask2]
        idx_prev = idx_prev[mask2]
        y_t = y_all[idx]
        y_tm1 = y_all[idx_prev]
        U_t = U[idx]

        # OLS: y_t = phi * y_tm1 + β @ U_t + eps
        # 标准正规方程
        if len(y_t) < 100:
            logger.warning(f"RTForecaster.fit: 样本太少 ({len(y_t)})，用默认参数")
            self._fitted = True
            return self

        X = np.column_stack([y_tm1, U_t]) if U_t.shape[1] > 0 else y_tm1[:, None]
        # Ridge（很小的正则避免共线）
        ridge_eye = 1e-6 * np.eye(X.shape[1])
        theta = np.linalg.solve(X.T @ X + ridge_eye, X.T @ y_t)

        phi_raw = float(theta[0])
        self.phi = float(np.clip(phi_raw, self.config.min_phi, self.config.max_phi))
        if U_t.shape[1] > 0:
            self.beta = theta[1:].astype(np.float64)
        else:
            self.beta = np.zeros(0)

        # Residual variance = Q
        y_hat = X @ theta
        resid = y_t - y_hat
        self.Q = float(np.var(resid))
        self.fit_rmse = float(np.sqrt(np.mean(resid ** 2)))

        # R：假设 RT - DA 观测无额外噪声（= 对齐误差 ~0），设很小
        self.R = max(self.Q * 0.01, 1.0)

        self._fitted = True
        logger.info(
            f"RTForecaster fit: phi={self.phi:.3f}, beta={self.beta.tolist() if self.beta.size else '[]'}, "
            f"Q={self.Q:.1f}, R={self.R:.1f}, RMSE={self.fit_rmse:.1f}"
        )
        return self

    # ============================================================
    # Online Kalman filter
    # ============================================================

    def reset_state(self, x0: float = 0.0, P0: float = 100.0) -> None:
        self.state = KalmanState(x_post=x0, P_post=P0)

    def update(
        self,
        observed_rt: float,
        observed_da: float,
        drivers: np.ndarray | None = None,
    ) -> None:
        """在 t 时刻观测到 (RT, DA, drivers) 后，更新 Kalman 状态。"""
        assert self._fitted, "Not fitted"
        y = observed_rt - observed_da

        u_contrib = 0.0
        if self.beta is not None and drivers is not None and len(drivers) == len(self.beta):
            u_contrib = float(np.dot(self.beta, drivers))

        # Predict
        x_pri = self.phi * self.state.x_post + u_contrib
        P_pri = self.phi ** 2 * self.state.P_post + self.Q

        # Update
        K = P_pri / (P_pri + self.R)
        x_post = x_pri + K * (y - x_pri)
        P_post = (1 - K) * P_pri

        self.state = KalmanState(x_post=x_post, P_post=P_post)

    def predict(
        self,
        horizon: int | None = None,
        future_da: np.ndarray | None = None,
        future_drivers: np.ndarray | None = None,
    ) -> dict:
        """
        预测未来 horizon 步。

        Args:
            horizon: 预测步数（默认 config.horizon）
            future_da: (H,) D 日未来 H 步的 DA 价（必须提供，因为 RT = DA + x）
            future_drivers: (H, n_drivers) 未来的 drivers（可选）

        Returns:
            {
                "rt_mean": (H,) 点估计
                "rt_std":  (H,) 标准差
                "residual_mean": (H,) x 的均值
            }
        """
        assert self._fitted, "Not fitted"
        H = horizon or self.config.horizon
        assert future_da is not None and len(future_da) >= H, "future_da 必须提供 H 步"

        x = self.state.x_post
        P = self.state.P_post

        x_path = np.zeros(H)
        P_path = np.zeros(H)

        for t in range(H):
            u_t = 0.0
            if self.beta is not None and self.beta.size and future_drivers is not None:
                u_t = float(np.dot(self.beta, future_drivers[t]))
            x = self.phi * x + u_t
            P = self.phi ** 2 * P + self.Q
            x_path[t] = x
            P_path[t] = P

        rt_mean = future_da[:H] + x_path
        rt_std = np.sqrt(P_path)
        return {
            "rt_mean": rt_mean,
            "rt_std": rt_std,
            "residual_mean": x_path,
        }
