# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FDHFrontierEstimator:
    """
    Single output, multiple inputs.
    FDH frontier at x0:
        y_hat(x0) = max{ y_j : x_j <= x0 (componentwise) }

    If no sample dominates x0, return NaN.
    """
    X: np.ndarray  # shape (n, m)
    y: np.ndarray  # shape (n,) or (n,1)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        y = np.asarray(self.y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1:
            raise ValueError("y must be shape (n,) or (n,1)")
        if self.X.ndim != 2:
            raise ValueError("X must be 2D array, shape (n,m)")
        if self.X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of rows")
        self.y = y

    def frontier_at(self, x0: np.ndarray) -> float:
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        n, m = self.X.shape
        if x0.shape[0] != m:
            raise ValueError(f"x0 must have length {m}, got {x0.shape[0]}")

        dominated = np.all(self.X <= x0[None, :], axis=1)  # x_j <= x0
        if not np.any(dominated):
            return float("nan")
        return float(np.max(self.y[dominated]))

    def frontier_batch(self, Xq: np.ndarray) -> np.ndarray:
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim != 2 or Xq.shape[1] != self.X.shape[1]:
            raise ValueError(f"Xq must be shape (G,{self.X.shape[1]})")
        out = np.empty(Xq.shape[0], dtype=float)
        for i in range(Xq.shape[0]):
            out[i] = self.frontier_at(Xq[i])
        return out

@dataclass
class FDHEstimatorWrapper:
    """
    统一 runner 接口的 FDH 前沿估计器（单产出，多投入）。

    接口：
        fit(X, y)  保存训练样本并构建内部 FDHFrontierEstimator
        predict(Xq) 返回每个查询投入点的前沿估计 y_hat(Xq)

    行为选项：
        - add_origin=True：给训练集额外加入一个 (0,...,0)->0 的 DMU，
          从而保证对任意 Xq>=0 至少有一个支配点（原点）可用，避免 NaN。
        - fill_value：若仍出现无支配点（或你关闭 add_origin），则用该值填充
          （建议 0.0，避免 runner 误差计算被 NaN 破坏）。
    """
    add_origin: bool = True
    fill_value: float = 0.0

    _est: Optional[FDHFrontierEstimator] = None
    _m: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FDHEstimatorWrapper":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be 2D array, shape (n,m)")
        n, m = X.shape

        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)
        if y.ndim != 1 or y.shape[0] != n:
            raise ValueError("y must be shape (n,) or (n,1) and aligned with X")

        if self.add_origin:
            X0 = np.zeros((1, m), dtype=float)
            y0 = np.zeros((1,), dtype=float)
            X = np.vstack([X, X0])
            y = np.concatenate([y, y0])

        self._m = m
        self._est = FDHFrontierEstimator(X=X, y=y)
        return self

    def predict(self, Xq: np.ndarray) -> np.ndarray:
        if self._est is None or self._m is None:
            raise RuntimeError("Call fit(X, y) before predict(Xq)")

        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.ndim != 2 or Xq.shape[1] != self._m:
            raise ValueError(f"Xq must be shape (G,{self._m})")

        yhat = self._est.frontier_batch(Xq)

        # 将 NaN（无支配点）替换为 fill_value，便于 runner 直接计算误差
        if np.isnan(yhat).any():
            yhat = np.nan_to_num(yhat, nan=float(self.fill_value))

        return yhat