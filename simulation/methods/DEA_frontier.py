# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import cvxpy as cp


ReturnsToScale = Literal["vrs", "crs"]  # vrs=BCC, crs=CCR

@dataclass
class DEAEstimatorWrapper:
    """
    统一 runner 接口的 DEA 前沿估计器（单产出，多投入）。

    接口：
        fit(X, y)  保存训练样本并构建内部 DEAFrontierEstimator
        predict(Xq) 返回每个查询投入点的前沿估计 y_hat(Xq)

    关键点：
        add_origin=True 时，会给训练集额外加入一个 (0,...,0)->0 的 DMU，
        以保证在 VRS(BCC) 下对任意 Xq >= 0 都可行，从而实现“样本外”稳定预测。
    """
    rts: ReturnsToScale = "vrs"          # "vrs"=BCC，"crs"=CCR
    solver: Optional[str] = None         # 例如 "MOSEK"，或 None 让 cvxpy 自选
    add_origin: bool = True              # 是否增广原点 DMU
    eps: float = 0.0                     # 数值容忍，用于 X^T λ <= x0 + eps

    _est: Optional[DEAFrontierEstimator] = None
    _m: Optional[int] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DEAEstimatorWrapper":
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
        self._est = DEAFrontierEstimator(
            X=X,
            y=y,
            rts=self.rts,
            solver=self.solver,
            eps=self.eps,
        )
        return self

    def predict(self, Xq: np.ndarray) -> np.ndarray:
        # Note: 如果用到训练样本外的预测上会出现问题
        if self._est is None or self._m is None:
            raise RuntimeError("Call fit(X, y) before predict(Xq)")

        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim == 1:
            Xq = Xq.reshape(1, -1)
        if Xq.ndim != 2 or Xq.shape[1] != self._m:
            raise ValueError(f"Xq must be shape (G,{self._m})")

        yhat = self._est.frontier_batch(Xq)

        return yhat


@dataclass
class DEAFrontierEstimator:
    """
    Single output, multiple inputs.
    Estimate frontier value y_hat(x0) under free disposability.

    Given training data (X, y), for a query input x0, solve:

    VRS (BCC):
        max_{λ}  sum_j λ_j y_j
        s.t.     sum_j λ_j x_j <= x0
                 sum_j λ_j = 1
                 λ_j >= 0

    CRS (CCR):
        max_{λ}  sum_j λ_j y_j
        s.t.     sum_j λ_j x_j <= x0
                 λ_j >= 0
    """
    X: np.ndarray  # shape (n, m)
    y: np.ndarray  # shape (n,) or (n, 1)
    rts: ReturnsToScale = "vrs"
    solver: Optional[str] = "ECOS"  # e.g. "MOSEK", "ECOS", "GLPK"
    eps: float = 0.0  # numerical tolerance for feasibility

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
        """
        Evaluate estimated frontier value y_hat(x0).
        """
        x0 = np.asarray(x0, dtype=float).reshape(-1)
        n, m = self.X.shape
        if x0.shape[0] != m:
            raise ValueError(f"x0 must have length {m}, got {x0.shape[0]}")

        lam = cp.Variable(n, nonneg=True)

        obj = cp.Maximize(self.y @ lam)

        constraints = []
        # sum_j λ_j x_j <= x0
        constraints.append(self.X.T @ lam <= x0 + self.eps)

        # VRS adds convexity constraint
        if self.rts == "vrs":
            constraints.append(cp.sum(lam) == 1.0)
        elif self.rts == "crs":
            pass
        else:
            raise ValueError("rts must be 'vrs' or 'crs'")

        prob = cp.Problem(obj, constraints)

        # Solve
        if self.solver is None:
            prob.solve()  # let cvxpy choose
        else:
            prob.solve(solver=self.solver)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            # If infeasible, in theory should not happen if x0 >= 0 and data includes origin conditions,
            # but can happen in practice due to tight x0. Return NaN.
            return float("nan")

        return float(prob.value)

    def frontier_batch(self, Xq: np.ndarray) -> np.ndarray:
        """
        Batch evaluation. Solves one LP per query point.
        Xq shape: (G, m)
        """
        Xq = np.asarray(Xq, dtype=float)
        if Xq.ndim != 2 or Xq.shape[1] != self.X.shape[1]:
            raise ValueError(f"Xq must be shape (G,{self.X.shape[1]})")
        out = np.empty(Xq.shape[0], dtype=float)
        for i in range(Xq.shape[0]):
            out[i] = self.frontier_at(Xq[i])
        return out


def make_input_grid(
    x_min: np.ndarray,
    x_max: np.ndarray,
    num: int = 50,
) -> np.ndarray:
    """
    Simple grid generator for m inputs.
    For m=2, returns num*num points.
    For general m, returns num^m points, may be large.
    """
    x_min = np.asarray(x_min, dtype=float).reshape(-1)
    x_max = np.asarray(x_max, dtype=float).reshape(-1)
    if x_min.shape != x_max.shape:
        raise ValueError("x_min and x_max must have same shape")
    m = x_min.shape[0]
    axes = [np.linspace(x_min[k], x_max[k], num) for k in range(m)]
    mesh = np.meshgrid(*axes, indexing="xy")
    grid = np.stack([g.reshape(-1) for g in mesh], axis=1)
    return grid