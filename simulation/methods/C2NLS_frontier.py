# -*- coding: utf-8 -*-
#   C²NLS (Corrected Concave Nonparametric Least Squares) frontier estimator
#   for single output, multiple inputs.
#
#   This implementation mirrors the "fit / predict" runner interface used in DEA_frontier.py,
#   and provides deterministic frontier predictions y_hat(x) based on:
#       1) CNLS: least squares with global concavity + monotonicity
#       2) Correction: shift the fitted function upward by max residual so the frontier envelops data.
#
#   References / conceptual background:
#   - Kuosmanen & Johnson (2010): DEA as shape-constrained least squares; C2NLS as corrected CNLS.
#   - pyStoNED documentation (Corrected CNLS model).
#
#   Notes:
#   - This is a baseline implementation intended for research comparison.
#   - CNLS uses O(n^2) Afriat inequalities; expect it to be slow for large n.
#   - For larger problems, consider pyStoNED's CNLS-G (constraint generation) implementation.

from __future__ import annotations


import numpy as np
from pystoned import CNLS
from pystoned.constant import CET_ADDI, FUN_PROD, RTS_VRS, OPT_LOCAL, RTS_CRS


class C2NLSEstimatorWrapper:
    def __init__(self, rts: str = "vrs", add_origin: bool = True, solver: str = "mosek"):
        self.rts = rts.lower()
        self.add_origin = add_origin
        self.solver = solver
        self.model = None
        self.shift_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)  # 关键：用一维

        if self.add_origin:
            X = np.vstack([X, np.zeros((1, X.shape[1]))])
            y = np.hstack([y, np.array([0.0])])

        self.model = CNLS.CNLS(
            y=y,                       # 传一维也可以，你这份CNLS内部会校验
            x=X,
            z=None,
            cet=CET_ADDI,
            fun=FUN_PROD,
            rts=RTS_VRS if self.rts == "vrs" else RTS_CRS,
        )

        self.model.optimize(OPT_LOCAL, solver="mosek")

        # CNLS fitted frontier on training points
        y_hat = np.asarray(self.model.get_frontier(), dtype=float).reshape(-1)  # 关键：一维对齐

        # C2NLS correction shift = max residual
        resid = y - y_hat  # 现在是逐点残差，不会广播
        self.shift_ = float(np.max(resid))

        return self

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("C2NLS model is not fitted.")

        X_query = np.asarray(X_query, dtype=float)
        y_hat = np.asarray(self.model.get_predict(X_query), dtype=float).reshape(-1)

        return y_hat + self.shift_
