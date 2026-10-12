# -*- coding: utf-8 -*-
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseEstimator(ABC):
    """
    统一接口：fit / predict
    - X: (n,d)
    - y: (n,) 或 (n,m)
    - predict输出形状与训练y一致（单产出返回(n,), 多产出返回(n,m)）
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError