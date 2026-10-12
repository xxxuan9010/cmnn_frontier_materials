# -*- coding: utf-8 -*-
"""
数据生成模块：
1）按情景生成投入X，默认每个投入独立采样自Uni(1,10)
2）按情景函数计算无效率前沿产出y_frontier
3）加入无效率项u得到观测产出y_obs

说明：
ineff_mode="div"  表示 y_obs = y_frontier / (1+u)
ineff_mode="sub"  表示 y_obs = y_frontier - u
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    input_dim: int
    frontier_fn: Callable[[np.ndarray], np.ndarray]  # 输入X(n,d)，输出y_frontier(n,)


def _frontier_A(X: np.ndarray) -> np.ndarray:
    """
    Scenario A (Concave, single input, single output)
    y = 3 * log(1 + x1)
    """
    x1 = X[:, 0]
    # return np.power(x1, 0.5)
    return 3 * np.log(1 + x1)


def _frontier_B(X: np.ndarray) -> np.ndarray:
    """
    Scenario B (Concave, multiple inputs, single output)
    y = 0.3 x1 + 0.3 x2 + 0.01 (x1 x2)^0.5
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    return 0.3 * x1 + 0.3 * x2 + 0.01 * (x1 * x2) ** 0.5


def _frontier_C(X: np.ndarray) -> np.ndarray:
    """
    Scenario C (Concave, multiple inputs, single output)
    y = 0.3 x1 + 0.3 x2 + 0.3 x3 + 0.01 (x1 x2 x3)^(1/3)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    return 0.3 * x1 + 0.3 * x2 + 0.3 * x3 + 0.01 * np.cbrt(x1 * x2 * x3)


def _frontier_D(X: np.ndarray) -> np.ndarray:
    """
    Scenario D (S-shape, single input, single output)
    y = 3 x1^4 / (x1^4 + 50)
    """
    x1 = X[:, 0]
    num = 3.0 * (x1 ** 4)
    den = (x1 ** 4) + 50
    return num / den



def _frontier_E(X: np.ndarray) -> np.ndarray:
    """
    Scenario E (S-shape, multiple inputs, single output)
    y = 4(x1 + x2)^3 / ((x1 + x2)^3 + 120)
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    a = x1 + x2
    b = x1 + x2
    num = 4.0 * (a ** 3)
    den = (b ** 3) + 120.0
    return num / den


def _frontier_F(X: np.ndarray) -> np.ndarray:
    """
    Scenario F (S-shape, multiple inputs, single output)
    y = 4A^2 / (A^2 + 20), A = 0.3x1 + 0.3x2 + 0.3x3 + 0.01x1x2x3
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    x3 = X[:, 2]
    a = 0.3 * x1 + 0.3 * x2 + 0.3 * x3 + 0.01 * x1 * x2 * x3
    num = 4.0 * (a ** 2)
    den = (a ** 2) + 20.0
    return num / den




def get_default_scenarios() -> Dict[str, ScenarioSpec]:
    return {
        "A": ScenarioSpec(name="A", input_dim=1, frontier_fn=_frontier_A),
        "B": ScenarioSpec(name="B", input_dim=2, frontier_fn=_frontier_B),
        "C": ScenarioSpec(name="C", input_dim=3, frontier_fn=_frontier_C),
        "D": ScenarioSpec(name="D", input_dim=1, frontier_fn=_frontier_D),
        "E": ScenarioSpec(name="E", input_dim=2, frontier_fn=_frontier_E),
        "F": ScenarioSpec(name="F", input_dim=3, frontier_fn=_frontier_F),
        # "G": ScenarioSpec(name="G", input_dim=2, frontier_fn=_frontier_G),
        # "H": ScenarioSpec(name="H", input_dim=2, frontier_fn=None),
    }


def sample_inputs(
    n: int,
    d: int,
    low: float,
    high: float,
    seed: int = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=(n, d)).astype(np.float32)
    return X


def sample_inefficiency_normal(
    n: int,
    mean: float = 0.0,
    var: float = 0.4,
    seed: int = 0
) -> np.ndarray:
    """
    u ~ N(mean, var)，返回shape(n,)
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(loc=mean, scale=float(np.sqrt(var)), size=(n,)).astype(np.float32)
    return u


def apply_inefficiency(
    y_frontier: np.ndarray,
    u: np.ndarray,
    ineff_mode: str = "div"
) -> np.ndarray:
    """
    ineff_mode:
    div: y_obs = y_frontier/(1+u)
    sub: y_obs = y_frontier-u
    """
    y_frontier = np.asarray(y_frontier, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32).reshape(-1)  # (n,)
    assert y_frontier.shape == u.shape

    # 仅允许 (n,) 或 (n,m) 这两种表示（单产出或多产出）
    if y_frontier.ndim == 1:
        y2 = y_frontier.reshape(-1, 1)               # (n,1)
        is_single_output = True
    elif y_frontier.ndim == 2:
        y2 = y_frontier                               # (n,m)
        is_single_output = False
    else:
        raise ValueError("y_frontier 必须是形状(n,)（单产出）或(n,m)（多产出）的数组。")

    if y2.shape[0] != u.shape[0]:
        raise ValueError("y_frontier 与 u 的样本数 n 不一致。")

    u2 = u.reshape(-1, 1)  # (n,1)，将广播到(n,m)

    if ineff_mode == "div":
        u2 = np.abs(u2)
        y_obs_2d = y2 / (1.0 + u2)
        y_obs_2d = np.maximum(0.0, y_obs_2d)
    elif ineff_mode == "sub":
        u2 = np.abs(u2)
        y_obs_2d = y2 - u2
        y_obs_2d = np.maximum(0.0, y_obs_2d)
    else:
        raise ValueError("ineff_mode必须是'div'或'sub'。")

    return y_obs_2d.reshape(-1).astype(np.float32) if is_single_output else y_obs_2d.astype(np.float32)


def generate_dataset(
    scenario: str,
    n: int,
    x_low: float = 1.0,
    x_high: float = 10.0,
    u_mean: float = 0.0,
    u_var: float = 0.4,
    ineff_mode: str = "div",
    seed: int = 0,
    scenarios: Optional[Dict[str, ScenarioSpec]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
    X: (n,d)
    y_obs: (n,) 或 (n,m)
    y_frontier: (n,) 或 (n,m)
    u: (n,)
    """
    scenarios = get_default_scenarios() if scenarios is None else scenarios
    if scenario not in scenarios:
        raise ValueError(f"未知scenario={scenario}，请先在get_default_scenarios中定义。")

    spec = scenarios[scenario]
    X = sample_inputs(n=n, d=spec.input_dim, low=x_low, high=x_high, seed=seed)

    y_frontier = np.asarray(spec.frontier_fn(X), dtype=np.float32)
    if y_frontier.ndim == 2 and y_frontier.shape[0] != n:
        raise ValueError("frontier_fn 返回二维y时，第一维必须是样本数n。")

    u = sample_inefficiency_normal(n=n, mean=u_mean, var=u_var, seed=seed + 1)
    y_obs = apply_inefficiency(y_frontier=y_frontier, u=u, ineff_mode=ineff_mode)

    return X, y_obs, y_frontier, u

if __name__ == "__main__":
    for sc in ["A", "B", "C"]:
        X, y, yf, u = generate_dataset(sc, n=5, seed=20260101, ineff_mode="div")
        print(sc, X.shape, y.shape, yf.shape, u.shape)
