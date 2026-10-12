# -*- coding: utf-8 -*-
"""
评估指标模块：支持单产出或多产出（n,) 或 (n,m)

- 单产出：y形状(n,)
- 多产出：y形状(n,m)，m为产出维度

默认口径：
- MSE：对所有元素求均值（等价于对每个产出维度等权）
- Bias：对所有元素求均值（mean(y_pred - y_ref)），同样等权

"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Union, Optional
import numpy as np


def _to_2d(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    将y规范化到二维：
    - (n,) -> (n,1)，并返回is_single=True
    - (n,m) -> 原样返回is_single=False
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y.reshape(-1, 1), True
    if y.ndim == 2:
        return y, False
    raise ValueError("y必须是形状(n,)（单产出）或(n,m)（多产出）的数组。")


def mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    return_per_output: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Mean Squared Error.
    return_per_output=False: 返回标量（对所有产出等权平均）
    return_per_output=True: 返回(总体MSE, 每个产出MSE数组shape(m,))
    """
    y_t, _ = _to_2d(y_true)
    y_p, _ = _to_2d(y_pred)

    if y_t.shape != y_p.shape:
        raise ValueError(f"y_true与y_pred形状不一致：{y_t.shape} vs {y_p.shape}")

    err2 = (y_t - y_p) ** 2  # (n,m)
    per_out = err2.mean(axis=0)  # (m,)
    overall = float(per_out.mean())

    if return_per_output:
        return overall, per_out
    return overall


def bias(
    y_ref: np.ndarray,
    y_pred: np.ndarray,
    return_per_output: bool = False
) -> Union[float, Tuple[float, np.ndarray]]:
    """
    Bias = mean(y_pred - y_ref).
    return_per_output=False: 返回标量（对所有产出等权平均）
    return_per_output=True: 返回(总体Bias, 每个产出Bias数组shape(m,))
    """
    y_r, _ = _to_2d(y_ref)
    y_p, _ = _to_2d(y_pred)

    if y_r.shape != y_p.shape:
        raise ValueError(f"y_ref与y_pred形状不一致：{y_r.shape} vs {y_p.shape}")

    diff = (y_p - y_r)  # (n,m)
    per_out = diff.mean(axis=0)  # (m,)
    overall = float(per_out.mean())

    if return_per_output:
        return overall, per_out
    return overall


def summarize_records(records: list[dict]) -> Dict[str, Any]:
    """
    对runner收集的records做简单汇总
    """
    if len(records) == 0:
        return {"n_records": 0}

    # 先不强制依赖pandas，保持轻量
    # 按 (scenario, n, method) 聚合均值
    agg: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (r["scenario"], r["n"], r["method"])
        if key not in agg:
            agg[key] = {"count": 0, "mse_sum": 0.0, "bias_sum": 0.0}
        agg[key]["count"] += 1
        agg[key]["mse_sum"] += float(r["mse"])
        agg[key]["bias_sum"] += float(r["bias"])

    summary_rows = []
    for (scenario, n, method), v in agg.items():
        c = v["count"]
        summary_rows.append({
            "scenario": scenario,
            "n": n,
            "method": method,
            "count": c,
            "mse_mean": v["mse_sum"] / c,
            "bias_mean": v["bias_sum"] / c
        })

    return {
        "n_records": len(records),
        "groups": summary_rows
    }