# -*- coding: utf-8 -*-
# - MonotoneIntegralNet1D (architecture)
# - fit_base_model_mse (stage 1)
# - fit_boosting_envelope (stage 2)
# - predict_boosted_frontier (inference)
# - MonotoneEnvelopeEstimator (runner-compatible wrapper: fit/predict)
#
# Notes:
# 1) This file is designed to live under frontier_experiments/methods/.
# 2) It supports multi-input X: shape (n,d).
# 3) It currently supports single-output y: shape (n,) or (n,1).
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .base import BaseEstimator


# ----------------------------
# Architecture components
# ----------------------------

class ELU0(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return F.elu(x, alpha=self.alpha)

class PositiveScale(nn.Module):
    """
    Monotonicity-safe rescaling: h -> h * s, with s>0 (learnable).
    No centering, no mixing across units.
    """
    def __init__(self, dim: int, init_log_scale: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.log_scale = nn.Parameter(torch.full((dim,), float(init_log_scale)))
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        s = F.softplus(self.log_scale) + self.eps  # strictly positive
        return h * s


def rho_hat(rho_dot: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return -rho_dot(-x)


def rho_tilde(rho_dot: nn.Module, x: torch.Tensor) -> torch.Tensor:
    one = torch.ones_like(x)
    rho1 = rho_dot(one)
    left = rho_dot(x + 1.0) - rho1
    right = rho_hat(rho_dot, x - 1.0) + rho1
    return torch.where(x < 0, left, right)


def apply_monotone_constraint(W_raw: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Enforce sign constraints per input dimension.
    t: shape (in_features,), values in {-1,0,1}
      1: weights >= 0
     -1: weights <= 0
      0: free
    """
    W_abs = torch.abs(W_raw)
    t = t.view(1, -1)

    pos = (t == 1).expand_as(W_raw)
    neg = (t == -1).expand_as(W_raw)
    fre = (t == 0).expand_as(W_raw)

    W_eff = torch.empty_like(W_raw)
    W_eff[pos] = W_abs[pos]
    W_eff[neg] = -W_abs[neg]
    W_eff[fre] = W_raw[fre]
    return W_eff


class CombinedActivation(nn.Module):
    """
    Apply three monotone activations derived from a common basis rho_dot:
    - rho_dot
    - rho_hat
    - rho_tilde
    """
    def __init__(self, rho_dot: nn.Module, s_tuple: Tuple[int, int, int]):
        super().__init__()
        self.rho_dot = rho_dot
        self.s_dot, self.s_hat, self.s_tilde = map(int, s_tuple)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        m = h.shape[-1]
        if self.s_dot + self.s_hat + self.s_tilde != m:
            raise ValueError("s_dot + s_hat + s_tilde must equal the last dim of h.")

        outs: List[torch.Tensor] = []
        if self.s_dot > 0:
            outs.append(self.rho_dot(h[..., :self.s_dot]))
        if self.s_hat > 0:
            outs.append(rho_hat(self.rho_dot, h[..., self.s_dot:self.s_dot + self.s_hat]))
        if self.s_tilde > 0:
            outs.append(rho_tilde(self.rho_dot, h[..., self.s_dot + self.s_hat:]))

        if len(outs) == 1:
            return outs[0]
        return torch.cat(outs, dim=-1)


class MonotoneDense(nn.Module):
    """
    Monotone constrained fully connected layer.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        t: List[int],
        s_tuple: Tuple[int, int, int],
        rho_dot: Optional[nn.Module] = None,
        bias: bool = True,
        use_pos_scale: bool = False,
    ):
        super().__init__()
        t_tensor = torch.tensor(t, dtype=torch.int64)
        if t_tensor.numel() != in_features:
            raise ValueError("Length of t must equal in_features.")
        self.register_buffer("t", t_tensor)

        self.W_raw = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.rho_dot = rho_dot if rho_dot is not None else ELU0()
        self.act = CombinedActivation(self.rho_dot, s_tuple)

        self.use_pos_scale = bool(use_pos_scale)
        self.pos_scale = PositiveScale(out_features) if self.use_pos_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_eff = apply_monotone_constraint(self.W_raw, self.t)
        h = x @ W_eff.t()
        if self.b is not None:
            h = h + self.b

        a = self.act(h)

        if self.pos_scale is not None:
            a = self.pos_scale(a)

        return a


class MonotoneIntegralNet1D(nn.Module):
    """
    Monotone network that passes through the origin by subtracting f(0).
    If enforce_nonneg=True, output is constrained to be >= 0 via softplus differences.
    """
    def __init__(self, hidden: Tuple[int, ...] = (64, 64), in_dim: int = 1, eps: float = 1e-4):
        super().__init__()
        self.in_dim = int(in_dim)
        self.eps = float(eps)
        self.enforce_nonneg = False

        t_in = [1] * self.in_dim  # non-decreasing in each input

        def make_s(m: int) -> Tuple[int, int, int]:
            # Keep your current default: all neurons use rho_tilde group
            s_dot = int(round(0.0 * m))
            s_hat = int(round(0.0 * m))
            s_tilde = m - s_dot - s_hat
            return (s_dot, s_hat, s_tilde)

        rho = ELU0()

        layers: List[nn.Module] = []
        in_d = self.in_dim
        for h in hidden:
            layers.append(
                MonotoneDense(
                    in_features=in_d,
                    out_features=int(h),
                    t=t_in if in_d == self.in_dim else [1] * in_d,
                    s_tuple=make_s(int(h)),
                    rho_dot=rho,
                    use_pos_scale=False,
                )
            )
            in_d = int(h)

        self.features = nn.Sequential(*layers)
        self.out = MonotoneDense(
            in_features=in_d,
            out_features=1,
            t=[1] * in_d,
            s_tuple=(1, 0, 0),  # simple output activation
            use_pos_scale=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_dim)

        z = self.features(x)
        y_raw = self.out(z)

        x0 = torch.zeros((1, self.in_dim), device=x.device, dtype=x.dtype)
        z0 = self.features(x0)
        y0_raw = self.out(z0)

        if self.enforce_nonneg:
            y = F.softplus(y_raw) - F.softplus(y0_raw)
        else:
            y = y_raw - y0_raw
        return y


# ----------------------------
# Training routines (stage 1 & stage 2)
# ----------------------------

def fit_base_model_mse(
    x_np: np.ndarray,
    y_np: np.ndarray,
    hidden: Tuple[int, ...] = (32, ),
    lr: float = 3e-3,
    epochs: int = 2000,
    seed: int = 0,
    verbose: bool = True,
    device: Optional[str] = None,
) -> MonotoneIntegralNet1D:
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_np = np.asarray(x_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32)
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    if y_np.shape[1] != 1:
        raise ValueError("fit_base_model_mse currently supports single-output y (n,) or (n,1).")
    if x_np.ndim != 2:
        raise ValueError("x_np must be of shape (n,d).")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    x = torch.from_numpy(x_np).to(dev)
    y = torch.from_numpy(y_np).to(dev)

    n, d = x_np.shape
    model = MonotoneIntegralNet1D(hidden=hidden, in_dim=d).to(dev)
    opt = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, int(epochs) + 1):
        opt.zero_grad()
        yhat = model(x)
        loss = ((yhat - y) ** 2).mean()
        loss.backward()
        opt.step()

        if verbose and (ep == 1 or ep == epochs or ep % 400 == 0):
            print(f"base epoch={ep:4d}  mse={loss.item():.6f}")

    model.eval()
    return model


def fit_boosting_envelope(
    x_np: np.ndarray,
    y_np: np.ndarray,
    base_model: MonotoneIntegralNet1D,
    T: int = 20,
    eta: float = 0.2,
    weak_hidden: Tuple[int, ...] = (12,),
    lr: float = 5e-2,
    epochs: int = 800,
    seed: int = 0,
    verbose: bool = True,
    rho: float = 0.5,
    alpha_clip: float = 50.0,
    reg: float = 1.0,
    tol: float = 1e-6,
    pos_rank_mode: str = "sum",
    device: Optional[str] = None,
) -> Tuple[List[MonotoneIntegralNet1D], Dict[str, Any]]:
    """
    Boosting to enforce envelope: ensure f(x_i) >= y_i (within tol).
    This is designed to "lift" the base model by adding nonnegative weak learners.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_np = np.asarray(x_np, dtype=np.float32)
    y_np = np.asarray(y_np, dtype=np.float32)
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    if y_np.shape[1] != 1:
        raise ValueError("fit_boosting_envelope currently supports single-output y (n,) or (n,1).")
    if x_np.ndim != 2:
        raise ValueError("x_np must be of shape (n,d).")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    x = torch.from_numpy(x_np).to(dev)
    y = torch.from_numpy(y_np).to(dev)

    n, d = x_np.shape

    learners: List[MonotoneIntegralNet1D] = []
    history: Dict[str, Any] = {"violations": [], "max_gap": []}

    base_model = base_model.to(dev)
    base_model.eval()

    @torch.no_grad()
    def predict_current() -> torch.Tensor:
        f = base_model(x)
        for h in learners:
            f = f + float(eta) * h(x)
        return f

    # Position weighting (optional, scalar per sample)
    if pos_rank_mode == "none":
        w_pos = torch.ones((n, 1), dtype=x.dtype, device=dev)
    else:
        if pos_rank_mode == "sum":
            key = x.sum(dim=1)
        elif pos_rank_mode == "norm":
            key = torch.norm(x, dim=1, p=2)
        else:
            raise ValueError("pos_rank_mode must be 'sum', 'norm', or 'none'.")

        rank = torch.argsort(torch.argsort(key))
        w_pos = 1.0 / (rank.float() + 1.0)
        w_pos = w_pos / w_pos.mean()
        w_pos = w_pos.detach().view(-1, 1)

    for t in range(1, int(T) + 1):
        with torch.no_grad():
            f_prev = predict_current()
            gap_prev = (y - f_prev)
            r = torch.relu(gap_prev)
            viol = int((r > tol).sum().item())
            max_gap = float(r.max().item())

        history["violations"].append(viol)
        history["max_gap"].append(max_gap)

        if verbose:
            print(f"boost iter={t:2d}  violations={viol}  max_gap={max_gap:.6f}")

        if viol == 0:
            break

        h = MonotoneIntegralNet1D(hidden=weak_hidden, in_dim=d).to(dev)
        h.enforce_nonneg = True
        opt = optim.Adam(h.parameters(), lr=lr)

        # Lagrange multiplier (sample-wise)
        alpha = torch.zeros_like(y)

        h.train()
        for _ in range(int(epochs)):
            opt.zero_grad()

            pred_h = h(x)  # >=0
            f_new = f_prev + float(eta) * pred_h
            gap = (y - f_new)
            gap_pos = torch.relu(gap)

            loss_reg = ((float(eta) * pred_h) ** 2).mean()
            loss_lag = (alpha.detach() * gap_pos).mean()
            loss = loss_reg + loss_lag

            loss.backward()
            opt.step()

            with torch.no_grad():
                alpha = alpha + float(rho) * gap_pos
                if alpha_clip is not None:
                    alpha.clamp_(0.0, float(alpha_clip))

        h.eval()
        learners.append(h)

    if verbose:
        print("final learners:", len(learners))

    return learners, history


# ----------------------------
# Inference
# ----------------------------

@torch.no_grad()
def predict_boosted_frontier(
    x_np: np.ndarray,
    base_model: MonotoneIntegralNet1D,
    learners: List[MonotoneIntegralNet1D],
    eta: float = 0.2,
    device: Optional[str] = None,
) -> np.ndarray:
    x_arr = np.asarray(x_np, dtype=np.float32)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if x_arr.ndim != 2:
        raise ValueError("x_np must be of shape (n,d) or (n,) for d=1.")

    dev = torch.device(device) if device is not None else torch.device("cpu")
    x = torch.from_numpy(x_arr).to(dev)

    base_model = base_model.to(dev)
    base_model.eval()

    f = base_model(x)
    for h in learners:
        h = h.to(dev)
        h.eval()
        f = f + float(eta) * h(x)

    return f.detach().cpu().numpy().reshape(-1).astype(np.float32)


# ----------------------------
# Estimator wrapper (runner-compatible)
# ----------------------------

@dataclass
class MonotoneEnvelopeParams:
    # Stage 1 (base)
    base_hidden: Tuple[int, ...] = (32, )
    base_lr: float = 3e-3
    base_epochs: int = 2000

    # Stage 2 (boosting envelope)
    T: int = 50
    eta: float = 0.2
    weak_hidden: Tuple[int, ...] = (12, )
    boost_lr: float = 5e-2
    boost_epochs: int = 800
    rho: float = 0.5
    alpha_clip: float = 50.0
    reg: float = 1.0
    tol: float = 1e-6
    pos_rank_mode: str = "sum"

    # Misc
    seed: int = 0
    verbose: bool = True
    device: Optional[str] = None  # "cpu" or "cuda"


class MonotoneEnvelopeEstimator(BaseEstimator):
    """
    Our method:
    - fit: trains base model, then boosting to satisfy envelope
    - predict: boosted frontier prediction
    """

    def __init__(self, params: Optional[MonotoneEnvelopeParams] = None, **kwargs):
        if params is None:
            params = MonotoneEnvelopeParams(**kwargs)
        self.params = params

        self.base_model: Optional[MonotoneIntegralNet1D] = None
        self.learners: List[MonotoneIntegralNet1D] = []
        self.history: Optional[Dict[str, Any]] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 2 and y.shape[1] != 1:
            raise ValueError("MonotoneEnvelopeEstimator currently supports single-output y (n,) or (n,1).")

        p = self.params
        self.base_model = fit_base_model_mse(
            X, y,
            hidden=p.base_hidden,
            lr=p.base_lr,
            epochs=p.base_epochs,
            seed=p.seed,
            verbose=p.verbose,
            device=p.device,
        )

        self.learners, self.history = fit_boosting_envelope(
            X, y,
            base_model=self.base_model,
            T=p.T,
            eta=p.eta,
            weak_hidden=p.weak_hidden,
            lr=p.boost_lr,
            epochs=p.boost_epochs,
            seed=p.seed,
            verbose=p.verbose,
            rho=p.rho,
            alpha_clip=p.alpha_clip,
            reg=p.reg,
            tol=p.tol,
            pos_rank_mode=p.pos_rank_mode,
            device=p.device,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.base_model is None:
            raise RuntimeError("Call fit() before predict().")

        p = self.params
        return predict_boosted_frontier(
            X,
            base_model=self.base_model,
            learners=self.learners,
            eta=p.eta,
            device=p.device,
        )