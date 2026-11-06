"""Day-ahead price forecaster built from minigpt's transformer blocks, plus the ridge baseline.

Reuse, stated plainly. Reused unchanged from `minigpt.model`: `TransformerBlock` (pre-LN causal
multi-head attention + GELU MLP), `GPTConfig` (as a hyperparameter container), and the GPT-style
weight init. Not reused: the token embedding, weight-tied LM head, and `GPTModel` itself, because
there is no vocabulary. They are replaced by a linear projection of continuous features and a
linear regression head. The attention stays causal.

Sequence layout (T = 192): 168 history tokens (hours of D-7 ... D-1) followed by 24 query tokens
(hours of D). Causality means a query token sees all history and earlier query tokens. Query
tokens carry no price (the price channel is zero and `is_query` = 1), only calendar features, so
no target information can reach the prediction. The head reads the 24 query positions.

Per-window normalisation: prices are standardised with the mean and std of the 168 history hours
and the output is mapped back (`pred = out * sd + mu`). This removes the price level, which
drifts a lot between months. The loss is L1 in EUR/MWh (the reported metric is MAE).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import torch
import torch.nn as nn

from minigpt.model import GPTConfig, TransformerBlock
from minigpt.price_data import HOURS, LOOKBACK_DAYS, Windows

SD_FLOOR = 1.0  # EUR/MWh; keeps flat history windows from blowing up the scaling
N_FEATURES = 6  # price, is_query, sin/cos hour of day, sin/cos day of week
QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)  # levels of the probabilistic forecast; 0.1-0.9 = 80% PI


def window_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-window mean and std (floored) of the history, shape (N, 1) each."""
    mu = x.mean(axis=1, keepdims=True)
    sd = np.maximum(x.std(axis=1, keepdims=True), SD_FLOOR)
    return mu, sd


def target_weekdays(dates: list[date]) -> np.ndarray:
    return np.asarray([d.weekday() for d in dates], dtype=np.int64)


def build_features(x: np.ndarray, dates: list[date], lookback: int = LOOKBACK_DAYS) -> torch.Tensor:
    """(N, 24 * (lookback + 1), N_FEATURES) model input from history only."""
    n = x.shape[0]
    t_hist = HOURS * lookback
    t = t_hist + HOURS
    mu, sd = window_stats(x)
    feats = np.zeros((n, t, N_FEATURES), dtype=np.float32)
    feats[:, :t_hist, 0] = (x - mu) / sd
    feats[:, t_hist:, 1] = 1.0
    pos = np.arange(t)
    hour_angle = 2 * math.pi * (pos % HOURS) / HOURS
    feats[:, :, 2] = np.sin(hour_angle)
    feats[:, :, 3] = np.cos(hour_angle)
    day_offset = pos // HOURS - lookback  # -lookback ... -1 history days, 0 = target day
    dow = (target_weekdays(dates)[:, None] + day_offset[None, :]) % 7
    dow_angle = 2 * math.pi * dow / 7
    feats[:, :, 4] = np.sin(dow_angle)
    feats[:, :, 5] = np.cos(dow_angle)
    return torch.from_numpy(feats)


@dataclass
class PriceConfig:
    lookback: int = LOOKBACK_DAYS
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    n_quantiles: int = 1  # 1 = point forecast head; >1 = one output per quantile level

    @property
    def context(self) -> int:
        return HOURS * (self.lookback + 1)


class PriceTransformer(nn.Module):
    """GPT blocks over continuous hourly features, regression head on the 24 query positions."""

    def __init__(self, cfg: PriceConfig) -> None:
        super().__init__()
        self.cfg = cfg
        gpt_cfg = GPTConfig(
            vocab_size=1,
            context_length=cfg.context,
            d_model=cfg.d_model,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.inp = nn.Linear(N_FEATURES, cfg.d_model)
        self.wpe = nn.Embedding(cfg.context, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(gpt_cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_quantiles)
        for module in self.modules():  # same init as GPTModel
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats (B, T, N_FEATURES) -> normalised forecast, (B, 24) or (B, 24, n_quantiles)."""
        pos = torch.arange(feats.size(1), device=feats.device)
        h = self.drop(self.inp(feats) + self.wpe(pos))
        for block in self.blocks:
            h = block(h)
        h = self.ln_f(h)
        out: torch.Tensor = self.head(h[:, -HOURS:])
        if self.cfg.n_quantiles == 1:
            out = out.squeeze(-1)
        return out


@dataclass
class TrainConfig:
    epochs: int = 80
    patience: int = 15
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.05
    seeds: int = 5


def _denorm(out: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    if out.ndim == 3:  # quantile heads: broadcast the per-window stats over the quantile axis
        mu, sd = mu.unsqueeze(-1), sd.unsqueeze(-1)
    return out * sd + mu


def pinball_loss(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Pinball (quantile) loss per element, shape of `pred` (..., Q) for `y` (...,).

    With error e = y - pred_q the loss is q * e if e >= 0 and (q - 1) * e otherwise, i.e.
    max(q * e, (q - 1) * e). Its minimiser is the q-quantile of y.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    err = y[..., None] - pred
    out: np.ndarray = np.maximum(q * err, (q - 1.0) * err)
    return out


def _pinball_torch(pred: torch.Tensor, y: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    err = y.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err)


@torch.no_grad()
def predict_one(
    model: PriceTransformer, feats: torch.Tensor, x: np.ndarray, device: torch.device
) -> np.ndarray:
    """Forecast in EUR/MWh, shape (N, 24); (N, 24, Q) for a quantile model.

    Quantile outputs are sorted along the last axis, so crossing quantiles never reach the caller
    (sorting is the standard fix and never raises the pinball loss).
    """
    model.eval()
    mu, sd = window_stats(x)
    out = model(feats.to(device)).cpu().double().numpy()
    if out.ndim == 3:
        result: np.ndarray = np.sort(out * sd[..., None] + mu[..., None], axis=-1)
        return result
    result = out * sd + mu
    return result


def train_one(
    train: Windows,
    val: Windows,
    cfg: PriceConfig,
    tcfg: TrainConfig,
    seed: int,
    device: torch.device,
    quantiles: tuple[float, ...] | None = None,
) -> tuple[dict[str, torch.Tensor], float, int]:
    """Train one model; keep the epoch with the lowest validation score.

    Point model (`quantiles` None): L1 loss, score = validation MAE in EUR/MWh. Quantile model:
    pinball loss averaged over the levels, score = validation mean pinball in EUR/MWh. The
    validation window is the 28 days before the test month, as for the point model.
    Returns (best state_dict on CPU, best val score, best epoch).
    """
    if (quantiles is None) != (cfg.n_quantiles == 1):
        raise ValueError("quantiles must be given exactly when cfg.n_quantiles > 1")
    if quantiles is not None and len(quantiles) != cfg.n_quantiles:
        raise ValueError(f"{len(quantiles)} quantile levels but n_quantiles={cfg.n_quantiles}")
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = PriceTransformer(cfg).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": tcfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=tcfg.lr,
    )
    f_train = build_features(train.x, train.dates, cfg.lookback)
    f_val = build_features(val.x, val.dates, cfg.lookback)
    mu, sd = window_stats(train.x)
    mu_t = torch.from_numpy(mu).float().to(device)
    sd_t = torch.from_numpy(sd).float().to(device)
    y_t = torch.from_numpy(train.y).float().to(device)
    scale = float(np.std(train.x))  # global EUR/MWh scale so the loss is O(1)
    q_t = None if quantiles is None else torch.tensor(quantiles, dtype=torch.float32, device=device)
    f_train = f_train.to(device)
    n = len(train)
    steps_per_epoch = math.ceil(n / tcfg.batch_size)
    total = tcfg.epochs * steps_per_epoch
    step = 0
    best_mae, best_epoch = math.inf, 0
    best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    for epoch in range(tcfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for i in range(0, n, tcfg.batch_size):
            idx = perm[i : i + tcfg.batch_size]
            lr_mult = 0.5 * (1 + math.cos(math.pi * step / total))
            for g in opt.param_groups:
                g["lr"] = tcfg.lr * max(lr_mult, 0.05)
            pred = _denorm(model(f_train[idx]), mu_t[idx], sd_t[idx])
            if q_t is None:
                loss = (pred - y_t[idx]).abs().mean() / scale
            else:
                loss = _pinball_torch(pred, y_t[idx], q_t).mean() / scale
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        val_pred = predict_one(model, f_val, val.x, device)
        if quantiles is None:
            val_mae = float(np.abs(val_pred - val.y).mean())
        else:
            val_mae = float(pinball_loss(val_pred, val.y, quantiles).mean())
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= tcfg.patience:
            break
    return best_state, best_mae, best_epoch


def train_ensemble(
    train: Windows,
    val: Windows,
    cfg: PriceConfig,
    tcfg: TrainConfig,
    device: torch.device,
    quantiles: tuple[float, ...] | None = None,
) -> dict[str, object]:
    """Train `tcfg.seeds` independent models; return a `torch.load(weights_only=True)`-safe dict."""
    states: list[dict[str, torch.Tensor]] = []
    val_scores: list[float] = []
    epochs: list[int] = []
    unit = "MAE" if quantiles is None else "pinball"
    for seed in range(tcfg.seeds):
        t0 = time.time()
        state, score, epoch = train_one(train, val, cfg, tcfg, seed, device, quantiles)
        states.append(state)
        val_scores.append(score)
        epochs.append(epoch)
        secs = time.time() - t0
        print(f"seed {seed}: best val {unit} {score:.2f} EUR/MWh at epoch {epoch} ({secs:.0f}s)")
    ckpt: dict[str, object] = {
        "config": asdict(cfg),
        "train_config": asdict(tcfg),
        "states": states,
        "val_mae" if quantiles is None else "val_pinball": val_scores,
        "best_epoch": epochs,
    }
    if quantiles is not None:
        ckpt["quantiles"] = list(quantiles)
    return ckpt


def load_models(ckpt: dict[str, object], device: torch.device) -> list[PriceTransformer]:
    cfg = PriceConfig(**ckpt["config"])  # type: ignore[arg-type]
    models: list[PriceTransformer] = []
    for state in ckpt["states"]:  # type: ignore[attr-defined]
        m = PriceTransformer(cfg).to(device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
    return models


def predict_ensemble(
    models: list[PriceTransformer], w: Windows, device: torch.device
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Mean forecast over the ensemble, plus each member's forecast."""
    feats = build_features(w.x, w.dates, models[0].cfg.lookback)
    members = [predict_one(m, feats, w.x, device) for m in models]
    return np.mean(members, axis=0), members


# --------------------------------------------------------------------------- baselines


def seasonal_naive(w: Windows) -> np.ndarray:
    """Same hour yesterday (D-1)."""
    out: np.ndarray = w.x[:, -HOURS:].copy()
    return out


def weekly_naive(w: Windows) -> np.ndarray:
    """Same hour one week ago (D-7)."""
    out: np.ndarray = w.x[:, :HOURS].copy()
    return out


def lag_design(w: Windows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalised lags + one-hot target weekday; returns (features, mu, sd)."""
    mu, sd = window_stats(w.x)
    onehot = np.eye(7)[target_weekdays(w.dates)]
    return np.hstack([(w.x - mu) / sd, onehot]), mu, sd


@dataclass
class RidgeModel:
    coef: np.ndarray
    intercept: np.ndarray
    alpha: float

    def predict(self, w: Windows) -> np.ndarray:
        feats, mu, sd = lag_design(w)
        out: np.ndarray = (feats @ self.coef + self.intercept) * sd + mu
        return out


def fit_ridge(w: Windows, alpha: float) -> RidgeModel:
    """Multi-output ridge on the same normalised 168 lags (+ weekday one-hot) as the transformer."""
    feats, mu, sd = lag_design(w)
    y = (w.y - mu) / sd
    fm, ym = feats.mean(axis=0), y.mean(axis=0)
    fc, yc = feats - fm, y - ym
    gram = fc.T @ fc + alpha * np.eye(fc.shape[1])
    coef = np.linalg.solve(gram, fc.T @ yc)
    return RidgeModel(coef=coef, intercept=ym - fm @ coef, alpha=alpha)


@dataclass
class BaggedRidge:
    """Mean forecast of ridge members fitted on bootstrap resamples of the training rows."""

    members: list[RidgeModel]

    def predict(self, w: Windows) -> np.ndarray:
        out: np.ndarray = np.mean([m.predict(w) for m in self.members], axis=0)
        return out


def fit_bagged_ridge(w: Windows, alpha: float, n_members: int = 5) -> BaggedRidge:
    """Bag `fit_ridge`: member k trains on `len(w)` rows drawn with replacement (seed k).

    All members share `alpha`; nothing is re-tuned per member.
    """
    members = []
    for k in range(n_members):
        idx = np.random.default_rng(k).integers(0, len(w), size=len(w))
        members.append(fit_ridge(Windows([w.dates[i] for i in idx], w.x[idx], w.y[idx]), alpha))
    return BaggedRidge(members)


RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def fit_ridge_tuned(train: Windows, val: Windows) -> RidgeModel:
    """Pick alpha by validation MAE (EUR/MWh); fit on train only."""
    best: RidgeModel | None = None
    best_mae = math.inf
    for alpha in RIDGE_ALPHAS:
        model = fit_ridge(train, alpha)
        mae = float(np.abs(model.predict(val) - val.y).mean())
        if mae < best_mae:
            best, best_mae = model, mae
    assert best is not None
    return best
