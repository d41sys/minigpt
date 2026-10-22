"""Probabilistic day-ahead price forecasts: quantile baselines, interval metrics, rolling eval.

Protocol, fixed before any test score was computed. It is the point-forecast protocol of
`price_eval` with one change of target: 12 monthly folds, per fold validation = the 28 days before
the test month, training = everything before that. Every selection (transformer early stopping,
ridge alpha, quantile-regression penalty) uses the validation window only; residual quantiles use
the training window only. No test day enters any fit, selection or calibration step. The quantile
levels (`QUANTILES`), the penalty grid and the 5-seed ensemble were set up front and never changed
after seeing a test result.

Forecasts are arrays of shape (N, 24, Q) in EUR/MWh, Q ascending quantile levels; every producer
sorts along the last axis, so no consumer sees crossing quantiles.

Methods:

* seasonal naive + residual quantiles: D-1 price plus the empirical per-hour-of-day quantiles of
  (y - D-1 price) over the training days.
* ridge + residual quantiles: the tuned point ridge of the point-forecast study plus empirical
  per-hour quantiles of its out-of-sample residuals. Training residuals of a fitted ridge are
  in-sample and too small, so they are cross-fitted: the training window is cut into 5 contiguous
  blocks and each block is predicted by a ridge (same alpha) fit on the other four. Still training
  window only.
* linear quantile regression: one L2-penalised linear model per (hour, level) on the same 168
  normalised lags + weekday one-hot as ridge, fit by minimising the pinball loss.
* transformer: `PriceTransformer` with a Q-output head trained on the pinball loss. The ensemble
  forecast averages the members' quantiles (Vincentisation), which keeps them sorted.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import torch

from minigpt.price_data import Windows, split_windows
from minigpt.price_eval import (
    ALPHA,
    diebold_mariano_hln,
    month_label,
    paired_bootstrap,
    resolution_flag,
)
from minigpt.price_model import (
    QUANTILES,
    fit_ridge,
    fit_ridge_tuned,
    lag_design,
    load_models,
    pinball_loss,
    predict_ensemble,
    seasonal_naive,
)

LOWER, UPPER = 0.1, 0.9  # the 80% prediction interval
METHODS = (
    "seasonal naive + residual quantiles",
    "ridge + residual quantiles",
    "linear quantile regression",
)
MODEL = "transformer (pinball)"
QR_PENALTIES = (1.0, 10.0, 100.0, 1000.0)  # diagonal L2 term of the IRLS normal equations
QR_ITERS = 100
QR_EPS = 1e-3  # floor of |residual| in normalised units inside the IRLS weights
CROSS_FIT_BLOCKS = 5


# --------------------------------------------------------------------------- metrics


def sort_quantiles(pred: np.ndarray) -> np.ndarray:
    """Sort along the quantile axis. Fixes crossing and never raises the pinball loss."""
    out: np.ndarray = np.sort(pred, axis=-1)
    return out


def crps_weights(quantiles: tuple[float, ...]) -> np.ndarray:
    """Riemann weights of the levels on (0, 1): each level owns the stretch nearest to it."""
    q = np.asarray(quantiles, dtype=np.float64)
    edges = np.concatenate([[0.0], (q[:-1] + q[1:]) / 2.0, [1.0]])
    out: np.ndarray = np.diff(edges)
    return out


def crps_from_quantiles(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> float:
    """CRPS approximated from quantiles: 2 * sum_q w_q * mean pinball_q (EUR/MWh).

    CRPS = 2 * integral over q in (0, 1) of the pinball loss. With a handful of levels the integral
    is a Riemann sum and the tails beyond the outer levels are extrapolated flat, so treat the
    value as a coarse score for comparing methods, not as an exact CRPS.
    """
    per_level = pinball_loss(pred, y, quantiles).mean(axis=tuple(range(pred.ndim - 1)))
    return float(2.0 * (crps_weights(quantiles) * per_level).sum())


def interval_coverage(
    pred: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...],
    lo: float = LOWER,
    hi: float = UPPER,
) -> float:
    """Share of target hours inside [q_lo, q_hi] (inclusive). Nominal is hi - lo."""
    lower, upper = pred[..., quantiles.index(lo)], pred[..., quantiles.index(hi)]
    return float(((y >= lower) & (y <= upper)).mean())


def interval_width(
    pred: np.ndarray, quantiles: tuple[float, ...], lo: float = LOWER, hi: float = UPPER
) -> float:
    """Mean width q_hi - q_lo in EUR/MWh."""
    return float((pred[..., quantiles.index(hi)] - pred[..., quantiles.index(lo)]).mean())


def per_day_pinball(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Pinball loss averaged over the 24 hours and the levels, one value per day."""
    out: np.ndarray = pinball_loss(pred, y, quantiles).mean(axis=(1, 2))
    return out


def score(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> dict[str, Any]:
    pin = pinball_loss(pred, y, quantiles)
    return {
        "mean_pinball": float(pin.mean()),
        "pinball_by_level": [float(v) for v in pin.mean(axis=(0, 1))],
        "crps_approx": crps_from_quantiles(pred, y, quantiles),
        "coverage_80": interval_coverage(pred, y, quantiles),
        "width_80": interval_width(pred, quantiles),
        "median_mae": float(np.abs(pred[..., quantiles.index(0.5)] - y).mean()),
    }


# --------------------------------------------------------------------------- baselines


def residual_offsets(resid: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    """Empirical per-hour-of-day quantiles of residuals (n_days, 24) -> (24, Q)."""
    out: np.ndarray = np.quantile(resid, quantiles, axis=0).T
    return out


def add_offsets(point: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Point forecast (N, 24) + per-hour residual quantiles (24, Q) -> sorted (N, 24, Q)."""
    return sort_quantiles(point[..., None] + offsets[None])


def seasonal_naive_quantiles(
    train: Windows, test: Windows, quantiles: tuple[float, ...]
) -> np.ndarray:
    """D-1 forecast with residual quantiles from the training days (no fit, so out-of-sample)."""
    offsets = residual_offsets(train.y - seasonal_naive(train), quantiles)
    return add_offsets(seasonal_naive(test), offsets)


def cross_fit_ridge_residuals(
    train: Windows, alpha: float, n_blocks: int = CROSS_FIT_BLOCKS
) -> np.ndarray:
    """Out-of-sample ridge residuals on the training window via contiguous-block cross-fitting."""
    n = len(train)
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    resid = np.empty_like(train.y)
    for a, b in itertools.pairwise(edges):
        held = np.zeros(n, dtype=bool)
        held[a:b] = True
        model = fit_ridge(train.subset(~held), alpha)
        resid[a:b] = train.y[a:b] - model.predict(train.subset(held))
    return resid


def ridge_quantiles(
    train: Windows, val: Windows, targets: Sequence[Windows], quantiles: tuple[float, ...]
) -> tuple[list[np.ndarray], float]:
    """Tuned ridge point forecast + cross-fitted training-window residual quantiles.

    One fit; returns a (N, 24, Q) forecast for each of `targets`, and the chosen alpha.
    """
    ridge = fit_ridge_tuned(train, val)
    offsets = residual_offsets(cross_fit_ridge_residuals(train, ridge.alpha), quantiles)
    return [add_offsets(ridge.predict(w), offsets) for w in targets], ridge.alpha


def _irls_quantile(
    x1: np.ndarray, z: np.ndarray, q: float, penalty: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Minimise sum pinball_q(z - x1 @ b) + L2 by majorise-minimise (Hunter & Lange, 2000).

    |e| <= e^2 / (2 |e_k|) + |e_k| / 2 turns each step into a weighted ridge solve with weights
    1 / (2 max(|e_k|, eps)); pinball_q(e) = |e| / 2 + (q - 1/2) e.
    """
    for _ in range(QR_ITERS):
        e = z - x1 @ b
        w = 0.5 / np.maximum(np.abs(e), QR_EPS)
        xw = x1 * w[:, None]
        rhs = xw.T @ z + (q - 0.5) * x1.sum(axis=0)
        new = np.linalg.solve(x1.T @ xw + penalty, rhs)
        done = float(np.abs(new - b).max()) < 1e-7
        b = new
        if done:
            break
    return b


@dataclass
class QuantileRegression:
    coef: np.ndarray  # (24, Q, n_features) in normalised units
    intercept: np.ndarray  # (24, Q)
    penalty: float

    def predict(self, w: Windows) -> np.ndarray:
        feats, mu, sd = lag_design(w)
        z = np.einsum("np,hqp->nhq", feats, self.coef) + self.intercept[None]
        return sort_quantiles(z * sd[..., None] + mu[..., None])


def fit_quantile_regression(
    w: Windows, quantiles: tuple[float, ...], penalty: float
) -> QuantileRegression:
    """One penalised linear pinball model per (hour, level) on normalised lags + weekday."""
    feats, mu, sd = lag_design(w)
    z_all = (w.y - mu) / sd
    n, p = feats.shape
    x1 = np.hstack([feats, np.ones((n, 1))])
    pen = penalty * np.eye(p + 1)
    pen[-1, -1] = 0.0  # intercept is free
    start = np.linalg.solve(x1.T @ x1 + pen, x1.T @ z_all)  # ridge start, (p+1, 24)
    coef = np.empty((z_all.shape[1], len(quantiles), p))
    intercept = np.empty((z_all.shape[1], len(quantiles)))
    for h in range(z_all.shape[1]):
        for j, q in enumerate(quantiles):
            b = _irls_quantile(x1, z_all[:, h], q, pen, start[:, h])
            coef[h, j], intercept[h, j] = b[:-1], b[-1]
    return QuantileRegression(coef=coef, intercept=intercept, penalty=penalty)


def fit_quantile_regression_tuned(
    train: Windows, val: Windows, quantiles: tuple[float, ...]
) -> QuantileRegression:
    """Pick the penalty by validation mean pinball; fit on train only."""
    best: QuantileRegression | None = None
    best_loss = np.inf
    for penalty in QR_PENALTIES:
        model = fit_quantile_regression(train, quantiles, penalty)
        loss = float(pinball_loss(model.predict(val), val.y, quantiles).mean())
        if loss < best_loss:
            best, best_loss = model, loss
    assert best is not None
    return best


# --------------------------------------------------------------------------- folds


@dataclass
class QuantileFoldResult:
    month: tuple[int, int]
    test: Windows
    preds: dict[str, np.ndarray]  # method -> (N, 24, Q)
    member_preds: list[np.ndarray]
    val: Windows  # the validation window: calibration data for `price_conformal`
    val_preds: dict[str, np.ndarray]  # method -> (N_val, 24, Q), same models as `preds`
    val_member_preds: list[np.ndarray]
    ridge_alpha: float
    qr_penalty: float
    resolution: str


def evaluate_quantile_fold(
    windows: Windows,
    month: tuple[int, int],
    ckpt: dict[str, Any],
    device: torch.device,
    cutover: date | None,
) -> QuantileFoldResult:
    """Score the fold's quantile transformer checkpoint and the quantile baselines."""
    quantiles = tuple(float(q) for q in ckpt["quantiles"])
    split = split_windows(windows, month, int(ckpt["val_days"]))
    test = split.test
    if len(test) == 0:
        raise ValueError(f"no complete test days in {month_label(month)}")
    (ridge_test, ridge_val), ridge_alpha = ridge_quantiles(
        split.train, split.val, (test, split.val), quantiles
    )
    qr = fit_quantile_regression_tuned(split.train, split.val, quantiles)
    models = load_models(ckpt, device)
    ens, members = predict_ensemble(models, test, device)
    ens_val, members_val = predict_ensemble(models, split.val, device)
    return QuantileFoldResult(
        month=month,
        test=test,
        preds={
            METHODS[0]: seasonal_naive_quantiles(split.train, test, quantiles),
            METHODS[1]: ridge_test,
            METHODS[2]: qr.predict(test),
            MODEL: ens,
        },
        member_preds=list(members),
        val=split.val,
        val_preds={
            METHODS[0]: seasonal_naive_quantiles(split.train, split.val, quantiles),
            METHODS[1]: ridge_val,
            METHODS[2]: qr.predict(split.val),
            MODEL: ens_val,
        },
        val_member_preds=list(members_val),
        ridge_alpha=ridge_alpha,
        qr_penalty=qr.penalty,
        resolution=resolution_flag(month, cutover),
    )


def paired_versus(
    pred: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...],
    n_boot: int,
) -> dict[str, Any]:
    diff = per_day_pinball(pred, y, quantiles) - per_day_pinball(base, y, quantiles)
    mean, lo, hi = paired_bootstrap(diff, n_boot=n_boot)
    stat, p = diebold_mariano_hln(diff, h=1)
    return {"diff_mean": mean, "ci95_lo": lo, "ci95_hi": hi, "dm_statistic": stat, "dm_p": p}


def paired_verdict(v: dict[str, Any]) -> str:
    significant = v["dm_p"] < ALPHA and (v["ci95_lo"] > 0 or v["ci95_hi"] < 0)
    if not significant:
        return "ties"
    return "beats" if v["diff_mean"] < 0 else "loses to"


def summarize_quantile(
    folds: list[QuantileFoldResult], quantiles: tuple[float, ...] = QUANTILES, n_boot: int = 10_000
) -> dict[str, Any]:
    """Pooled scores, per-fold pinball, and ensemble / single-seed tests vs the best baseline."""
    y = np.concatenate([f.test.y for f in folds])
    pooled_preds = {n: np.concatenate([f.preds[n] for f in folds]) for n in folds[0].preds}
    pooled = {n: score(p, y, quantiles) for n, p in pooled_preds.items()}
    best = min(METHODS, key=lambda n: pooled[n]["mean_pinball"])
    base = pooled_preds[best]
    ens = paired_versus(pooled_preds[MODEL], base, y, quantiles, n_boot)
    ens["verdict"] = paired_verdict(ens)
    per_fold: list[dict[str, Any]] = []
    for f in folds:
        per_fold.append(
            {
                "month": month_label(f.month),
                "days": len(f.test),
                "resolution": f.resolution,
                "ridge_alpha": f.ridge_alpha,
                "qr_penalty": f.qr_penalty,
                "mean_pinball": {
                    n: float(pinball_loss(p, f.test.y, quantiles).mean())
                    for n, p in f.preds.items()
                },
                "coverage_80": {
                    n: interval_coverage(p, f.test.y, quantiles) for n, p in f.preds.items()
                },
            }
        )
    single_seed: list[dict[str, Any]] = []
    for k in range(len(folds[0].member_preds)):
        member = np.concatenate([f.member_preds[k] for f in folds])
        row: dict[str, Any] = {"seed": k, **score(member, y, quantiles)}
        row.update(paired_versus(member, base, y, quantiles, n_boot))
        row["verdict"] = paired_verdict(row)
        single_seed.append(row)
    return {
        "folds": len(folds),
        "days": int(y.shape[0]),
        "hours": int(y.size),
        "quantiles": list(quantiles),
        "pooled": pooled,
        "best_baseline": best,
        "ensemble_vs_best_baseline": ens,
        "folds_model_better": int(
            sum(r["mean_pinball"][MODEL] < r["mean_pinball"][best] for r in per_fold)
        ),
        "per_fold": per_fold,
        "single_seed_vs_best_baseline": single_seed,
    }


def format_quantile_summary(s: dict[str, Any]) -> str:
    """Markdown: per-fold mean pinball, pooled scores, and the tests vs the best baseline."""
    names = [*METHODS, MODEL]
    lines = ["| Fold | Days | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for r in s["per_fold"]:
        cells = " | ".join(f"{r['mean_pinball'][n]:.2f}" for n in names)
        lines.append(f"| {r['month']} | {r['days']} | {cells} |")
    nominal = (UPPER - LOWER) * 100
    lines += [
        "",
        f"Pooled over {s['folds']} folds, {s['days']} days, {s['hours']} hours "
        f"(levels {', '.join(f'{q:g}' for q in s['quantiles'])}; EUR/MWh):",
        "",
        f"| Method | Mean pinball | CRPS approx. | {nominal:.0f}% coverage | "
        f"{nominal:.0f}% width | Median MAE |",
        "|---|---|---|---|---|---|",
    ]
    for n in names:
        p = s["pooled"][n]
        lines.append(
            f"| {n} | {p['mean_pinball']:.2f} | {p['crps_approx']:.2f} | "
            f"{100 * p['coverage_80']:.1f}% | {p['width_80']:.1f} | {p['median_mae']:.2f} |"
        )
    e = s["ensemble_vs_best_baseline"]
    lines += [
        "",
        f"Best baseline (pooled mean pinball): {s['best_baseline']}. Per-day mean pinball, "
        f"5-seed transformer ensemble minus baseline: {e['diff_mean']:+.2f} EUR/MWh, 95% "
        f"paired-bootstrap CI [{e['ci95_lo']:+.2f}, {e['ci95_hi']:+.2f}] over {s['days']} days; "
        f"DM-HLN statistic {e['dm_statistic']:+.2f}, p = {e['dm_p']:.3g}; the ensemble has the "
        f"lower mean pinball in {s['folds_model_better']}/{s['folds']} folds.",
        "",
        f"Verdict: the transformer ensemble {e['verdict']} the best baseline "
        f"({s['best_baseline']}).",
        "",
        f"Each ensemble member alone vs {s['best_baseline']} (per-day mean pinball difference):",
        "",
        "| Seed | Mean pinball | CRPS approx. | Coverage | Width | Diff vs baseline | 95% CI "
        "| DM-HLN p | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s["single_seed_vs_best_baseline"]:
        lines.append(
            f"| {r['seed']} | {r['mean_pinball']:.2f} | {r['crps_approx']:.2f} | "
            f"{100 * r['coverage_80']:.1f}% | {r['width_80']:.1f} | {r['diff_mean']:+.2f} | "
            f"[{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}] | {r['dm_p']:.3g} | {r['verdict']} |"
        )
    return "\n".join(lines)
