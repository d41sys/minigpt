"""Metrics, significance tests, and rolling-origin evaluation of day-ahead price forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import torch

from minigpt.price_data import Windows, month_bounds, split_windows
from minigpt.price_model import (
    fit_bagged_ridge,
    fit_ridge_tuned,
    load_models,
    predict_ensemble,
    seasonal_naive,
    weekly_naive,
)

RIDGE = "ridge on lags"
BAGGED = "bagged ridge (5)"
BASELINES = ("seasonal naive (D-1)", "weekly naive (D-7)", RIDGE, BAGGED)
RIDGE_BASELINES = (RIDGE, BAGGED)  # the verdict must hold against both
MODEL = "transformer"
NEAR_ZERO = 5.0  # EUR/MWh: |price| below this makes sMAPE's denominator tiny
ALPHA = 0.05
BLOCK_DAYS = 7  # moving-block length used by the verdict (weekly cycle); see docs/protocol-*.md
SENSITIVITY_BLOCKS = (3, 14)  # post-hoc only; never enters the verdict


# --------------------------------------------------------------------------- metrics


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(((pred - y) ** 2).mean()))


def smape(pred: np.ndarray, y: np.ndarray) -> float:
    """Symmetric MAPE in percent, 200 * |f - y| / (|f| + |y|).

    Pairs with |f| + |y| = 0 are dropped. The measure is unreliable when prices are near zero or
    negative: the denominator shrinks (or the signs differ) and a small EUR error becomes a
    huge percentage, bounded only by 200.
    """
    denom = np.abs(pred) + np.abs(y)
    ok = denom > 0
    return float((200.0 * np.abs(pred - y)[ok] / denom[ok]).mean())


def per_day_mae(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.abs(pred - y).mean(axis=1)
    return out


# --------------------------------------------------------------------------- significance


def paired_bootstrap(
    diff: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of per-day differences and its 95% percentile bootstrap CI.

    Resamples days i.i.d. with replacement. Adjacent days are autocorrelated, so this CI is
    probably somewhat too narrow.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def block_bootstrap(
    diff: np.ndarray, block: int, n_boot: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of per-day differences and its 95% percentile moving-block bootstrap CI.

    Overlapping blocks of `block` consecutive days (start uniform on 0..n-block) are drawn with
    replacement, concatenated and cut to n days, which keeps dependence up to `block` days
    inside a resample. `block` = 1 is the i.i.d. day bootstrap; `block` = n collapses every
    resample to the data itself.
    """
    n = diff.size
    if not 1 <= block <= n:
        raise ValueError(f"block must be in 1..{n}, got {block}")
    rng = np.random.default_rng(seed)
    n_blocks = -(-n // block)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(log_front) * _betacf(a, b, x) / a
    return 1.0 - math.exp(log_front) * _betacf(b, a, 1.0 - x) / b


def student_t_two_sided_p(t: float, df: float) -> float:
    """P(|T| > |t|) for Student's t with `df` degrees of freedom."""
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def diebold_mariano_hln(d: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    `d` is the loss differential series (loss of forecast A minus loss of forecast B) for
    `h`-step-ahead forecasts. The long-run variance of the mean uses autocovariances up to lag
    h-1 (rectangular kernel). The statistic is multiplied by
    sqrt((n + 1 - 2h + h(h-1)/n) / n) and compared with Student's t on n-1 degrees of freedom.
    Returns (corrected statistic, two-sided p-value). A negative statistic means A has the lower
    loss.
    """
    n = d.size
    if n < 3:
        raise ValueError("need at least 3 observations")
    centred = d - d.mean()
    gamma = [float((centred[k:] * centred[: n - k]).sum() / n) for k in range(h)]
    var_mean = (gamma[0] + 2.0 * sum(gamma[1:])) / n
    if var_mean <= 0.0:
        raise ValueError("non-positive variance estimate (identical forecasts or h too large)")
    dm = float(d.mean()) / math.sqrt(var_mean)
    stat = dm * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, student_t_two_sided_p(stat, n - 1)


# --------------------------------------------------------------------------- folds


@dataclass
class FoldResult:
    month: tuple[int, int]
    test: Windows
    preds: dict[str, np.ndarray]
    ridge_alpha: float
    train_days: int
    train_days_quarter_hour: int
    seed_mae: list[float]
    member_preds: list[np.ndarray]
    resolution: str


def month_label(month: tuple[int, int]) -> str:
    return f"{month[0]}-{month[1]:02d}"


def resolution_flag(month: tuple[int, int], cutover: date | None) -> str:
    """'hourly', '15-min' or 'mixed' for the raw source resolution of a test month."""
    if cutover is None:
        return "hourly"
    first, last = month_bounds(*month)
    if last < cutover:
        return "hourly"
    return "15-min" if first >= cutover else "mixed"


def evaluate_fold(
    windows: Windows,
    month: tuple[int, int],
    ckpt: dict[str, Any],
    device: torch.device,
    cutover: date | None,
) -> FoldResult:
    """Score the fold's transformer checkpoint and the baselines on one test month."""
    split = split_windows(windows, month, int(ckpt["val_days"]))
    test = split.test
    if len(test) == 0:
        raise ValueError(f"no complete test days in {month_label(month)}")
    ridge = fit_ridge_tuned(split.train, split.val)
    bagged = fit_bagged_ridge(split.train, ridge.alpha)
    models = load_models(ckpt, device)
    ens, members = predict_ensemble(models, test, device)
    preds = {
        BASELINES[0]: seasonal_naive(test),
        BASELINES[1]: weekly_naive(test),
        RIDGE: ridge.predict(test),
        BAGGED: bagged.predict(test),
        MODEL: ens,
    }
    return FoldResult(
        month=month,
        test=test,
        preds=preds,
        ridge_alpha=ridge.alpha,
        train_days=len(split.train),
        train_days_quarter_hour=(
            sum(d >= cutover for d in split.train.dates) if cutover is not None else 0
        ),
        seed_mae=[mae(m, test.y) for m in members],
        member_preds=list(members),
        resolution=resolution_flag(month, cutover),
    )


def _compare(
    model_day: np.ndarray, base_day: np.ndarray, n_boot: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Per-day MAE difference (model minus baseline) with iid and block CIs and the DM-HLN test."""
    diff = model_day - base_day
    d_mean, lo, hi = paired_bootstrap(diff, n_boot=n_boot)
    _, b_lo, b_hi = block_bootstrap(diff, BLOCK_DAYS, n_boot=n_boot)
    dm_stat, dm_p = diebold_mariano_hln(diff, h=1)
    out = {
        "mean": d_mean,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "block_days": BLOCK_DAYS,
        "block_ci95_lo": b_lo,
        "block_ci95_hi": b_hi,
        "dm_statistic": dm_stat,
        "dm_p": dm_p,
        "n_days": int(diff.size),
    }
    return out, diff


def _direction(c: dict[str, Any]) -> str:
    """'beats' / 'loses to' if DM p < ALPHA and both CIs exclude 0 on the same side, else ''."""
    if c["dm_p"] >= ALPHA:
        return ""
    better = c["mean"] < 0
    for lo, hi in ((c["ci95_lo"], c["ci95_hi"]), (c["block_ci95_lo"], c["block_ci95_hi"])):
        if not (hi < 0 if better else lo > 0):
            return ""
    return "beats" if better else "loses to"


def summarize(folds: list[FoldResult], n_boot: int = 10_000) -> dict[str, Any]:
    """Per-fold and pooled scores, tests vs the baselines, and the verdict.

    The verdict compares the transformer with both ridge variants (plain and bagged). "beats" (or
    "loses to") needs, against each of them, DM-HLN p < 0.05 and both the iid and the moving-block
    (7-day) bootstrap CIs excluding zero on the same side; otherwise "no detectable difference".
    """
    y = np.concatenate([f.test.y for f in folds])
    pooled_preds = {name: np.concatenate([f.preds[name] for f in folds]) for name in folds[0].preds}
    pooled = {
        name: {"mae": mae(p, y), "rmse": rmse(p, y), "smape": smape(p, y)}
        for name, p in pooled_preds.items()
    }
    best = min(BASELINES, key=lambda n: pooled[n]["mae"])
    day = {name: per_day_mae(p, y) for name, p in pooled_preds.items()}
    comparisons: dict[str, dict[str, Any]] = {}
    for name in dict.fromkeys((best, *RIDGE_BASELINES)):
        comparisons[name], k_diff = _compare(day[MODEL], day[name], n_boot)
        comparisons[name]["folds_model_better"] = sum(
            mae(f.preds[MODEL], f.test.y) < mae(f.preds[name], f.test.y) for f in folds
        )
        comparisons[name]["post_hoc_block_sensitivity"] = {
            str(b): {"ci95_lo": lo, "ci95_hi": hi}
            for b in SENSITIVITY_BLOCKS
            for _, lo, hi in [block_bootstrap(k_diff, b, n_boot)]
        }
    directions = {_direction(comparisons[name]) for name in RIDGE_BASELINES}
    verdict = directions.pop() if len(directions) == 1 and "" not in directions else ""
    verdict = verdict or "no detectable difference"
    d_best = comparisons[best]
    # Each ensemble member alone vs the same baseline: separates ensembling from the model.
    single_seed = []
    for k in range(len(folds[0].member_preds)):
        member = np.concatenate([f.member_preds[k] for f in folds])
        k_diff = per_day_mae(member, y) - day[best]
        k_mean, k_lo, k_hi = paired_bootstrap(k_diff, n_boot=n_boot)
        k_stat, k_p = diebold_mariano_hln(k_diff, h=1)
        single_seed.append(
            {
                "seed": k,
                "mae": mae(member, y),
                "diff_mean": k_mean,
                "ci95_lo": k_lo,
                "ci95_hi": k_hi,
                "dm_statistic": k_stat,
                "dm_p": k_p,
            }
        )
    per_fold = []
    for f in folds:
        row: dict[str, Any] = {
            "month": month_label(f.month),
            "days": len(f.test),
            "resolution": f.resolution,
            "train_days": f.train_days,
            "train_days_quarter_hour": f.train_days_quarter_hour,
            "ridge_alpha": f.ridge_alpha,
            "mae": {name: mae(p, f.test.y) for name, p in f.preds.items()},
            "transformer_seed_mae": f.seed_mae,
            "negative_hours": int((f.test.y < 0).sum()),
            "near_zero_hours": int((np.abs(f.test.y) < NEAR_ZERO).sum()),
        }
        per_fold.append(row)
    return {
        "folds": len(folds),
        "days": int(y.shape[0]),
        "hours": int(y.size),
        "negative_hours": int((y < 0).sum()),
        "near_zero_hours": int((np.abs(y) < NEAR_ZERO).sum()),
        "pooled": pooled,
        "best_baseline": best,
        "mae_diff_vs_best_baseline": d_best,
        "ridge_comparisons": {name: comparisons[name] for name in RIDGE_BASELINES},
        "verdict": verdict,
        "per_fold": per_fold,
        "single_seed_vs_best_baseline": single_seed,
    }


def format_summary(s: dict[str, Any]) -> str:
    """Markdown: per-fold MAE table, pooled table, and the test results."""
    names = [*BASELINES, MODEL]
    lines = ["| Fold | Days | Source res. | 15-min train days | " + " | ".join(names) + " |"]
    lines.append("|---|---|---|---|" + "---|" * len(names))
    for r in s["per_fold"]:
        cells = " | ".join(f"{r['mae'][n]:.2f}" for n in names)
        lines.append(
            f"| {r['month']} | {r['days']} | {r['resolution']} | "
            f"{r['train_days_quarter_hour']}/{r['train_days']} | {cells} |"
        )
    lines += ["", f"Pooled over {s['folds']} folds, {s['days']} days, {s['hours']} hours:", ""]
    lines.append("| Model | MAE | RMSE | sMAPE % |")
    lines.append("|---|---|---|---|")
    for name in names:
        p = s["pooled"][name]
        lines.append(f"| {name} | {p['mae']:.2f} | {p['rmse']:.2f} | {p['smape']:.1f} |")
    lines += [
        "",
        "Per-day MAE, transformer minus baseline (negative = transformer better). CIs are 95%: "
        f"iid = day bootstrap, block = moving-block bootstrap, {BLOCK_DAYS}-day blocks (the "
        "verdict's CI). Post-hoc block-length sensitivity in the last column.",
        "",
        f"| Baseline | Diff | iid CI | block CI ({BLOCK_DAYS} d) | DM-HLN p | Better in folds | "
        "Block CI, 3 d / 14 d (post-hoc) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, c in s["ridge_comparisons"].items():
        sens = c["post_hoc_block_sensitivity"]
        sens_cells = " / ".join(
            f"[{v['ci95_lo']:+.2f}, {v['ci95_hi']:+.2f}]" for v in sens.values()
        )
        lines.append(
            f"| {name} | {c['mean']:+.2f} | [{c['ci95_lo']:+.2f}, {c['ci95_hi']:+.2f}] | "
            f"[{c['block_ci95_lo']:+.2f}, {c['block_ci95_hi']:+.2f}] | {c['dm_p']:.3g} | "
            f"{c['folds_model_better']}/{s['folds']} | {sens_cells} |"
        )
    lines += [
        "",
        f"Best baseline (pooled MAE): {s['best_baseline']}. {s['days']} days.",
        f"Pooled hours with negative price: {s['negative_hours']}; with |price| < "
        f"{NEAR_ZERO:g}: {s['near_zero_hours']} (sMAPE is unreliable there).",
        "",
        f"Verdict for the transformer vs the ridge baselines ({', '.join(RIDGE_BASELINES)}): "
        f"{s['verdict']}.",
        "",
        f"Each ensemble member alone vs {s['best_baseline']} (per-day MAE difference):",
        "",
        "| Seed | Pooled MAE | Diff vs baseline | 95% CI | DM-HLN p |",
        "|---|---|---|---|---|",
    ]
    for r in s["single_seed_vs_best_baseline"]:
        lines.append(
            f"| {r['seed']} | {r['mae']:.2f} | {r['diff_mean']:+.2f} | "
            f"[{r['ci95_lo']:+.2f}, {r['ci95_hi']:+.2f}] | {r['dm_p']:.3g} |"
        )
    return "\n".join(lines)
