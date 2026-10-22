"""Conformalized quantile regression (CQR) of the 80% price interval.

Romano, Patterson and Candes (2019), "Conformalized Quantile Regression", NeurIPS 32,
arXiv:1905.03222. Given a forecast interval [lo(x), hi(x)] (here the q0.1 and q0.9 forecasts), the
conformity score of a calibration point is E = max(lo(x) - y, y - hi(x)): positive when y falls
outside, negative (minus the distance to the nearest edge) when inside. With n calibration scores
and miscoverage alpha, Q is the ceil((n + 1)(1 - alpha))-th smallest score, and the calibrated
interval is [lo - Q, hi + Q]. Q > 0 widens the band, Q < 0 shrinks it. If calibration and test
points are exchangeable, the interval covers y with probability >= 1 - alpha.

Protocol, fixed before any conformal number was computed:

* Calibration window: the fold's existing validation window, the 28 days before the test month,
  for every method. Strictly before the test month, so no test day enters calibration. The
  alternative (the last N training days) was rejected up front: a fitted model's training-window
  forecasts are in-sample, so its scores there are too small. Caveat: validation days also chose
  the transformer epoch, the ridge alpha and the quantile-regression penalty, so the scores are
  mildly optimistic and the correction mildly too small.
* alpha = 1 - (0.9 - 0.1) = 0.2. Only the q0.1 and q0.9 forecasts are shifted; q0.25, q0.5 and
  q0.75 are left alone, and every forecast is re-sorted along the level axis (a shrunk band can
  cross an inner level). Pinball over the five levels therefore changes only through the two
  outer levels and the re-sort.
* Variants: `cqr` pools the 28 x 24 = 672 scores of the window into one Q per fold (the primary
  variant); `cqr_hour` uses one Q per hour of day from the 28 scores of that hour (secondary).
* Methods: the four methods of `price_quantile` plus each single transformer seed.
* Verdict rule: coverage "reaches nominal" if the 95% day-cluster bootstrap CI of the pooled
  coverage contains 0.80; otherwise it "undercovers" or "overcovers". The width cost is the
  relative change in mean q0.9 - q0.1. The Wilson CI is also reported, but it treats the 8,760
  hours as independent, which they are not (24 hours of one day share one price curve), so it is
  narrower than the day-cluster CI and is not used in the rule.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import numpy as np

from minigpt.price_eval import month_label, paired_bootstrap
from minigpt.price_quantile import (
    LOWER,
    METHODS,
    MODEL,
    UPPER,
    QuantileFoldResult,
    paired_verdict,
    paired_versus,
    per_day_pinball,
    score,
    sort_quantiles,
)

NOMINAL = UPPER - LOWER
ALPHA_MISS = 1.0 - NOMINAL
VARIANTS = ("raw", "cqr", "cqr_hour")
VARIANT_LABELS = {
    "raw": "raw",
    "cqr": "CQR (pooled hours)",
    "cqr_hour": "CQR (per hour of day)",
}


# --------------------------------------------------------------------------- CQR math


def conformity_scores(
    pred: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...],
    lo: float = LOWER,
    hi: float = UPPER,
) -> np.ndarray:
    """E = max(q_lo - y, y - q_hi), shape of `y`. Negative inside the band, positive outside."""
    lower, upper = pred[..., quantiles.index(lo)], pred[..., quantiles.index(hi)]
    out: np.ndarray = np.maximum(lower - y, y - upper)
    return out


def conformal_rank(n: int, alpha: float) -> int:
    """1-based rank ceil((n + 1)(1 - alpha)) of the score that bounds the new point's score."""
    rank = math.ceil((n + 1) * (1.0 - alpha) - 1e-12)
    if rank > n:
        raise ValueError(f"{n} calibration scores are too few for miscoverage {alpha}")
    return rank


def conformal_quantile(scores: np.ndarray, alpha: float = ALPHA_MISS) -> float:
    """The finite-sample conformal quantile Q of all `scores`."""
    flat = np.sort(scores.ravel())
    return float(flat[conformal_rank(flat.size, alpha) - 1])


def conformal_quantile_by_column(scores: np.ndarray, alpha: float = ALPHA_MISS) -> np.ndarray:
    """One Q per column of (n, k) `scores`, each from its own n scores."""
    srt = np.sort(scores, axis=0)
    out: np.ndarray = srt[conformal_rank(scores.shape[0], alpha) - 1]
    return out


def apply_cqr(
    pred: np.ndarray,
    q_hat: np.ndarray | float,
    quantiles: tuple[float, ...],
    lo: float = LOWER,
    hi: float = UPPER,
) -> np.ndarray:
    """[q_lo - Q, q_hi + Q] on a copy of `pred`; `q_hat` is a scalar or one value per hour."""
    out = pred.copy()
    out[..., quantiles.index(lo)] -= q_hat
    out[..., quantiles.index(hi)] += q_hat
    return sort_quantiles(out)


def calibrate(
    cal_pred: np.ndarray,
    cal_y: np.ndarray,
    test_pred: np.ndarray,
    quantiles: tuple[float, ...],
    per_hour: bool,
    alpha: float = ALPHA_MISS,
) -> tuple[np.ndarray, np.ndarray]:
    """CQR-adjusted test forecasts and the Q used, shape (24,) (all equal when pooled)."""
    scores = conformity_scores(cal_pred, cal_y, quantiles)  # (n_cal_days, 24)
    if per_hour:
        q_hat = conformal_quantile_by_column(scores, alpha)
    else:
        q_hat = np.full(scores.shape[1], conformal_quantile(scores, alpha))
    return apply_cqr(test_pred, q_hat, quantiles), q_hat


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion k / n."""
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def interval_score(
    pred: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...],
    alpha: float = ALPHA_MISS,
    lo: float = LOWER,
    hi: float = UPPER,
) -> float:
    """Winkler / interval score (EUR/MWh): width + (2/alpha) x distance outside. Lower is better."""
    lower, upper = pred[..., quantiles.index(lo)], pred[..., quantiles.index(hi)]
    s = (upper - lower) + (2 / alpha) * (np.maximum(lower - y, 0) + np.maximum(y - upper, 0))
    return float(s.mean())


# --------------------------------------------------------------------------- evaluation


def window_era(dates: list[date], cutover: date | None) -> str:
    """'hourly', '15-min' or 'mixed' by the raw source resolution of the given local days."""
    if cutover is None or all(d < cutover for d in dates):
        return "hourly"
    return "15-min" if all(d >= cutover for d in dates) else "mixed"


def _day_covered(pred: np.ndarray, y: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    lower, upper = pred[..., quantiles.index(LOWER)], pred[..., quantiles.index(UPPER)]
    out: np.ndarray = ((y >= lower) & (y <= upper)).mean(axis=1)
    return out


def _coverage_stats(day_cov: np.ndarray, n_boot: int) -> dict[str, Any]:
    n_hours = int(day_cov.size * 24)
    k = round(float(day_cov.sum()) * 24)
    w_lo, w_hi = wilson_interval(k, n_hours)
    mean, b_lo, b_hi = paired_bootstrap(day_cov, n_boot=n_boot)
    return {
        "coverage": mean,
        "wilson_lo": w_lo,
        "wilson_hi": w_hi,
        "cluster_lo": b_lo,
        "cluster_hi": b_hi,
        "status": coverage_status(b_lo, b_hi),
    }


def coverage_status(lo: float, hi: float, nominal: float = NOMINAL) -> str:
    """Pre-registered rule: nominal is reached iff the day-cluster CI contains it."""
    if lo <= nominal <= hi:
        return "reaches nominal"
    return "undercovers" if hi < nominal else "overcovers"


def _fold_series(
    folds: list[QuantileFoldResult],
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Series name -> per-fold (calibration forecast, test forecast)."""
    series: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name in (*METHODS, MODEL):
        series[name] = [(f.val_preds[name], f.preds[name]) for f in folds]
    for k in range(len(folds[0].member_preds)):
        series[f"{MODEL}, seed {k}"] = [(f.val_member_preds[k], f.member_preds[k]) for f in folds]
    return series


def _variant_preds(
    folds: list[QuantileFoldResult],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    variant: str,
    quantiles: tuple[float, ...],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-fold forecasts and per-fold Q (24,) of one variant."""
    preds: list[np.ndarray] = []
    q_hats: list[np.ndarray] = []
    for f, (cal, test) in zip(folds, pairs, strict=True):
        if variant == "raw":
            preds.append(test)
            q_hats.append(np.zeros(24))
        else:
            adjusted, q_hat = calibrate(cal, f.val.y, test, quantiles, variant == "cqr_hour")
            preds.append(adjusted)
            q_hats.append(q_hat)
    return preds, q_hats


def summarize_conformal(
    folds: list[QuantileFoldResult],
    cutover: date | None,
    quantiles: tuple[float, ...],
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Raw vs CQR vs per-hour CQR for every method and single seed, pooled and per fold."""
    y = np.concatenate([f.test.y for f in folds])
    test_dates = [d for f in folds for d in f.test.dates]
    is_new_era = np.asarray([cutover is not None and d >= cutover for d in test_dates], dtype=bool)
    era_masks = {
        era: mask
        for era, mask in (("hourly", np.logical_not(is_new_era)), ("15-min", is_new_era))
        if mask.any()
    }
    fold_days = [len(f.test) for f in folds]
    fold_cal_era = [window_era(f.val.dates, cutover) for f in folds]
    cal_era_of_day = np.repeat(fold_cal_era, fold_days)
    series = _fold_series(folds)

    pooled_pred: dict[str, dict[str, np.ndarray]] = {}
    out: dict[str, dict[str, Any]] = {}
    for name, pairs in series.items():
        out[name] = {}
        pooled_pred[name] = {}
        raw_pred = np.concatenate([p[1] for p in pairs])
        for variant in VARIANTS:
            preds, q_hats = _variant_preds(folds, pairs, variant, quantiles)
            pooled = np.concatenate(preds)
            pooled_pred[name][variant] = pooled
            day_cov = _day_covered(pooled, y, quantiles)
            fold_cov = [float(c.mean()) for c in np.split(day_cov, np.cumsum(fold_days)[:-1])]
            stats = _coverage_stats(day_cov, n_boot)
            entry: dict[str, Any] = {
                **score(pooled, y, quantiles),
                **stats,
                "interval_score": interval_score(pooled, y, quantiles),
                "fold_coverage": fold_cov,
                "fold_coverage_min": min(fold_cov),
                "fold_coverage_max": max(fold_cov),
                "fold_coverage_std": float(np.std(fold_cov)),
                "q_hat_mean": float(np.mean([q.mean() for q in q_hats])),
                "q_hat_by_fold": [float(q.mean()) for q in q_hats],
                "coverage_by_test_era": {
                    era: float(day_cov[mask].mean()) for era, mask in era_masks.items()
                },
                "days_by_test_era": {era: int(mask.sum()) for era, mask in era_masks.items()},
                "coverage_by_calibration_era": {
                    era: float(day_cov[cal_era_of_day == era].mean())
                    for era in sorted(set(fold_cal_era))
                },
                "folds_by_calibration_era": {
                    era: [
                        month_label(f.month)
                        for f, e in zip(folds, fold_cal_era, strict=True)
                        if e == era
                    ]
                    for era in sorted(set(fold_cal_era))
                },
            }
            if variant != "raw":
                diff = per_day_pinball(pooled, y, quantiles) - per_day_pinball(
                    raw_pred, y, quantiles
                )
                mean, lo, hi = paired_bootstrap(diff, n_boot=n_boot)
                entry["pinball_diff_vs_raw"] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi}
                entry["width_change_vs_raw"] = (
                    entry["width_80"] / out[name]["raw"]["width_80"] - 1.0
                )
            out[name][variant] = entry

    versus: dict[str, Any] = {}
    for variant in VARIANTS:
        best = min(METHODS, key=lambda n: out[n][variant]["mean_pinball"])
        v = paired_versus(
            pooled_pred[MODEL][variant], pooled_pred[best][variant], y, quantiles, n_boot
        )
        v["verdict"] = paired_verdict(v)
        v["best_baseline"] = best
        versus[variant] = v
    return {
        "calibration": "fold validation window (28 days before the test month)",
        "calibration_days": [len(f.val) for f in folds],
        "calibration_era_by_fold": dict(
            zip((month_label(f.month) for f in folds), fold_cal_era, strict=True)
        ),
        "folds": [month_label(f.month) for f in folds],
        "days": int(y.shape[0]),
        "hours": int(y.size),
        "nominal": NOMINAL,
        "results": out,
        "ensemble_vs_best_baseline": versus,
    }


def _row(label: str, e: dict[str, Any]) -> str:
    extra = ""
    if "pinball_diff_vs_raw" in e:
        d = e["pinball_diff_vs_raw"]
        extra = f"{d['mean']:+.2f} [{d['ci95_lo']:+.2f}, {d['ci95_hi']:+.2f}]"
    return (
        f"| {label} | {100 * e['coverage']:.1f}% | "
        f"[{100 * e['wilson_lo']:.1f}, {100 * e['wilson_hi']:.1f}] | "
        f"[{100 * e['cluster_lo']:.1f}, {100 * e['cluster_hi']:.1f}] | {e['status']} | "
        f"{e['width_80']:.1f} | {e['interval_score']:.1f} | {e['mean_pinball']:.2f} | "
        f"{extra or '-'} | {100 * e['fold_coverage_min']:.0f}-{100 * e['fold_coverage_max']:.0f}% "
        f"(sd {100 * e['fold_coverage_std']:.1f}) |"
    )


def format_conformal_summary(s: dict[str, Any]) -> str:
    """Markdown tables: pooled coverage / width / score before and after, then era splits."""
    lines = [
        f"Conformal calibration: {s['calibration']}, nominal {100 * s['nominal']:.0f}%. "
        f"Pooled over {len(s['folds'])} folds, {s['days']} days, {s['hours']} hours.",
        "",
        "| Method / variant | Coverage | Wilson 95% (hours) | Day-cluster 95% | Verdict "
        "| Width | Interval score | Mean pinball | Pinball diff vs raw [95% CI] "
        "| Per-fold coverage range |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, by_variant in s["results"].items():
        for variant in VARIANTS:
            lines.append(_row(f"{name}, {VARIANT_LABELS[variant]}", by_variant[variant]))
    lines += [
        "",
        "Ensemble minus best baseline (per-day mean pinball), same variant on both sides:",
        "",
    ]
    lines += [
        "| Variant | Best baseline | Diff | 95% CI | DM-HLN p | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for variant, v in s["ensemble_vs_best_baseline"].items():
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {v['best_baseline']} | {v['diff_mean']:+.2f} | "
            f"[{v['ci95_lo']:+.2f}, {v['ci95_hi']:+.2f}] | {v['dm_p']:.3g} | {v['verdict']} |"
        )
    lines += ["", "Coverage of the transformer ensemble by era:", ""]
    lines += [
        "| Variant | Test days by era | Folds by calibration era -> coverage |",
        "|---|---|---|",
    ]
    for variant in VARIANTS:
        e = s["results"][MODEL][variant]
        days = ", ".join(f"{era}: {n}" for era, n in e["days_by_test_era"].items())
        cal = "; ".join(
            f"{era} ({len(e['folds_by_calibration_era'][era])} folds) -> {100 * c:.1f}%"
            for era, c in e["coverage_by_calibration_era"].items()
        )
        lines.append(f"| {VARIANT_LABELS[variant]} | {days} | {cal} |")
    lines += ["", "Per-fold coverage of the transformer ensemble:", ""]
    lines += [
        "| Fold | Calibration era | " + " | ".join(VARIANT_LABELS[v] for v in VARIANTS) + " |"
    ]
    lines += ["|---|---|" + "---|" * len(VARIANTS)]
    for i, month in enumerate(s["folds"]):
        cells = " | ".join(
            f"{100 * s['results'][MODEL][v]['fold_coverage'][i]:.1f}%" for v in VARIANTS
        )
        lines.append(f"| {month} | {s['calibration_era_by_fold'][month]} | {cells} |")
    return "\n".join(lines)
