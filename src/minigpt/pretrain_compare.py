"""Pre-registered comparison of pretraining variants on the held-out test half.

Implements docs/protocol-pretrain-regularization.md: per-seed paired bootstrap over test windows,
and the adoption rule (every seed improves by >= MIN_GAIN and its 95% CI excludes 0).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from minigpt.evaluate import paired_bootstrap_test

MIN_GAIN = 0.05
BASELINE = "base"


def load_records(directory: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Read `pre_<variant>_s<seed>.json` files into {(variant, seed): record}."""
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(Path(directory).glob("pre_*_s*.json")):
        variant, seed = path.stem.removeprefix("pre_").rsplit("_s", 1)
        out[(variant, int(seed))] = json.loads(path.read_text(encoding="utf-8"))
    return out


def compare(
    records: dict[tuple[str, int], dict[str, Any]], n_boot: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Per-seed test deltas vs the baseline, bootstrap CIs, and the adoption verdict per variant."""
    seeds = sorted({s for v, s in records if v == BASELINE})
    variants = sorted({v for v, _ in records if v != BASELINE})
    report: dict[str, Any] = {
        "seeds": seeds,
        "min_gain": MIN_GAIN,
        "baseline": {
            str(s): {
                "test_loss": records[(BASELINE, s)]["test_loss"],
                "test_ppl": math.exp(records[(BASELINE, s)]["test_loss"]),
                "val_loss": records[(BASELINE, s)]["val_loss"],
                "selected_step": records[(BASELINE, s)]["selected_step"],
            }
            for s in seeds
        },
        "variants": {},
    }
    base_mean = float(np.mean([records[(BASELINE, s)]["test_loss"] for s in seeds]))
    report["baseline_mean_test_loss"] = base_mean
    report["baseline_mean_test_ppl"] = math.exp(base_mean)
    for v in variants:
        per_seed: dict[str, Any] = {}
        for s in seeds:
            base, var = records[(BASELINE, s)], records[(v, s)]
            diff = np.asarray(var["test_window_losses"]) - np.asarray(base["test_window_losses"])
            mean, lo, hi, p = paired_bootstrap_test(diff, n_boot=n_boot, seed=seed)
            per_seed[str(s)] = {
                "test_loss": var["test_loss"],
                "val_loss": var["val_loss"],
                "selected_step": var["selected_step"],
                "delta": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "p": p,
                "meets_rule": bool(-mean >= MIN_GAIN and hi < 0),
            }
        mean_test = float(np.mean([per_seed[str(s)]["test_loss"] for s in seeds]))
        report["variants"][v] = {
            "per_seed": per_seed,
            "mean_test_loss": mean_test,
            "mean_delta": mean_test - base_mean,
            "adopt": all(per_seed[str(s)]["meets_rule"] for s in seeds),
        }
    adopted = [v for v in variants if report["variants"][v]["adopt"]]
    report["adopted"] = min(
        adopted, key=lambda v: report["variants"][v]["mean_test_loss"], default=None
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    """Markdown tables of per-seed test loss and deltas, plus the verdict line."""
    seeds = report["seeds"]
    lines = [
        "| Variant | seed | selected step | val loss | test loss | test ppl "
        "| Δ test vs base [95% CI] |"
    ]
    lines.append("|---|---|---|---|---|---|---|")
    for s in seeds:
        b = report["baseline"][str(s)]
        lines.append(
            f"| base | {s} | {b['selected_step']} | {b['val_loss']:.4f} | {b['test_loss']:.4f} "
            f"| {b['test_ppl']:.2f} | n/a |"
        )
    for v, info in report["variants"].items():
        for s in seeds:
            r = info["per_seed"][str(s)]
            lines.append(
                f"| {v} | {s} | {r['selected_step']} | {r['val_loss']:.4f} | {r['test_loss']:.4f} "
                f"| {math.exp(r['test_loss']):.2f} | {r['delta']:+.4f} "
                f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |"
            )
    lines.append("")
    lines.append(
        f"base mean test loss {report['baseline_mean_test_loss']:.4f} "
        f"(ppl {report['baseline_mean_test_ppl']:.2f})"
    )
    for v, info in report["variants"].items():
        lines.append(
            f"{v}: mean test loss {info['mean_test_loss']:.4f} (Δ {info['mean_delta']:+.4f}); "
            f"rule met on every seed: {info['adopt']}"
        )
    lines.append(f"adopted default: {report['adopted'] or 'none (keep the existing recipe)'}")
    return "\n".join(lines)
