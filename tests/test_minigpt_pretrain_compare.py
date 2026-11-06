import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from minigpt.pretrain_compare import compare, format_report, load_records


def record(windows: np.ndarray, step: int = 800) -> dict[str, Any]:
    return {
        "test_window_losses": windows.tolist(),
        "test_loss": float(windows.mean()),
        "val_loss": 3.5,
        "selected_step": step,
    }


def write_runs(directory: Path, gains: dict[str, list[float]]) -> None:
    """Variant `v` beats base by gains[v][seed] on every test window (plus tiny jitter)."""
    rng = np.random.default_rng(0)
    for seed in range(3):
        base = 3.8 + 0.1 * rng.standard_normal(89)
        (directory / f"pre_base_s{seed}.json").write_text(json.dumps(record(base)))
        for variant, per_seed in gains.items():
            var = base - per_seed[seed] + 0.005 * rng.standard_normal(89)
            (directory / f"pre_{variant}_s{seed}.json").write_text(json.dumps(record(var)))


def test_adopts_only_when_every_seed_clears_the_margin(tmp_path: Path) -> None:
    write_runs(tmp_path, {"good": [0.08, 0.07, 0.06], "weak": [0.08, 0.07, 0.02]})
    report = compare(load_records(tmp_path), n_boot=500)
    assert report["variants"]["good"]["adopt"] is True
    assert report["variants"]["weak"]["adopt"] is False
    assert report["adopted"] == "good"
    assert "adopted default: good" in format_report(report)


def test_margin_met_but_ci_includes_zero_is_not_adopted(tmp_path: Path) -> None:
    # large gain on average, but the per-window differences are so noisy the CI spans 0
    rng = np.random.default_rng(1)
    for seed in range(3):
        base = 3.8 + 0.1 * rng.standard_normal(89)
        (tmp_path / f"pre_base_s{seed}.json").write_text(json.dumps(record(base)))
        z = rng.standard_normal(89)
        var = base + 1.5 * (z - z.mean()) - 0.06  # mean gain exactly 0.06
        (tmp_path / f"pre_noisy_s{seed}.json").write_text(json.dumps(record(var)))
    report = compare(load_records(tmp_path), n_boot=500)
    noisy = report["variants"]["noisy"]["per_seed"]["0"]
    assert -noisy["delta"] == pytest.approx(0.06, abs=1e-6)
    assert noisy["ci_lo"] < 0 < noisy["ci_hi"]
    assert report["variants"]["noisy"]["adopt"] is False
    assert report["adopted"] is None
    assert "keep the existing recipe" in format_report(report)


def test_worse_variant_is_not_adopted_and_both_adopted_picks_lower_mean(tmp_path: Path) -> None:
    write_runs(tmp_path, {"a": [0.08, 0.08, 0.08], "b": [0.12, 0.12, 0.12], "bad": [-0.1] * 3})
    report = compare(load_records(tmp_path), n_boot=500)
    assert report["variants"]["bad"]["adopt"] is False
    assert report["adopted"] == "b"
