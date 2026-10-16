"""Offline tests for probabilistic price forecasts: pinball, sorting, coverage, baselines, eval."""

from datetime import date

import numpy as np
import pytest
import torch
from test_minigpt_price import synthetic_table

from minigpt.price_data import Windows, make_windows, split_windows
from minigpt.price_model import (
    QUANTILES,
    PriceConfig,
    PriceTransformer,
    TrainConfig,
    build_features,
    fit_ridge,
    pinball_loss,
    predict_one,
    train_ensemble,
    train_one,
)
from minigpt.price_quantile import (
    METHODS,
    MODEL,
    QuantileFoldResult,
    _irls_quantile,
    add_offsets,
    cross_fit_ridge_residuals,
    crps_from_quantiles,
    crps_weights,
    evaluate_quantile_fold,
    fit_quantile_regression_tuned,
    format_quantile_summary,
    interval_coverage,
    interval_width,
    per_day_pinball,
    residual_offsets,
    ridge_quantiles,
    seasonal_naive_quantiles,
    sort_quantiles,
    summarize_quantile,
)


def windows_from(days: int = 200, seed: int = 0) -> Windows:
    return make_windows(synthetic_table(date(2025, 1, 1), days, seed))


# ------------------------------------------------------------------ pinball, sorting, coverage


def test_pinball_hand_computed() -> None:
    y = np.array([10.0])
    # y above the forecast: loss q * (y - f). y below the forecast: loss (1 - q) * (f - y).
    pred = np.array([[5.0, 15.0]])
    out = pinball_loss(pred, y, (0.1, 0.9))
    assert out[0] == pytest.approx([0.1 * 5.0, (1 - 0.9) * 5.0])
    # q = 0.5 is half the absolute error
    assert pinball_loss(np.array([[7.0]]), np.array([10.0]), (0.5,))[0, 0] == pytest.approx(1.5)
    # perfect forecast costs nothing
    assert pinball_loss(np.array([[10.0, 10.0]]), y, (0.1, 0.9)).sum() == 0.0


def test_pinball_is_minimised_by_the_empirical_quantile() -> None:
    y = np.arange(101.0)  # the 0.2-quantile is 20
    grid = np.arange(0.0, 101.0)
    for q in (0.2, 0.5, 0.8):
        losses = [pinball_loss(np.full((y.size, 1), g), y, (q,)).mean() for g in grid]
        assert grid[int(np.argmin(losses))] == pytest.approx(100 * q)


def test_sorting_repairs_crossing_and_never_hurts() -> None:
    crossed = np.array([[[20.0, 10.0, 15.0]]])  # (1 day, 1 hour, 3 levels), crossing
    fixed = sort_quantiles(crossed)
    assert fixed.tolist() == [[[10.0, 15.0, 20.0]]]
    levels = (0.1, 0.5, 0.9)
    for y in (5.0, 12.0, 17.0, 30.0):
        target = np.array([[y]])
        assert (
            pinball_loss(fixed, target, levels).sum() <= pinball_loss(crossed, target, levels).sum()
        )


def test_predict_one_sorts_crossing_head() -> None:
    cfg = PriceConfig(d_model=16, num_layers=1, num_heads=2, n_quantiles=3)
    model = PriceTransformer(cfg)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([3.0, 2.0, 1.0]))  # raw outputs descend: crossing
        model.head.weight.zero_()
    w = windows_from(40)
    feats = build_features(w.x, w.dates, cfg.lookback)
    raw = model.eval()(feats[:4]).detach().numpy()
    assert (np.diff(raw, axis=-1) < 0).all()
    out = predict_one(model, feats, w.x, torch.device("cpu"))
    assert out.shape == (len(w), 24, 3)
    assert (np.diff(out, axis=-1) >= 0).all()


def test_interval_coverage_and_width_hand_computed() -> None:
    levels = (0.1, 0.5, 0.9)
    pred = np.array(
        [[[0.0, 5.0, 10.0]], [[0.0, 5.0, 10.0]], [[0.0, 5.0, 10.0]], [[0.0, 5.0, 10.0]]]
    )
    y = np.array([[-1.0], [0.0], [10.0], [10.5]])  # below, on lower edge, on upper edge, above
    assert interval_coverage(pred, y, levels) == 0.5
    assert interval_width(pred, levels) == 10.0


def test_empirical_residual_quantiles_have_nominal_coverage() -> None:
    rng = np.random.default_rng(0)
    resid_train = rng.standard_t(5, size=(4000, 24)) * 10
    offsets = residual_offsets(resid_train, QUANTILES)
    assert offsets.shape == (24, len(QUANTILES))
    point = rng.normal(80, 20, size=(4000, 24))
    y = point + rng.standard_t(5, size=point.shape) * 10  # fresh draw from the same law
    pred = add_offsets(point, offsets)
    assert abs(interval_coverage(pred, y, QUANTILES) - 0.8) < 0.02
    assert (np.diff(pred, axis=-1) >= 0).all()


def test_crps_weights_and_point_forecast_crps() -> None:
    levels = (0.25, 0.5, 0.75)
    assert crps_weights(levels).tolist() == pytest.approx([0.375, 0.25, 0.375])
    assert crps_weights(QUANTILES).sum() == pytest.approx(1.0)
    # A point forecast has CRPS = |error|; the weighted sum reproduces it exactly here (hand: 1.0).
    pred = np.zeros((1, 1, 3))
    assert crps_from_quantiles(pred, np.array([[1.0]]), levels) == pytest.approx(1.0)


def test_per_day_pinball_averages_hours_and_levels() -> None:
    y = np.zeros((2, 24))
    pred = np.stack([np.full((2, 24), 2.0), np.full((2, 24), -4.0)], axis=-1)
    out = per_day_pinball(pred, y, (0.5, 0.5))
    assert out == pytest.approx([0.5 * (1.0 + 2.0) * 1.0] * 2)  # (0.5*2 + 0.5*4) / 2 = 1.5


# ------------------------------------------------------------------ baselines


def test_seasonal_naive_quantiles_use_only_the_training_window() -> None:
    w = windows_from(160)
    split = split_windows(w, (2025, 6), 28)
    base = seasonal_naive_quantiles(split.train, split.test, QUANTILES)
    assert base.shape == (len(split.test), 24, len(QUANTILES))
    # changing a test or validation target cannot move the forecast ...
    split.test.y += 1000.0
    split.val.y += 1000.0
    assert np.array_equal(base, seasonal_naive_quantiles(split.train, split.test, QUANTILES))
    # ... changing a training target does
    split.train.y += 5.0 * np.arange(len(split.train))[:, None]
    assert not np.array_equal(base, seasonal_naive_quantiles(split.train, split.test, QUANTILES))


def test_cross_fit_residuals_are_out_of_sample() -> None:
    rng = np.random.default_rng(1)
    w = windows_from(150)
    w.y = w.y + rng.normal(0, 10, w.y.shape)  # noise a 175-feature ridge can memorise
    in_sample = np.abs(w.y - fit_ridge(w, 0.1).predict(w)).mean()
    cross = np.abs(cross_fit_ridge_residuals(w, 0.1)).mean()
    assert cross > 2 * in_sample


def test_ridge_quantiles_shape_sorted_alpha_from_grid() -> None:
    w = windows_from(200)
    split = split_windows(w, (2025, 6), 28)
    pred, alpha = ridge_quantiles(split.train, split.val, split.test, QUANTILES)
    assert pred.shape == (len(split.test), 24, len(QUANTILES))
    assert alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert (np.diff(pred, axis=-1) >= 0).all()


def test_irls_recovers_empirical_quantiles_of_a_constant_model() -> None:
    z = np.arange(11.0)  # 0..10: the q-quantile of a sample of 11 is unique at 10 q
    x1 = np.ones((11, 1))
    for q in (0.1, 0.5, 0.9):
        b = _irls_quantile(x1, z, q, np.zeros((1, 1)), np.array([z.mean()]))
        assert b[0] == pytest.approx(10 * q, abs=0.05)


def test_irls_slope_and_calibration() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3000, 1))
    z = 2.0 * x[:, 0] + rng.laplace(0, 1, 3000)
    x1 = np.hstack([x, np.ones((3000, 1))])
    for q in (0.1, 0.5, 0.9):
        b = _irls_quantile(x1, z, q, np.zeros((2, 2)), np.array([2.0, 0.0]))
        assert b[0] == pytest.approx(2.0, abs=0.15)
        assert abs((z < x1 @ b).mean() - q) < 0.02


def test_quantile_regression_shapes_sorted_and_penalty_from_grid() -> None:
    w = windows_from(200)
    split = split_windows(w, (2025, 6), 28)
    qr = fit_quantile_regression_tuned(split.train, split.val, QUANTILES)
    pred = qr.predict(split.test)
    assert pred.shape == (len(split.test), 24, len(QUANTILES))
    assert np.isfinite(pred).all() and (np.diff(pred, axis=-1) >= 0).all()
    assert qr.penalty in (1.0, 10.0, 100.0, 1000.0)


# ------------------------------------------------------------------ transformer + evaluation


def test_train_one_rejects_inconsistent_head() -> None:
    w = windows_from(80)
    split = split_windows(w, (2025, 3), 28)
    tcfg = TrainConfig(epochs=1, seeds=1)
    device = torch.device("cpu")
    with pytest.raises(ValueError):
        train_one(split.train, split.val, PriceConfig(n_quantiles=3), tcfg, 0, device)
    with pytest.raises(ValueError):
        train_one(split.train, split.val, PriceConfig(), tcfg, 0, device, QUANTILES)
    with pytest.raises(ValueError):
        train_one(split.train, split.val, PriceConfig(n_quantiles=2), tcfg, 0, device, QUANTILES)


def test_quantile_ensemble_and_fold_end_to_end() -> None:
    w = windows_from(200)
    split = split_windows(w, (2025, 6), 28)
    device = torch.device("cpu")
    cfg = PriceConfig(d_model=16, num_layers=1, num_heads=2, n_quantiles=len(QUANTILES))
    ckpt = train_ensemble(
        split.train, split.val, cfg, TrainConfig(epochs=3, seeds=2), device, QUANTILES
    )
    ckpt.update(val_days=28)
    assert ckpt["quantiles"] == list(QUANTILES) and "val_pinball" in ckpt
    res = evaluate_quantile_fold(w, (2025, 6), ckpt, device, cutover=None)
    assert set(res.preds) == {*METHODS, MODEL}
    for pred in res.preds.values():
        assert pred.shape == (len(res.test), 24, len(QUANTILES))
        assert (np.diff(pred, axis=-1) >= 0).all()
    assert len(res.member_preds) == 2
    summary = summarize_quantile([res], n_boot=200)
    text = format_quantile_summary(summary)
    assert "Verdict" in text and summary["best_baseline"] in METHODS


def test_point_checkpoint_path_is_unchanged() -> None:
    w = windows_from(120)
    split = split_windows(w, (2025, 4), 28)
    cfg = PriceConfig(d_model=16, num_layers=1, num_heads=2)
    ckpt = train_ensemble(
        split.train, split.val, cfg, TrainConfig(epochs=1, seeds=1), torch.device("cpu")
    )
    assert "val_mae" in ckpt and "quantiles" not in ckpt and cfg.n_quantiles == 1
    model = PriceTransformer(cfg)
    assert model(build_features(w.x[:2], w.dates[:2])).shape == (2, 24)


def _fake_quantile_fold(month: int, model_noise: float, base_noise: float) -> QuantileFoldResult:
    rng = np.random.default_rng(month)
    table = synthetic_table(date(2025, month, 1), 30, seed=month)
    y = table.prices
    test = make_windows(table).subset(np.zeros(0, dtype=bool))
    test.dates, test.y, test.x = table.dates, y, y
    z = np.array([-1.28, -0.67, 0.0, 0.67, 1.28])  # normal quantiles of the five levels

    def forecast(noise: float) -> np.ndarray:
        centre = y + rng.normal(0, noise, y.shape)
        return sort_quantiles(centre[..., None] + noise * z)

    preds = {name: forecast(base_noise) for name in METHODS}
    preds[MODEL] = forecast(model_noise)
    return QuantileFoldResult((2025, month), test, preds, [preds[MODEL]], 1.0, 1.0, "hourly")


def test_summarize_quantile_verdicts() -> None:
    folds = (1, 2, 3)
    better = summarize_quantile([_fake_quantile_fold(m, 1.0, 20.0) for m in folds], n_boot=500)
    assert better["ensemble_vs_best_baseline"]["verdict"] == "beats"
    worse = summarize_quantile([_fake_quantile_fold(m, 20.0, 1.0) for m in folds], n_boot=500)
    assert worse["ensemble_vs_best_baseline"]["verdict"] == "loses to"
    same = summarize_quantile([_fake_quantile_fold(m, 5.0, 5.0) for m in folds], n_boot=500)
    assert same["ensemble_vs_best_baseline"]["verdict"] == "ties"
    assert same["days"] == 90 and same["single_seed_vs_best_baseline"][0]["seed"] == 0
    assert same["pooled"][MODEL]["coverage_80"] == pytest.approx(0.8, abs=0.1)
