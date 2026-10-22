"""Offline tests for probabilistic price forecasts: pinball, sorting, coverage, baselines, eval."""

from datetime import date

import numpy as np
import pytest
import torch
from test_minigpt_price import synthetic_table

from minigpt.price_conformal import (
    apply_cqr,
    calibrate,
    conformal_quantile,
    conformal_quantile_by_column,
    conformal_rank,
    conformity_scores,
    coverage_status,
    format_conformal_summary,
    interval_score,
    summarize_conformal,
    wilson_interval,
    window_era,
)
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
    (pred,), alpha = ridge_quantiles(split.train, split.val, (split.test,), QUANTILES)
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
    assert len(res.val) == 28 and len(res.val_member_preds) == 2
    for name, pred in res.val_preds.items():
        assert pred.shape == (28, 24, len(QUANTILES)) and name in res.preds
    assert set(res.preds) == {*METHODS, MODEL}
    for pred in res.preds.values():
        assert pred.shape == (len(res.test), 24, len(QUANTILES))
        assert (np.diff(pred, axis=-1) >= 0).all()
    assert len(res.member_preds) == 2
    summary = summarize_quantile([res], n_boot=200)
    text = format_quantile_summary(summary)
    assert "Verdict" in text and summary["best_baseline"] in METHODS
    conformal = summarize_conformal([res], None, QUANTILES, n_boot=200)
    assert set(conformal["results"]) == {*METHODS, MODEL, f"{MODEL}, seed 0", f"{MODEL}, seed 1"}
    assert "CQR (pooled hours)" in format_conformal_summary(conformal)


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


def _fake_window(start: date, days: int, seed: int) -> Windows:
    table = synthetic_table(start, days, seed=seed)
    win = make_windows(table).subset(np.zeros(0, dtype=bool))
    win.dates, win.y, win.x = table.dates, table.prices, table.prices
    return win


def _fake_quantile_fold(
    month: int, model_noise: float, base_noise: float, band: float = 1.0
) -> QuantileFoldResult:
    rng = np.random.default_rng(month)
    test = _fake_window(date(2025, month, 1), 30, month)
    val = _fake_window(date(2024, month, 1), 28, 100 + month)
    z = np.array([-1.28, -0.67, 0.0, 0.67, 1.28])  # normal quantiles of the five levels

    def forecast(y: np.ndarray, noise: float) -> np.ndarray:
        centre = y + rng.normal(0, noise, y.shape)
        return sort_quantiles(centre[..., None] + band * noise * z)

    preds = {name: forecast(test.y, base_noise) for name in METHODS}
    preds[MODEL] = forecast(test.y, model_noise)
    val_preds = {name: forecast(val.y, base_noise) for name in METHODS}
    val_preds[MODEL] = forecast(val.y, model_noise)
    return QuantileFoldResult(
        (2025, month),
        test,
        preds,
        [preds[MODEL]],
        val,
        val_preds,
        [val_preds[MODEL]],
        1.0,
        1.0,
        "hourly",
    )


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


# ------------------------------------------------------------------ conformalized QR

Q3 = (0.1, 0.5, 0.9)


def test_conformity_scores_hand_computed() -> None:
    # band [0, 10]: y = -3 is 3 below (score +3), y = 2 is 2 inside the lower edge (-2),
    # y = 12 is 2 above (+2)
    pred = np.tile(np.array([0.0, 5.0, 10.0]), (3, 1))
    scores = conformity_scores(pred, np.array([-3.0, 2.0, 12.0]), Q3)
    assert scores == pytest.approx([3.0, -2.0, 2.0])
    # inside, nearer the upper edge: y = 9 is 1 inside (-1)
    assert conformity_scores(pred[:1], np.array([9.0]), Q3)[0] == pytest.approx(-1.0)


def test_conformal_rank_finite_sample_hand_cases() -> None:
    assert conformal_rank(4, 0.2) == 4  # ceil(5 * 0.8)
    assert (
        conformal_rank(9, 0.2) == 8
    )  # ceil(10 * 0.8), exact in reals; float noise must not bump it
    assert conformal_rank(10, 0.2) == 9  # ceil(8.8)
    assert conformal_rank(672, 0.2) == 539  # ceil(673 * 0.8) = ceil(538.4)
    with pytest.raises(ValueError):
        conformal_rank(3, 0.2)  # ceil(3.2) = 4 > 3: no finite quantile


def test_conformal_quantile_picks_the_ranked_score() -> None:
    scores = np.array([5.0, -1.0, 3.0, 0.0, 2.0, 7.0, -4.0, 1.0, 6.0, 4.0])
    # sorted: -4 -1 0 1 2 3 4 5 6 7; rank 9 -> 6
    assert conformal_quantile(scores, 0.2) == 6.0
    two_d = np.stack([scores, scores * 2], axis=1)
    assert conformal_quantile_by_column(two_d, 0.2) == pytest.approx([6.0, 12.0])
    assert conformal_quantile(np.array([-3.0, -2.0, -1.0, -0.5]), 0.2) == -0.5  # negative: shrink


def test_apply_cqr_widens_shrinks_and_resorts() -> None:
    pred = np.array([[[0.0, 5.0, 10.0]]])
    assert apply_cqr(pred, 2.0, Q3).ravel() == pytest.approx([-2.0, 5.0, 12.0])
    assert apply_cqr(pred, -3.0, Q3).ravel() == pytest.approx([3.0, 5.0, 7.0])
    # shrinking past the median crosses it: the result is re-sorted, never crossed
    assert apply_cqr(pred, -6.0, Q3).ravel() == pytest.approx([4.0, 5.0, 6.0])
    assert pred.ravel() == pytest.approx([0.0, 5.0, 10.0])  # input untouched
    per_hour = np.zeros((1, 24, 3))
    per_hour[..., 2] = 10.0
    hourly = apply_cqr(per_hour, np.arange(24.0), Q3)
    assert hourly[0, 3, 2] == pytest.approx(13.0) and hourly[0, 3, 0] == pytest.approx(-3.0)


def test_calibrate_pooled_and_per_hour_hand_case() -> None:
    # band [0, 10] everywhere; hour h scores are (h + 1) * [1, 2, 3, 4] (y above the band)
    cal_pred = np.zeros((4, 24, 3))
    cal_pred[..., 2] = 10.0
    cal_y = 10.0 + (np.arange(24) + 1.0)[None] * np.array([1.0, 2.0, 3.0, 4.0])[:, None]
    test_pred = np.zeros((2, 24, 3))
    test_pred[..., 2] = 10.0
    _, per_hour_q = calibrate(cal_pred, cal_y, test_pred, Q3, per_hour=True)
    assert per_hour_q == pytest.approx(4.0 * (np.arange(24) + 1.0))  # rank 4 of 4 = the max
    adjusted, pooled_q = calibrate(cal_pred, cal_y, test_pred, Q3, per_hour=False)
    all_scores = conformity_scores(cal_pred, cal_y, Q3).ravel()
    assert pooled_q[0] == np.sort(all_scores)[conformal_rank(96, 0.2) - 1]
    assert adjusted[0, 5, 2] == pytest.approx(10.0 + pooled_q[5])
    assert adjusted[0, 5, 0] == pytest.approx(0.0 - pooled_q[5])


def test_cqr_restores_coverage_of_a_too_narrow_band() -> None:
    rng = np.random.default_rng(0)
    z80 = 1.2815516  # N(0,1) 0.9 quantile
    narrow = np.array([-0.5, 0.0, 0.5])  # a band that is far too tight for N(0, 1) noise

    def sample(n: int) -> tuple[np.ndarray, np.ndarray]:
        return np.broadcast_to(narrow, (n, 24, 3)).copy(), rng.normal(size=(n, 24))

    cal_pred, cal_y = sample(2000)
    test_pred, test_y = sample(2000)
    before = ((test_y >= test_pred[..., 0]) & (test_y <= test_pred[..., 2])).mean()
    assert before == pytest.approx(0.38, abs=0.02)  # P(|N| < 0.5)
    for per_hour in (False, True):
        adjusted, q_hat = calibrate(cal_pred, cal_y, test_pred, Q3, per_hour)
        after = ((test_y >= adjusted[..., 0]) & (test_y <= adjusted[..., 2])).mean()
        assert after == pytest.approx(0.8, abs=0.015)
        assert q_hat.mean() == pytest.approx(z80 - 0.5, abs=0.06)  # band half-width 0.5 -> 1.28


def test_wilson_interval_hand_cases() -> None:
    lo, hi = wilson_interval(50, 100)
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)
    lo0, hi0 = wilson_interval(0, 100)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0.03 < hi0 < 0.04  # known: 3.7%
    lo1, hi1 = wilson_interval(100, 100)
    assert hi1 == pytest.approx(1.0, abs=1e-12) and lo1 == pytest.approx(1 - hi0)


def test_interval_score_hand_computed() -> None:
    pred = np.tile(np.array([0.0, 5.0, 10.0]), (3, 1, 1))  # (3, 1, 3)
    y = np.array([[5.0], [-1.0], [13.0]])
    # width 10 each; penalty (2 / 0.2) = 10 per EUR outside: 0, 10 x 1, 10 x 3
    assert interval_score(pred, y, Q3) == pytest.approx((10 + 20 + 40) / 3)


def test_coverage_status_rule_and_window_era() -> None:
    assert coverage_status(0.78, 0.82) == "reaches nominal"
    assert coverage_status(0.70, 0.76) == "undercovers"
    assert coverage_status(0.83, 0.88) == "overcovers"
    cut = date(2025, 10, 1)
    assert window_era([date(2025, 9, 30)], cut) == "hourly"
    assert window_era([date(2025, 10, 1)], cut) == "15-min"
    assert window_era([date(2025, 9, 30), date(2025, 10, 1)], cut) == "mixed"
    assert window_era([date(2030, 1, 1)], None) == "hourly"


def test_summarize_conformal_fixes_undercoverage_and_reports_cost() -> None:
    folds = [_fake_quantile_fold(m, 5.0, 5.0, band=0.3) for m in (1, 2, 3)]
    s = summarize_conformal(folds, date(2025, 1, 1), QUANTILES, n_boot=300)
    ens = s["results"][MODEL]
    assert ens["raw"]["coverage"] < 0.55 and ens["raw"]["status"] == "undercovers"
    for variant in ("cqr", "cqr_hour"):
        assert ens[variant]["coverage"] == pytest.approx(0.8, abs=0.06)
        assert ens[variant]["width_change_vs_raw"] > 0.5  # the cost of the repair
        assert ens[variant]["q_hat_mean"] > 0
        assert ens[variant]["interval_score"] < ens["raw"]["interval_score"]
        assert ens[variant]["wilson_lo"] < ens[variant]["coverage"] < ens[variant]["wilson_hi"]
        assert len(ens[variant]["fold_coverage"]) == 3
    # test days are 2025 (15-min era by this cutover); the 2024 calibration windows are hourly
    assert ens["cqr"]["days_by_test_era"] == {"15-min": 90}
    assert s["calibration_era_by_fold"] == {
        "2025-01": "hourly",
        "2025-02": "hourly",
        "2025-03": "hourly",
    }
    assert set(ens["cqr"]["coverage_by_calibration_era"]) == {"hourly"}
    assert ens["raw"]["fold_coverage"][0] == pytest.approx(
        interval_coverage(folds[0].preds[MODEL], folds[0].test.y, QUANTILES)
    )
    assert "CQR (per hour of day)" in format_conformal_summary(s)


def test_cqr_changes_only_outer_levels_before_resort() -> None:
    folds = [_fake_quantile_fold(1, 5.0, 5.0)]
    f = folds[0]
    adjusted, q_hat = calibrate(f.val_preds[MODEL], f.val.y, f.preds[MODEL], QUANTILES, False)
    raw = f.preds[MODEL]
    unsorted = raw.copy()
    unsorted[..., 0] -= q_hat
    unsorted[..., 4] += q_hat
    assert adjusted == pytest.approx(np.sort(unsorted, axis=-1))
    if q_hat[0] >= 0:  # widening cannot cross the inner levels, so they are untouched
        assert adjusted[..., 1:4] == pytest.approx(raw[..., 1:4])


def test_conformal_flag_is_opt_in_on_quantile_eval_only() -> None:
    from minigpt.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["price-quantile-eval"]).conformal is False
    assert parser.parse_args(["price-quantile-eval", "--conformal"]).conformal is True
