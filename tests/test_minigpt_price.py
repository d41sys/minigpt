"""Offline tests for the day-ahead price forecaster: windowing, leakage, metrics, DM test."""

import itertools
import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
import torch

from minigpt.price_data import (
    TZ,
    DayTable,
    HourlySeries,
    chunk_ranges,
    daily_table,
    fold_months,
    last_full_month,
    make_windows,
    month_bounds,
    split_windows,
    to_hourly,
)
from minigpt.price_eval import (
    BAGGED,
    BASELINES,
    MODEL,
    FoldResult,
    betainc,
    block_bootstrap,
    diebold_mariano_hln,
    evaluate_fold,
    mae,
    paired_bootstrap,
    resolution_flag,
    rmse,
    smape,
    student_t_two_sided_p,
    summarize,
)
from minigpt.price_model import (
    PriceConfig,
    PriceTransformer,
    TrainConfig,
    build_features,
    fit_bagged_ridge,
    fit_ridge,
    seasonal_naive,
    train_ensemble,
    weekly_naive,
)


def synthetic_table(start: date, n_days: int, seed: int = 0) -> DayTable:
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    rows = []
    dates = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        base = 80 + 40 * np.sin(2 * np.pi * (hours - 6) / 24) + (15 if d.weekday() < 5 else -10)
        rows.append(base + rng.normal(0, 5, 24))
        dates.append(d)
    return DayTable(dates=dates, prices=np.asarray(rows), dst_adjusted=0)


def local_midnight_unix(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=TZ).timestamp())


def hourly_unix_series(first: date, last: date) -> HourlySeries:
    """Hourly series whose price equals its own unix start time (to prove timestamps)."""
    start = local_midnight_unix(first)
    end = local_midnight_unix(last + timedelta(days=1))
    hours = np.arange(start, end, 3600, dtype=np.int64)
    return HourlySeries(hours, hours.astype(np.float64), 0, None)


# ------------------------------------------------------------------ aggregation / DST / gaps


def test_chunk_ranges_cover_exactly() -> None:
    chunks = chunk_ranges(date(2024, 1, 1), date(2024, 4, 10), chunk_days=40)
    assert chunks[0][0] == date(2024, 1, 1) and chunks[-1][1] == date(2024, 4, 10)
    for (_, e), (s, _) in itertools.pairwise(chunks):
        assert s == e + timedelta(days=1)
    assert all((e - s).days + 1 <= 40 for s, e in chunks)


def test_to_hourly_averages_quarter_hours_and_drops_partial_hours() -> None:
    # hourly regime: hours at 0h, 1h; 15-min regime from 2h with a missing quarter in hour 3
    unix = np.array(
        [0, 3600, 7200, 7200 + 900, 7200 + 1800, 7200 + 2700, 10800, 10800 + 900, 10800 + 1800]
    )
    price = np.array([10.0, 20.0, 1.0, 2.0, 3.0, 6.0, 4.0, 4.0, 4.0])
    s = to_hourly(unix, price)
    assert s.hours.tolist() == [0, 3600, 7200]
    assert s.price.tolist() == [10.0, 20.0, 3.0]  # mean(1, 2, 3, 6)
    assert s.partial_hours_dropped == 1
    assert s.quarter_hour_from == 7200


def test_to_hourly_null_prices_make_gaps() -> None:
    s = to_hourly(np.array([0, 3600]), np.array([5.0, np.nan]))
    assert s.hours.tolist() == [0] and s.partial_hours_dropped == 1


def test_daily_table_dst_days() -> None:
    series = hourly_unix_series(date(2025, 3, 29), date(2025, 3, 31))
    t = daily_table(series, date(2025, 3, 29), date(2025, 3, 31))
    spring = t.prices[1]  # 2025-03-30 has 23 hours; local 02:00 does not exist
    assert np.isfinite(spring).all()
    assert spring[2] == (spring[1] + spring[3]) / 2
    assert t.dst_adjusted == 1

    series = hourly_unix_series(date(2025, 10, 25), date(2025, 10, 27))
    t = daily_table(series, date(2025, 10, 25), date(2025, 10, 27))
    fall = t.prices[1]  # 2025-10-26 has 25 hours; 02:00 occurs twice
    cest_2am = int(datetime(2025, 10, 26, 2, tzinfo=TZ, fold=0).timestamp())
    assert fall[2] == cest_2am + 1800  # average of the two 02:00 hours, 3600 s apart
    assert fall[3] == int(datetime(2025, 10, 26, 3, tzinfo=TZ).timestamp())
    assert t.dst_adjusted == 1


def test_gap_invalidates_only_windows_that_touch_it() -> None:
    table = synthetic_table(date(2025, 1, 1), 30)
    table.prices[15, 7] = np.nan  # day index 15 = 2025-01-16
    w = make_windows(table)
    dates = set(w.dates)
    assert date(2025, 1, 16) not in dates  # target has a hole
    for k in range(1, 8):  # the next 7 days use it as history
        assert date(2025, 1, 16) + timedelta(days=k) not in dates
    assert date(2025, 1, 24) in dates and date(2025, 1, 15) in dates
    assert np.isfinite(w.x).all() and np.isfinite(w.y).all()


# ------------------------------------------------------------------ windowing and leakage


def test_window_contents() -> None:
    table = synthetic_table(date(2025, 1, 1), 20)
    w = make_windows(table)
    assert w.dates[0] == date(2025, 1, 8)  # needs 7 days of history
    i = 10
    assert w.dates[i - 7] == table.dates[i]
    np.testing.assert_array_equal(w.x[i - 7], table.prices[i - 7 : i].reshape(-1))
    np.testing.assert_array_equal(w.y[i - 7], table.prices[i])


def test_inputs_never_reach_target_day() -> None:
    # price == unix time, so every input value is a timestamp we can compare to D 00:00 local
    first, last = date(2025, 5, 1), date(2025, 6, 30)
    table = daily_table(hourly_unix_series(first, last), first, last)
    w = make_windows(table)
    assert len(w) > 0
    for d, x, y in zip(w.dates, w.x, w.y, strict=True):
        midnight = local_midnight_unix(d)
        assert x.max() < midnight  # nothing at or after D 00:00
        assert x.max() >= midnight - 3600  # but D-1 23:00 (public at D-1 12:00) is used
        assert y.min() >= midnight and y.max() < midnight + 24 * 3600  # target is exactly day D


def test_future_values_cannot_change_inputs() -> None:
    table = synthetic_table(date(2025, 1, 1), 40)
    k = 25
    base = make_windows(table)
    corrupted = DayTable(table.dates, table.prices.copy(), 0)
    corrupted.prices[k:] = 1e9  # everything from day k onward is garbage
    alt = make_windows(corrupted)
    by_date_base = dict(zip(base.dates, zip(base.x, base.y, strict=True), strict=True))
    by_date_alt = dict(zip(alt.dates, zip(alt.x, alt.y, strict=True), strict=True))
    for d, (x, y) in by_date_base.items():
        if d > table.dates[k]:
            continue
        np.testing.assert_array_equal(by_date_alt[d][0], x)  # inputs identical up to day k
        if d < table.dates[k]:
            np.testing.assert_array_equal(by_date_alt[d][1], y)
    # features see only x: changing a target cannot change them
    f1 = build_features(base.x, base.dates)
    f2 = build_features(base.x.copy(), base.dates)
    assert torch.equal(f1, f2)
    assert float(f1[:, -24:, 0].abs().sum()) == 0.0  # query tokens carry no price


def test_model_output_shape_and_history_dependence() -> None:
    table = synthetic_table(date(2025, 1, 1), 20)
    w = make_windows(table)
    feats = build_features(w.x, w.dates)
    torch.manual_seed(0)
    model = PriceTransformer(PriceConfig()).eval()
    out = model(feats)
    assert out.shape == (len(w), 24)
    changed = feats.clone()
    changed[:, :24, 0] += 5.0  # D-7 only
    assert not torch.allclose(model(changed), out)  # forecast does depend on history


def test_split_is_chronological_and_disjoint() -> None:
    table = synthetic_table(date(2025, 1, 1), 200)
    w = make_windows(table)
    s = split_windows(w, (2025, 6), val_days=28)
    assert max(s.train.dates) < min(s.val.dates) <= max(s.val.dates) < min(s.test.dates)
    assert min(s.test.dates) == date(2025, 6, 1) and max(s.test.dates) == date(2025, 6, 30)
    assert min(s.val.dates) == date(2025, 5, 4) and max(s.train.dates) == date(2025, 5, 3)
    assert len(s.val) == 28


def test_calendar_helpers() -> None:
    assert month_bounds(2024, 2) == (date(2024, 2, 1), date(2024, 2, 29))
    assert month_bounds(2025, 12)[1] == date(2025, 12, 31)
    assert last_full_month(date(2026, 10, 15)) == (2026, 9)
    assert last_full_month(date(2026, 9, 30)) == (2026, 9)
    assert last_full_month(date(2026, 1, 10)) == (2025, 12)
    assert fold_months((2026, 2), 3) == [(2025, 12), (2026, 1), (2026, 2)]
    assert resolution_flag((2025, 9), date(2025, 10, 1)) == "hourly"
    assert resolution_flag((2025, 10), date(2025, 10, 1)) == "15-min"
    assert resolution_flag((2025, 10), date(2025, 10, 15)) == "mixed"


# ------------------------------------------------------------------ baselines and metrics


def test_naive_baselines_pick_the_right_day() -> None:
    table = synthetic_table(date(2025, 1, 1), 20)
    w = make_windows(table)
    np.testing.assert_array_equal(seasonal_naive(w)[0], table.prices[6])  # D-1 for index 7
    np.testing.assert_array_equal(weekly_naive(w)[0], table.prices[0])  # D-7 for index 7


def test_ridge_learns_a_linear_rule() -> None:
    table = synthetic_table(date(2025, 1, 1), 120, seed=1)
    w = make_windows(table)
    model = fit_ridge(w, alpha=1e-6)
    # in-sample the 175-feature fit is near exact on 113 samples
    assert mae(model.predict(w), w.y) < 1e-3
    strong = fit_ridge(w, alpha=1e4)
    assert mae(strong.predict(w), w.y) > mae(model.predict(w), w.y)


def test_bagged_ridge_averages_bootstrap_members() -> None:
    w = make_windows(synthetic_table(date(2025, 1, 1), 120, seed=1))
    bag = fit_bagged_ridge(w, alpha=10.0)
    assert len(bag.members) == 5
    # members see different resamples, so they differ from each other and from the plain fit
    assert not np.allclose(bag.members[0].coef, bag.members[1].coef)
    assert not np.allclose(bag.members[0].coef, fit_ridge(w, 10.0).coef)
    assert np.allclose(bag.predict(w), np.mean([m.predict(w) for m in bag.members], axis=0))
    again = fit_bagged_ridge(w, alpha=10.0)  # fixed seeds: deterministic
    assert np.array_equal(bag.predict(w), again.predict(w))


def test_metrics_hand_computed() -> None:
    y = np.array([[100.0, 200.0]])
    p = np.array([[110.0, 180.0]])
    assert mae(p, y) == 15.0
    assert rmse(p, y) == pytest.approx(math.sqrt((100 + 400) / 2))
    # 200*10/210 and 200*20/380
    assert smape(p, y) == pytest.approx((200 * 10 / 210 + 200 * 20 / 380) / 2)


def test_smape_breaks_near_zero_and_negative() -> None:
    assert smape(np.array([1.0]), np.array([-1.0])) == pytest.approx(200.0)  # sign flip = max
    assert smape(np.array([0.1]), np.array([0.0])) == pytest.approx(200.0)  # tiny absolute error


def test_paired_bootstrap() -> None:
    mean, lo, hi = paired_bootstrap(np.full(30, -2.0), n_boot=200)
    assert mean == lo == hi == -2.0
    rng = np.random.default_rng(0)
    mean, lo, hi = paired_bootstrap(rng.normal(0.0, 1.0, 400), n_boot=2000)
    assert lo < 0 < hi


def test_block_bootstrap_boundaries_and_dependence() -> None:
    mean, lo, hi = block_bootstrap(np.full(10, -2.0), block=4, n_boot=200)  # 10 is not 3 * 4
    assert mean == lo == hi == pytest.approx(-2.0)
    d = np.random.default_rng(0).normal(size=50)
    mean, lo, hi = block_bootstrap(d, block=50, n_boot=200)  # one block: every resample is d
    assert lo == hi == pytest.approx(mean)
    for bad in (0, 51):
        with pytest.raises(ValueError):
            block_bootstrap(d, block=bad)
    # AR(1) with strong positive dependence: blocks give a wider CI than i.i.d. resampling
    rng = np.random.default_rng(1)
    e = rng.normal(size=400)
    ar = np.zeros(400)
    for t in range(1, 400):
        ar[t] = 0.9 * ar[t - 1] + e[t]
    _, iid_lo, iid_hi = block_bootstrap(ar, block=1, n_boot=2000)
    _, blk_lo, blk_hi = block_bootstrap(ar, block=20, n_boot=2000)
    assert blk_hi - blk_lo > 1.5 * (iid_hi - iid_lo)


def test_diebold_mariano_hln_hand_computed() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # h=1: gamma0 = 2, V = 2/5, DM = 3/sqrt(0.4); HLN factor sqrt(4/5) => 3/sqrt(0.5) = sqrt(18)
    stat, p = diebold_mariano_hln(d, h=1)
    assert stat == pytest.approx(math.sqrt(18.0))
    # two-sided p for t = sqrt(18) on 4 df, from numerical integration of the t pdf
    assert p == pytest.approx(0.0132356, abs=1e-6)
    # h=2: gamma1 = 0.8, V = (2 + 1.6)/5 = 0.72, DM = 3/sqrt(0.72); factor sqrt(2.4/5) => sqrt(6)
    stat2, _ = diebold_mariano_hln(d, h=2)
    assert stat2 == pytest.approx(math.sqrt(6.0))
    assert diebold_mariano_hln(-d, h=1)[0] == pytest.approx(-math.sqrt(18.0))


def test_t_distribution_helpers() -> None:
    assert betainc(2.0, 2.0, 0.5) == pytest.approx(0.5)
    assert betainc(1.0, 1.0, 0.3) == pytest.approx(0.3)
    assert student_t_two_sided_p(1.0, 1.0) == pytest.approx(0.5)  # Cauchy: P(|T|>1) = 1/2
    assert student_t_two_sided_p(0.0, 7.0) == pytest.approx(1.0)


# ------------------------------------------------------------------ training and folds


def test_train_ensemble_and_evaluate_fold_end_to_end() -> None:
    table = synthetic_table(date(2025, 1, 1), 200)
    w = make_windows(table)
    split = split_windows(w, (2025, 6), 28)
    device = torch.device("cpu")
    cfg = PriceConfig(d_model=16, num_layers=1, num_heads=2)
    ckpt = train_ensemble(split.train, split.val, cfg, TrainConfig(epochs=3, seeds=2), device)
    ckpt.update(val_days=28)
    res = evaluate_fold(w, (2025, 6), ckpt, device, cutover=None)
    assert res.preds[MODEL].shape == res.test.y.shape
    assert len(res.seed_mae) == 2
    assert set(res.preds) == {*BASELINES, MODEL}
    assert res.train_days_quarter_hour == 0


def _fake_fold(
    month: tuple[int, int],
    model_noise: float,
    base_noise: float,
    bagged_noise: float | None = None,
) -> FoldResult:
    rng = np.random.default_rng(month[1])
    table = synthetic_table(date(2025, month[1], 1), 30, seed=month[1])
    y = table.prices
    test = make_windows(table).subset(np.zeros(0, dtype=bool))  # empty shell for dates
    test.dates = table.dates
    test.y = y
    test.x = y
    preds = {name: y + rng.normal(0, base_noise, y.shape) for name in BASELINES}
    if bagged_noise is not None:
        preds[BAGGED] = y + rng.normal(0, bagged_noise, y.shape)
    preds[MODEL] = y + rng.normal(0, model_noise, y.shape)
    return FoldResult(month, test, preds, 1.0, 100, 0, [1.0], [preds[MODEL]], "hourly")


def test_summarize_verdicts() -> None:
    better = summarize([_fake_fold((2025, m), 1.0, 20.0) for m in (1, 2, 3)], n_boot=500)
    assert better["verdict"] == "beats"
    assert better["best_baseline"] in BASELINES
    worse = summarize([_fake_fold((2025, m), 20.0, 1.0) for m in (1, 2, 3)], n_boot=500)
    assert worse["verdict"] == "loses to"
    same = summarize([_fake_fold((2025, m), 5.0, 5.0) for m in (1, 2, 3)], n_boot=500)
    assert same["verdict"] == "no detectable difference"
    # beating plain ridge is not enough: the matched (bagged) ridge must be beaten too
    matched = summarize(
        [_fake_fold((2025, m), 1.0, 20.0, bagged_noise=0.5) for m in (1, 2, 3)], n_boot=500
    )
    assert matched["verdict"] == "no detectable difference"
    assert matched["best_baseline"] == BAGGED
    assert set(matched["ridge_comparisons"]) == {"ridge on lags", BAGGED}
    assert matched["ridge_comparisons"]["ridge on lags"]["mean"] < -1.0
    assert same["days"] == 90 and len(same["per_fold"]) == 3


def test_fixture_timestamps_are_utc_hours() -> None:
    # sanity for the helper used above: local midnight in winter is 23:00 UTC the day before
    mid = datetime.fromtimestamp(local_midnight_unix(date(2025, 1, 10)), UTC)
    assert (mid.hour, mid.day) == (23, 9)
