# Protocol: dependence-robust test and a matched (bagged) ridge baseline

Written 2026-11-06, before any block-bootstrap or bagged-ridge number was computed.

## Claim under test

README, "Day-ahead electricity price forecasting": the 5-seed transformer ensemble **beats** ridge on lags
by 1.41 EUR/MWh pooled MAE (iid paired-bootstrap CI [-2.59, -0.28], DM-HLN p = 0.016, 365 days).
The README already says that the bootstrap treats days as exchangeable, that DM uses no autocorrelation
correction (h = 1), and that ridge is a single model while the transformer is a 5-seed ensemble. This
protocol tests whether the claim survives fixing those two issues.

## Data and models

No retraining, no downloads. The 12 existing fold checkpoints (`runs/price/<month>/ckpt.pt`) and the
cached prices (`data/raw/nl_dayahead`) are reused through `price-eval --folds 12`. The 365 test days are
pooled in chronological order (consecutive calendar days, months 2025-10 to 2026-09).
Per-day difference series: `d_t = MAE_t(transformer) - MAE_t(baseline)`, with `MAE_t` the mean absolute
error over the 24 hours of day t. Negative means the transformer is better.

## Change 1: moving-block bootstrap

- Resample the 365-day series `d` in overlapping blocks of length `b` (moving-block bootstrap, Kunsch 1989):
  `ceil(n / b)` block start indices drawn uniformly from `0 .. n - b`, blocks concatenated and truncated to n.
- Statistic: mean of the resampled series. CI: 2.5th and 97.5th percentiles of 10,000 resamples, seed 0.
- **Block length, fixed now by rule: b = 7 days.** Reason: price dynamics have a weekly cycle (weekday
  effects; the model's lookback is 7 days), so a block of one week keeps the dominant dependence inside a block.
  b = 7 is the only block length used by the decision rule.
- Sensitivity at b = 3 and b = 14 is reported and labelled **post-hoc**; it cannot change the decision.

## Change 2: bagged ridge baseline

- 5 members (to match the 5 transformer seeds). Member k (k = 0..4) is the existing multi-output ridge
  (`fit_ridge`) fitted on a bootstrap resample of the fold's training windows (rows drawn with replacement,
  same size, `numpy.random.default_rng(k)`).
- Alpha: the validation-tuned alpha of the fold's plain ridge (`fit_ridge_tuned`), shared by all members.
  The members are not re-tuned. Test days enter no fit or selection step.
- Forecast: mean of the 5 members. Reported in the per-fold and pooled tables as `bagged ridge (5)`.
- It is a baseline: "best baseline" is the lowest pooled MAE among the four baselines (two naive, ridge, bagged ridge).

## Comparisons and tests

For each of the two ridge variants (`ridge on lags`, `bagged ridge (5)`), on the per-day difference series:
iid paired-bootstrap CI (existing), block-bootstrap CI at b = 7 (new), and DM-HLN with h = 1 (existing).

## Decision rule (pre-registered)

The transformer "beats" the ridge baselines only if, for **both** `ridge on lags` and `bagged ridge (5)`,
**all three** hold, in the negative direction (transformer better): DM-HLN p < 0.05, the iid bootstrap CI
excludes 0, and the block-bootstrap (b = 7) CI excludes 0. If any of these fails for either comparison, in
particular if the b = 7 block-bootstrap CI against ridge includes 0, or the CI against bagged ridge includes 0,
or the DM p against bagged ridge is >= 0.05, the README wording changes from "beats" to
"no detectable difference". If all three hold in the positive direction the verdict is "loses to". The
code's verdict string follows the same rule.

## Reporting

Raw output of `uv run --offline minigpt price-eval --folds 12` is saved to
`runs/price/price-eval-block-bootstrap.txt` and `runs/price/report.json` is regenerated. The previous report
is kept as `runs/price/report_pre_block.json`; every previously reported number must reproduce from it.
Sensitivity (b = 3, 14) comes from a separate command whose output is saved next to it, labelled post-hoc.
When a CI includes 0 the result is written as "no detectable effect".

## Results (2026-11-06)

Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --offline minigpt price-eval --folds 12`
(raw output: `runs/price/price-eval-block-bootstrap.txt`, report: `runs/price/report.json`; both are under the
git-ignored `runs/`, so the table below is the committed copy). Re-running gave identical output.
Every number from the previous report (`runs/price/report_pre_block.json`) reproduced exactly.

Pooled MAE, EUR/MWh: ridge on lags 22.66, bagged ridge (5) 23.20, transformer 21.25.

| Baseline | Diff (transformer - baseline) | iid CI | block CI, b = 7 | DM-HLN p | Post-hoc block CI, b = 3 / 14 |
|---|---|---|---|---|---|
| ridge on lags | -1.41 | [-2.59, -0.28] | [-2.87, -0.20] | 0.0164 | [-2.73, -0.27] / [-2.79, -0.25] |
| bagged ridge (5) | -1.94 | [-3.28, -0.72] | [-3.65, -0.57] | 0.0030 | [-3.45, -0.70] / [-3.59, -0.62] |

Against the decision rule: both b = 7 block CIs exclude 0, both iid CIs exclude 0, both DM p < 0.05, all in
the negative direction. No trigger fired, so the README keeps "beats", now with the block CI. Bagging made
ridge worse (23.20 vs 22.66), so the matched baseline is not harder than plain ridge.

Limits that stay: DM-HLN at h = 1 has no autocorrelation correction; the 365 days are one market regime and
the folds are not independent; the single-seed transformers still show no detectable difference from ridge.
