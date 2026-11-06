# Protocol: paired bootstrap of DPO vs SFT token F1 on all 100 held-out rows

Written 2026-11-06, before any n=100 token F1 was computed.

## Claim under test

README, "DPO": "No β improves on SFT". It rests on token F1 point values over the **first 50** of the
100 held-out Alpaca rows (`minigpt eval --n 50`, the default), with no confidence interval and no
multiple-comparison control. This protocol re-tests it on all 100 held-out rows, with paired CIs.

## Data and models

No retraining, no downloads. Held-out rows: `load_alpaca()[-SFT_HELDOUT:]` (100 rows, never trained on).
Reference model: `runs/sft/ckpt.pt`. Variants (all existing checkpoints, all trained from the SFT
checkpoint on the same 300 pairs in `runs/dpo/pairs.json` unless README says otherwise):

`runs/dpo`, `runs/dpo_b0.05`, `runs/dpo_b0.1`, `runs/dpo_b0.5`, `runs/fx_b0.05_a`, `runs/fx_b0.1_a`,
`runs/fx_b0.1_b`, `runs/fx_b0.1_s1`, `runs/fx_b0.1_s2`, `runs/fx_b0.5_a`.

That is 10 variants. No variant is dropped or added after scoring. A weight comparison (before scoring)
found no two of the 10 plus SFT bit-identical, so none is merged.

## Scoring

`sample_response`: greedy decoding (temperature 0), at most 60 new tokens, one model loaded at a time,
sequential, on the repo's default device. Per-row score `f1_i = token_f1(response_i, reference_i)`
(existing `token_f1`). Per-row difference `d_i = f1_i(variant) - f1_i(sft)`; positive means the variant is better.
Mean difference is the statistic.

## Test

- Paired bootstrap over rows: 10,000 resamples of the 100 row indices, with replacement (rows are
  independent prompts), seed 0, the same index draws for every variant. CI: 2.5th and 97.5th percentiles
  of the resampled mean of `d`.
- Two-sided bootstrap p-value: `p = min(1, 2 * min(P(mean* <= 0), P(mean* >= 0)))`, with the add-one
  correction `(count + 1) / (10000 + 1)`.
- Holm step-down at family-wise alpha = 0.05 across the 10 variants, applied to these p-values.

## Decision rule (pre-registered)

A variant is **different** from SFT only if its 95% paired-bootstrap CI excludes 0 **and** its Holm-adjusted
p < 0.05 across the 10 variants. Otherwise it is **no detectable difference**. "No β improves on SFT" is
kept in the README only if no variant is different in the positive direction; it is rewritten if any is.

## Reporting

- Primary: n = 100 rows, all 10 variants, mean F1, mean difference, 95% CI, raw p, Holm-adjusted p, verdict.
- The n = 50 point values (first 50 rows of the same run) are reported next to them for continuity with
  the old table. They carry no decision.
- Anything else (e.g. a per-β pooling, a median, a different resample count) is labelled **post-hoc** and
  cannot change the decision.
- Raw output (per-row F1, responses, resample summary) is saved under `runs/dpo/` like the other runs.
