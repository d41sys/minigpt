# Protocol: overfitting and selection bias in pretraining

Written 2026-11-06, before any run below was trained and before any test-half loss was computed.

## Problem

README headline "val loss 6.993 → 3.648, perplexity 40.27, best at step 800" has two defects:

1. **Selection bias.** The checkpoint is chosen by validation loss and the same validation loss is
   reported. The reported number is the minimum over 11 evaluations (every 200 steps plus step 1999),
   so it is optimistically biased.
2. **Overfitting.** Validation loss bottoms at step 800 of 2000 and rises to 4.11. The training set is
   414,007 tokens in non-overlapping windows (stride = context length 256 → 1,617 windows); 2000 steps ×
   32 × 256 tokens ≈ 39.6 epochs. Dropout is hardcoded at 0.1 in `cmd_pretrain`.

## Run-time check (stated before this commit)

One full run (2000 steps, `device=mps`) took 5 min 42 s in the README. That is under the 8-minute
threshold, so the full design below is kept: 3 variants × 3 seeds = 9 runs, one at a time, no background
launches, 0 dataloader workers, `uv run --offline`, no downloads.

## Data split (deterministic, no randomness)

The existing validation tokens (`shakespeare_splits` → last 10% of tiny-Shakespeare, 46,001 BPE tokens)
are cut into two contiguous halves with `h = len // 2 = 23,000`:

- **val** = `val_ids[:h]` (23,000 tokens). Used only for checkpoint selection.
- **test** = `val_ids[h:]` (23,001 tokens). Used only to report. Never used for selection or any design choice.

The train split (first 90%) is unchanged. Both halves are cut into all non-overlapping 256-token windows
(`stride = 256`, no windows dropped): 89 windows each. Loss = token-weighted mean cross-entropy over all
windows, dropout off.

The existing `runs/pretrain/ckpt.pt` was selected on the **full** validation set, which contains the test
half, so its test-half loss is selection-contaminated. It is reported only as a labelled legacy reference
and is never used in the decision. The decision uses fresh runs selected on the val half only.
`runs/pretrain/` is not overwritten (SFT, DPO and the spam classifier were trained from it).

## Variants (one factor changed at a time)

All share the CLI defaults: 2000 steps, batch 32, lr 1e-3, context 256, d_model 256, 6 layers, 8 heads,
eval every 200 steps, warmup steps/20, AdamW wd 0.1, grad clip 1.0.

| Name | Window sampling | Dropout |
|---|---|---|
| `base` | aligned: stride 256, shuffled each epoch | 0.1 |
| `randoff` | random offset: stride-1 windows, shuffled, drawn without replacement (each window start uniform over all offsets) | 0.1 |
| `drop02` | aligned | 0.2 |

The step budget (2000) is identical for all. The combination `randoff + drop 0.2` is not run.
Seeds: 0, 1, 2 for each variant (`seed_everything(seed)` seeds init, dropout, and shuffling).
Each run is named `pre_<variant>_s<seed>` under `runs/` (gitignored); per-run raw JSON (history, selected
step, val loss, test loss, per-window test losses) is copied to `docs/pretrain-regularization/`.

## Statistics

- Per run: the val-selected checkpoint (lowest val-half loss among the 11 evaluations; a checkpoint is saved
  only if it beats the step-0 loss) is evaluated on the test half. That test loss is the only outcome.
- Per seed `s` and variant `V`: `delta(V, s) = test(V, s) − test(base, s)`; negative means V is better.
- Per-seed 95% CI of `delta`: paired bootstrap over the 89 test windows (per-window loss differences,
  10,000 resamples, seed 0, existing `paired_bootstrap_test`). This captures test-set sampling noise only;
  training-seed noise is shown by the three per-seed deltas. A CI containing 0 is reported as
  **no detectable effect**.
- Baseline test loss is reported per seed and as the mean over seeds 0, 1, 2 (seed range shown). Perplexity
  `exp(loss)` is reported next to it.

## Decision rule (pre-registered, verbatim)

Adopt variant V as the default **only if, on every seed s ∈ {0, 1, 2}, `test(base, s) − test(V, s) ≥ 0.05`
and the per-seed 95% paired-bootstrap CI of `delta(V, s)` excludes 0 (upper bound < 0)**. If both variants
meet this, adopt the one with the lower mean test loss over the three seeds. Otherwise keep the existing
recipe (dropout 0.1, aligned windows) as the default.

Consequences:

- Adopt → the changed default is edited in `cmd_pretrain`; the other behaviour stays reachable by flag.
  SFT/DPO/spam tables were trained from the old checkpoint; they are **not retrained** and the README says so.
- Not adopt → defaults stay unchanged. The flags `--dropout` and `--sampling` remain, default-preserving.
- Either way the README headline is corrected to report the baseline **test** loss of the val-selected
  checkpoint (3-seed mean and range) and states that 3.648 / 40.27 was selection-biased.

## Reporting rules

- Report all 9 runs; no run is dropped, re-seeded or re-run after seeing its result (a crash with no
  result is re-run once with the same seed and stated).
- Anything beyond the above (combination variants, selection-step analysis, comparisons on val loss,
  pooled CIs) is labelled **post-hoc** and carries no decision.
- Caveats stated up front: the test half is 23k tokens (89 windows) from one contiguous passage; test and
  val halves are adjacent text, so the split is a within-corpus hold-out, not an independent corpus.
