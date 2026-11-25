# Toy DeepSeek-V4: which parts help at 1-6M parameters

Source: DeepSeek-V4 report (preview), sections 2.1-2.4. Every part is an opt-in flag; the
defaults reproduce the earlier dense model, so old checkpoints and all post-training commands work
unchanged. All runs on the Korea server (RTX 4090, CUDA), never on a laptop. Toy scale: the
results say which mechanisms still help at this size, not how V4 behaves at 1.6T parameters.

## What was built

| Report | Toy version | Flag |
|---|---|---|
| 2.1 DeepSeekMoE | routed experts (per-expert `nn.Linear`) plus shared experts, top-k routing on sqrt(softplus) affinities; aux-loss-free balancing: a per-expert bias (buffer, not trained) used only for selection, moved by `gamma * sign(load error)` each training step; small sequence-wise balance loss added to `loss` | `--n-experts 8 --n-shared-experts 1 --moe-top-k 2`, `--moe-bias-rate`, `--moe-balance-alpha` |
| 2.1 Multi-token prediction | one extra sequential module predicts token t+2 from hidden state t and the embedding of t+1, shares embedding and head; its loss (weight `mtp_weight`) is added only in training, never in `logits` | `--mtp-depth 1` |
| 2.2 mHC | 4 residual streams; the stream-mixing matrix is projected to doubly stochastic by Sinkhorn iterations | `--hc-streams 4` |
| 2.3.1 Compressed sparse attention | KV compressed every m tokens; a lightning indexer scores compressed blocks and keeps the top-k; each query attends to those plus a causal local window of raw tokens. `generate` caches raw keys/values, finished compressed blocks and the unfinished block's inputs; past the context length it recomputes | `--attn-type csa` (`--csa-block`, `--csa-topk`, `--csa-window`) |
| 2.4 Muon | Nesterov momentum 0.95, 8 + 2 hybrid Newton-Schulz steps, update RMS 0.18; on 2-D block weights only (experts included), AdamW on embeddings, head, router, `hc_` and 1-D params | `--optimizer muon` |

Skipped: heavily compressed attention (HCA; it only pays at 1M-token context), FP4 quantization-
aware training, hash routing, CSA's overlapping blocks, the low-rank indexer query, and the
indexer's separate distillation loss (here the indexer score is added to the attention logits, so
the LM loss trains it). At our 128-256-token contexts CSA cannot save compute; it is tested for
quality only.

Causality check of the trained CSA checkpoint `v4_rs_csa_s0` (context 128, block 4, top-4,
window 8): for every cut t = 1..127, in both `eval()` and `train()` mode (dropout off),
re-drawing tokens t..127 leaves logits 0..t-1 unchanged (max difference 0.0). The CSA gains below
are not a future-token leak.

## 1. Language-model pretraining (tiny-Shakespeare)

`minigpt pretrain --steps B --eval-interval 100 --final --seed S [flags]`, default 5.07M dense
model, B = 1000 and 2000 steps, seeds 0 and 1. `--final` scores the model as training left it,
not the best-val checkpoint (tiny-Shakespeare overfits; see the README section on the
selection-biased headline). Test and val are the two disjoint held-out halves. Verdict: both seeds
on the same side of dense and the smaller gap larger than the larger seed spread. "Train loss" is
plain cross-entropy of the final model on the training split in eval mode, without the MTP and
balance terms, so it is comparable across variants (the per-step `train` lines in the logs do
include those terms).

### 1000 steps (dense has not started to overfit: its best val is at the last step)

| Variant | Params (active) | Test loss s0 / s1 | Val loss s0 / s1 | Train loss | Val − train | Wall s | Verdict test / val |
|---|---|---|---|---|---|---|---|
| dense (AdamW) | 5.07M | 3.694 / 3.727 | 3.423 / 3.441 | 3.015 | 0.42 | 43 | baseline |
| + Muon | 5.07M | 3.712 / 3.715 | 3.386 / 3.420 | 2.852 | 0.55 | 67 | none / none |
| + MoE 8 × top-2 + 1 shared | 11.39M (5.08M) | 3.712 / 3.689 | 3.357 / 3.361 | 2.797 | 0.56 | 208 | none / better |
| + MTP depth 1 | 5.99M | 3.707 / 3.662 | 3.383 / 3.371 | 2.978 | 0.40 | 49 | none / better |
| + mHC 4 streams | 5.36M | 3.709 / 3.720 | 3.388 / 3.391 | 2.977 | 0.41 | 128 | none / better |
| + CSA | 5.27M | 3.665 / 3.669 | 3.292 / 3.289 | 2.617 | 0.67 | 90 | none / better |
| all V4 (Muon + MoE + MTP + mHC + CSA) | 12.84M (6.53M) | 4.010 / 4.027 | 3.561 / 3.557 | 1.502 | 2.06 | 540 | worse / worse |
| MoE, balancing off (control) | 11.39M (5.08M) | 3.722 / 3.701 | 3.377 / 3.368 | 2.839 | 0.53 | 230 | none / better |

### 2000 steps (dense overfits: its best val is at step 1100-1200)

| Variant | Test loss s0 / s1 | Val loss s0 / s1 | Train loss | Val − train | Verdict test / val |
|---|---|---|---|---|---|
| dense (AdamW) | 3.830 / 3.848 | 3.417 / 3.413 | 2.081 | 1.33 | baseline |
| + Muon | 4.001 / 4.011 | 3.601 / 3.593 | 1.645 | 1.95 | worse / worse |
| + MoE | 4.243 / 4.220 | 3.769 / 3.743 | 1.416 | 2.34 | worse / worse |
| + MTP | 3.761 / 3.752 | 3.364 / 3.353 | 2.121 | 1.24 | **better / better** |
| + mHC | 3.877 / 3.828 | 3.463 / 3.427 | 2.002 | 1.44 | none / none |
| + CSA | 4.120 / 4.127 | 3.659 / 3.675 | 1.431 | 2.24 | worse / worse |
| all V4 | 5.577 / 5.595 | 5.055 / 5.020 | 0.144 | 4.89 | worse / worse |
| MoE, balancing off | 4.169 / 4.152 | 3.683 / 3.677 | 1.537 | 2.14 | worse / worse |

Expert load in the last training step, max / mean tokens per expert (1.00 = perfectly even):

| Variant | 1000 steps, worst layer s0 / s1 | 2000 steps, worst layer s0 / s1 |
|---|---|---|
| MoE, aux-loss-free balancing | 1.14 / 1.12 | 1.10 / 1.12 |
| all V4 | 1.09 / 1.16 | 1.16 / 1.13 |
| MoE, balancing off | 2.72 / 2.79 | 2.28 / 2.15 |

Reading:

- Aux-loss-free balancing works as described: it keeps expert load within 16% of even, against
  2.2-2.8× without it, and costs no loss (1000-step test 3.700 vs 3.712 mean).
- On this 1 MB corpus every part that adds capacity or speeds up fitting (Muon, MoE, CSA, all
  combined) lowers train loss and widens the val-train gap; past the overfitting point it loses
  on held-out loss. All V4 together memorises the corpus (train 0.14 at 2000 steps).
- 1000 steps is the main budget. On the test half no single part differs from dense; only all V4
  together is worse. The val half sits about 0.27 below test and favours MoE, MTP, mHC and CSA;
  the halves disagree, so those val wins are split-specific and not claimed.
- At 2000 steps every variant is past its best-val point and the verdicts mostly measure how fast
  each variant memorises. MTP is the one that ends ahead (test 3.76 vs 3.84) with the smallest
  val-train gap (1.24 vs 1.33): it slowed overfitting. [INFERENCE] A regularising effect of the
  extra target, not a better LM: at 1000 steps it is no different on test. No run separates it
  from MTP's extra 0.9M parameters.

## 1b. Language-model pretraining on TinyStories

Section 1 mostly measures memorisation: tiny-Shakespeare has 414k training tokens and dense
overfits by step ~1100. This repeats it on a corpus that does not overfit at the same model size.

- Corpus: the first 50 MB of the TinyStories train file (`roneneldan/TinyStories`, cut after the
  last whole story), `minigpt pretrain --corpus tinystories`. 18.7M training tokens, 1.04M val and
  1.04M test tokens (same 90/10 cut and `split_val_test` as section 1).
- Tokenizer: the existing 1024-token byte-level BPE (trained on Shakespeare) is reused. It covers
  every byte, so nothing is out of vocabulary and decoding is lossless; the cost is compression,
  2.40 bytes/token on TinyStories against 3.10 for a BPE trained on it (5 MB sample). Reusing it
  keeps every model at the same size as section 1 and `runs/tokenizer.json` untouched. Losses are
  per token, so they are comparable across rows here but not with section 1.
- Budget: one dense curve (seed 0, 4000 steps, val every 250) fell at every evaluation, 1.5693
  → 1.5595 over the last 250, and took 158 s. 4000 steps is 32.8M tokens, 1.75 epochs. The best
  val step is the last step in all 16 runs below and val − train is 0.07-0.10 (dense on
  Shakespeare: 0.42 at 1000 steps, 1.33 at 2000), so these runs are not in the overfitting regime.

```
minigpt pretrain --corpus tinystories --steps 4000 --eval-interval 250 --final --seed S \
  --results-out runs/logs/ts_NAME_sS.json [flags]
```

Flags: dense none; Muon `--optimizer muon`; MoE `--n-experts 8 --n-shared-experts 1
--moe-top-k 2`; MTP `--mtp-depth 1`; mHC `--hc-streams 4`; CSA `--attn-type csa`; all V4 = Muon +
MoE + MTP + mHC + CSA; all V4 without Muon = the same with AdamW. Seeds 0 and 1, `--final` (the
model as training left it). Columns as in section 1; Δ test is the change of the two-seed mean
test loss against dense. Verdict: the two seed ranges do not overlap and the gap between them is
larger than the wider of the two ranges (stricter than section 1's paired rule, which it implies).

| Variant | Params (active) | Test loss s0 / s1 | Δ test | Val loss s0 / s1 | Train loss s0 / s1 | Val − train | Wall s s0 / s1 | Verdict test / val |
|---|---|---|---|---|---|---|---|---|
| dense (AdamW) | 5.07M | 1.418 / 1.420 | | 1.504 / 1.509 | 1.431 / 1.434 | 0.07 | 155 / 153 | baseline |
| + Muon | 5.07M | 1.369 / 1.370 | −0.050 (−3.5%) | 1.458 / 1.458 | 1.378 / 1.380 | 0.08 | 257 / 254 | better / better |
| + MoE 8 × top-2 + 1 shared | 11.39M (5.08M) | 1.345 / 1.352 | −0.071 (−5.0%) | 1.436 / 1.442 | 1.353 / 1.359 | 0.08 | 787 / 758 | better / better |
| + MTP depth 1 | 5.99M | 1.413 / 1.413 | −0.006 (−0.4%) | 1.500 / 1.499 | 1.427 / 1.428 | 0.07 | 179 / 178 | better / none |
| + mHC 4 streams | 5.36M | 1.396 / 1.397 | −0.023 (−1.6%) | 1.483 / 1.485 | 1.409 / 1.410 | 0.07 | 469 / 483 | better / better |
| + CSA | 5.27M | 1.436 / 1.429 | +0.014 (+1.0%) | 1.522 / 1.517 | 1.442 / 1.438 | 0.08 | 341 / 340 | worse / worse |
| all V4 (Muon + MoE + MTP + mHC + CSA) | 12.84M (6.53M) | 1.284 / 1.288 | −0.133 (−9.4%) | 1.376 / 1.381 | 1.274 / 1.278 | 0.10 | 2069 / 2102 | better / better |
| all V4 without Muon | 12.84M (6.53M) | 1.331 / 1.330 | −0.089 (−6.2%) | 1.421 / 1.419 | 1.330 / 1.328 | 0.09 | 1287 / 1300 | better / better |

Expert load in the last training step, worst layer, max / mean tokens per expert (1.00 = even),
seed 0 / seed 1: MoE 1.27 / 1.17; all V4 1.33 / 1.33; all V4 without Muon 1.20 / 1.21 (section 1,
1000 steps: 1.09-1.16).

Reading:

- Held-out loss, ranked by Δ test: all V4 (−0.133) > MoE (−0.071) > Muon (−0.050) > mHC (−0.023)
  > MTP (−0.006, test only); CSA is worse (+0.014). Test and val agree on every verdict except MTP,
  whose val gap (0.004) is below the seed spread (0.005). The MTP and mHC gains are small: the
  MTP gap (0.005) is about 2× the seed spread (0.002), so it is at the edge of the rule.
- Unlike section 1, parts that add capacity or speed up fitting (Muon, MoE) lower val and test loss
  here, with a val − train gap that does not grow (0.07 → 0.08). [INFERENCE] On Shakespeare their
  "worse" verdicts at 2000 steps were memorisation, not a worse model; on 1.75 epochs of a larger
  corpus the same mechanisms give a better fit that generalises.
- All V4 is the best row, and the effects are close to additive: the four architecture parts alone
  sum to −0.086 (measured, all V4 without Muon: −0.089); Muon alone is −0.050 and adds −0.044 on
  top of the architecture. [INFERENCE] The parts do not interfere at this size. CSA hurts on its
  own yet all V4 still wins; the part-by-part ablation (all minus one) was not run.
- Muon lowers test loss 3.5% at 1.7× dense's wall time. Its learning rate and AdamW's were not
  tuned, so this is the default-settings comparison only.
- MoE's gain is at matched active parameters (5.08M vs 5.07M) but with 2.2× the total; the
  size-matched controls below separate the two. Aux-loss-free balancing keeps the worst layer
  within 1.17-1.33× of even.

### Size-matched dense controls

Dense models widened with `--d-model` so their parameter count matches all V4's active (288:
6.36M vs 6.53M) or total parameters (416: 13.03M vs 12.84M), each with AdamW and with Muon, so
the optimizer is matched too. Same corpus, 4000 steps, `--final`, seeds 0 and 1, same rule.

| Variant | Params | Test loss s0 / s1 | Val loss s0 / s1 | Train loss s0 / s1 | Wall s s0 / s1 |
|---|---|---|---|---|---|
| dense `--d-model 288` (AdamW) | 6.36M | 1.391 / 1.387 | 1.481 / 1.478 | 1.403 / 1.400 | 123 / 138 |
| dense `--d-model 288`, Muon | 6.36M | 1.337 / 1.337 | 1.427 / 1.429 | 1.344 / 1.345 | 224 / 248 |
| dense `--d-model 416` (AdamW) | 13.03M | 1.289 / 1.289 | 1.378 / 1.379 | 1.292 / 1.292 | 147 / 139 |
| dense `--d-model 416`, Muon | 13.03M | 1.243 / 1.249 | 1.337 / 1.341 | 1.239 / 1.245 | 231 / 238 |
| all V4 without Muon (from above) | 12.84M (6.53M) | 1.331 / 1.330 | 1.421 / 1.419 | 1.330 / 1.328 | 1287 / 1300 |
| all V4 (from above) | 12.84M (6.53M) | 1.284 / 1.288 | 1.376 / 1.381 | 1.274 / 1.278 | 2069 / 2102 |

- At matched active parameters V4 wins: all V4 without Muon beats dense 288 by 0.056 (1.330-1.331
  vs 1.387-1.391), all V4 beats dense 288 + Muon by 0.049. Both pass the rule on test and val.
- At matched total parameters V4 loses: dense 416 beats all V4 without Muon by 0.041, and dense
  416 + Muon beats all V4 by 0.035 (1.243-1.249 vs 1.284-1.288). Both pass the rule on test and
  val. Plain AdamW dense 416 (1.289) is level with all V4 with Muon (1.284-1.288).
- So the 9.4% gain over the 5.07M dense model is mostly size. V4's parts buy loss per active
  parameter, not per stored parameter. [INFERENCE] At this scale the practical case for the
  architecture is cheaper compute per token, which this unfused implementation does not deliver:
  all V4 took 9× the wall time of dense 416 + Muon.

Limits of this table:

- Two seeds change initialisation and batch order only (spreads up to 0.007); they do not cover
  corpus slice, learning rate or budget. One slice (first 50 MB), one budget (4000 steps, 1.75
  epochs). The budget was chosen from a single dense seed-0 curve.
- Size is covered only by the two dense widths above, at one budget; FLOPs per token were not
  measured.
- Wall times come from a GPU shared with other jobs (it showed 98% utilisation from other users'
  processes before the first run); the two seeds agree within 3%, but compare ratios, not seconds.
  The toy MoE, mHC and CSA are unfused PyTorch loops: 13× dense for all V4.
- Logs: `runs/logs/ts_*` (gitignored); the dense curve is `ts_curve_dense_s0`.

## 2. Reasoning pipeline (verifiable addition)

`v4rp.sh`: `reason-pretrain` (1500 steps, default 0.84M dense) → `reason-sft --n 10000 --epochs 3`
→ `reason-eval --n 200`, seeds 0-3. "all" = `--optimizer muon --n-experts 8 --mtp-depth 1
--hc-streams 4 --attn-type csa`. Every Muon row uses Muon in both pretrain and SFT.

| Variant | Params (active) | Pretrain final val loss, s0-s3 | Greedy accuracy after SFT, 3 digits, s0 / s1 / s2 / s3 | 1 / 2 / 3 digits, mean of 4 seeds |
|---|---|---|---|---|
| dense | 0.84M | 1.721 / 1.733 / 1.737 / 1.726 | 0.08 / 0.31 / 0.46 / 0.71 | 0.88 / 0.66 / 0.39 |
| + MoE 8 × top-2 | 1.90M (0.85M) | 1.731 / 1.727 / 1.731 / 1.710 | 0.55 / 0.27 / 0.60 / 0.54 | 0.90 / 0.69 / 0.49 |
| + CSA | 0.91M | 1.516 / 1.514 / 1.533 / 1.516 | 0.65 / 0.39 / 0.51 / 0.86 | 0.97 / 0.88 / 0.60 |
| + MTP depth 1 | 1.07M | 1.734 / 1.740 / 1.659 / 1.735 | 0.46 / 0.35 / 0.78 / 0.52 | 0.91 / 0.72 / 0.53 |
| + mHC 4 streams | 0.94M | 1.734 / 1.747 / 1.742 / 1.737 | 0.19 / 0.40 / 0.16 / 0.10 | 0.76 / 0.44 / 0.21 |
| + Muon | 0.84M | 1.441 / 1.485 / 1.453 / 1.448 | 0.97 / 0.97 / 0.97 / 0.95 | 1.00 / 0.99 / 0.96 |
| all V4 | 2.31M (1.26M) | 1.205 / 1.211 / 1.205 / 1.206 | 0.98 / 0.99 / 0.99 / 0.86 | 1.00 / 0.99 / 0.95 |
| control: dense, `--d-model 176` | 1.56M | 1.684 / 1.730 / 1.672 / 1.640 | 0.49 / 0.56 / 0.62 / 0.12 | 0.87 / 0.65 / 0.45 |
| control: all V4 without Muon (AdamW in pretrain and SFT) | 2.31M (1.26M) | 1.494 / 1.545 / 1.517 / 1.547 | 0.90 / 0.79 / 0.63 / 0.79 | 0.99 / 0.92 / 0.78 |

Verdict rule for this table (4 seeds per row): a difference is measurable only when the two seed
ranges do not overlap and the gap between the ranges is larger than 0.05 for accuracy (about one
standard error of a difference at 200 tasks) or larger than the wider of the two ranges for val
loss. Per-seed comparisons are given as description only.

- Size does not explain the all-V4 result. A dense model larger than all V4's active parameters
  (1.56M vs 1.26M) reaches 0.45 at 3 digits, about the same as the 0.84M dense (0.39).
- The V4 architecture without Muon beats that larger dense model on pretrain val loss: ranges
  1.49-1.55 vs 1.64-1.73, gap 0.09, just above the wider range (0.09). On 3-digit accuracy it is
  higher on every seed (mean 0.78 vs 0.45), but the ranges only just clear each other (0.63-0.90
  vs 0.12-0.62, gap 0.005), so accuracy is not a measurable difference.
- Muon alone, on the unchanged dense model, is the largest single gain, measurable on both
  metrics: val loss ranges 1.44-1.49 vs 1.72-1.74 (gap 0.24, wider range 0.04); 3-digit accuracy
  ranges 0.95-0.97 vs 0.08-0.71 (gap 0.24). All V4 vs dense is measurable the same way (0.86-0.99
  vs 0.08-0.71, gap 0.16). Muon also removes the seed lottery: dense spans 0.63 at 3 digits,
  Muon 0.02.
- Muon alone matches all V4 on accuracy (0.96 vs 0.95); accuracy is near its ceiling for both. On
  val loss the two gains roughly add: Muon −0.27, architecture −0.20, both −0.52 (1.73 → 1.21).
- The AdamW settings in this table are the untuned defaults. The next section tunes AdamW in
  both stages; Muon in both stages still wins on accuracy.
- MTP alone: no measurable difference (val loss and accuracy ranges overlap dense; 3-digit
  accuracy higher on 3 seeds, lower on 1). mHC alone: no measurable difference (val loss higher
  on all 4 seeds by 0.005-0.014, ranges overlap; 3-digit accuracy higher on 2 seeds, lower on 2).
- Of the four architecture parts run alone (MoE, MTP, mHC, CSA), only CSA lowers pretrain val
  loss, and CSA alone equals the whole architecture without Muon (mean 1.52 vs 1.53).
  [INFERENCE] CSA carries the architecture gain on this task; combinations other than "all" were
  not run.
- CSA: pretrain loss measurably lower (1.51-1.53 vs 1.72-1.74); 3-digit accuracy higher on all 4
  seeds, but the ranges overlap (0.39-0.86 vs 0.08-0.71), so no measurable accuracy difference.
  [INFERENCE] The local window plus selected blocks may make digit alignment easier; not tested.
- MoE at matched active params: no measurable difference in loss or accuracy.
- The dense SFT result swings 0.08-0.71 across seeds at 3 digits, which is why 4 seeds were used.

### Muon against a tuned AdamW

Same dense 0.84M model and pipeline, seeds 0-3. Only the optimizer settings change. Defaults:
AdamW `--lr 1e-3` for 1500 pretrain steps, AdamW `--lr 5e-4` in SFT. Muon keeps its default
settings in every row. Accuracy cells are rounded half up from 200 tasks.

| Pretrain | SFT | Pretrain final val loss, s0-s3 | 3 digits, s0 / s1 / s2 / s3 | 1 / 2 / 3 digits, mean |
|---|---|---|---|---|
| AdamW (default) | AdamW (default) | 1.721 / 1.733 / 1.737 / 1.726 | 0.08 / 0.31 / 0.46 / 0.71 | 0.88 / 0.66 / 0.39 |
| AdamW `--lr 5e-4` | AdamW | 1.737 / 1.749 / 1.745 / 1.740 | 0.59 / 0.53 / 0.55 / 0.54 | 0.88 / 0.77 / 0.55 |
| AdamW `--lr 2e-3` | AdamW | 1.558 / 1.713 / 1.724 / 1.696 | 0.21 / 0.37 / 0.04 / 0.25 | 0.86 / 0.56 / 0.21 |
| AdamW `--lr 4e-3` | AdamW | 1.642 / 1.601 / 1.702 / 1.575 | 0.10 / 0.02 / 0.01 / 0.01 | 0.67 / 0.22 / 0.03 |
| AdamW `--lr 8e-3` | AdamW | 1.743 / 1.770 / 1.757 / 1.730 | 0.00 / 0.00 / 0.01 / 0.00 | 0.19 / 0.01 / 0.00 |
| AdamW `--steps 3000` | AdamW | 1.518 / 1.555 / 1.686 / 1.536 | 0.86 / 0.83 / 0.35 / 0.35 | 0.96 / 0.81 / 0.60 |
| AdamW | AdamW `--lr 2.5e-4` | as default | 0.02 / 0.00 / 0.08 / 0.04 | 0.70 / 0.23 / 0.03 |
| AdamW | AdamW `--lr 1e-3` | as default | 0.61 / 0.59 / 0.61 / 0.61 | 0.96 / 0.79 / 0.60 |
| AdamW | AdamW `--lr 2e-3` | as default | 0.43 / 0.18 / 0.53 / 0.04 | 0.89 / 0.61 / 0.29 |
| AdamW `--steps 3000` | AdamW `--lr 1e-3` | 1.518 / 1.555 / 1.686 / 1.536 | 0.79 / 0.59 / 0.59 / 0.48 | 0.90 / 0.85 / 0.61 |
| AdamW | Muon | as default | 0.75 / 0.82 / 0.73 / 0.82 | 0.99 / 0.94 / 0.78 |
| Muon | AdamW | 1.441 / 1.485 / 1.453 / 1.448 | 0.81 / 0.64 / 0.65 / 0.90 | 0.99 / 0.87 / 0.75 |
| Muon | Muon | 1.441 / 1.485 / 1.453 / 1.448 | 0.97 / 0.97 / 0.97 / 0.95 | 1.00 / 0.99 / 0.96 |

- Accuracy: Muon in both stages beats every AdamW setting tried. The best AdamW rows reach a
  3-digit mean of 0.60-0.61; the closest range is 0.35-0.86 (3000 steps), a gap of 0.09 to Muon's
  0.95-0.97. Against the combined row (3000 steps, SFT `--lr 1e-3`, 0.48-0.79) the gap is 0.17.
- Both stages contribute. Muon in SFT only (0.73-0.82) beats the best AdamW SFT rate on the same
  base (0.59-0.61, gap 0.12). Muon in both stages beats Muon in SFT only (gap 0.14) and Muon in
  pretrain only (0.64-0.90, gap 0.055, just above the 0.05 threshold).
- One-stage Muon is not measurably better than the best AdamW: both one-stage ranges overlap the
  3000-step rows.
- Pretrain val loss: every Muon seed is below every AdamW seed at every rate, and below AdamW
  with twice the steps (1.44-1.49 vs 1.52-1.69). The gap (0.03) is smaller than AdamW's own seed
  range (0.17), so by the rule above this is not a measurable loss difference against tuned
  AdamW. Against the default it is (previous section).
- Pretrain val loss does not predict accuracy after SFT. `--lr 4e-3` has lower val loss than the
  default on every seed and a 3-digit mean of 0.03; `--lr 5e-4` has the highest val loss of the
  working rows and the steadiest AdamW accuracy from a 1500-step base (0.53-0.59).
- The default SFT rate was a poor AdamW setting: `--lr 1e-3` lifts the mean 0.39 → 0.60 and
  shrinks the seed spread 0.63 → 0.02. Combining it with 3000 pretrain steps adds nothing (0.61).
- Not done: a joint grid over both AdamW rates, AdamW weight decay or schedule, and any tuning
  of Muon. More than 3000 AdamW pretrain steps were not tried.

### GRPO on the V4 parts

`grpo --steps 200 --seed S` from each seed's SFT checkpoint of the dense, MoE and CSA rows above
(seeds 0-3), scored with `reason-eval --n 200` on the same tasks as the SFT table. 3-digit
accuracy, SFT base → after GRPO:

| Variant | s0 | s1 | s2 | s3 | Mean |
|---|---|---|---|---|---|
| dense | 0.080 → 0.125 | 0.305 → 0.705 | 0.460 → 0.950 | 0.705 → 0.925 | 0.39 → 0.68 |
| + MoE | 0.545 → 0.945 | 0.270 → 0.720 | 0.595 → 0.930 | 0.540 → 0.910 | 0.49 → 0.88 |
| + CSA | 0.645 → 0.990 | 0.385 → 0.915 | 0.505 → 0.955 | 0.855 → 0.955 | 0.60 → 0.95 |

- GRPO raises 3-digit accuracy on all 12 runs. After it, the three ranges still overlap (dense
  0.13-0.95, MoE 0.72-0.95, CSA 0.92-0.99), so no architecture difference is measurable after RL
  either. CSA has the narrowest spread.
- The seed-0 dense base (0.08) is the one run GRPO barely moves. An earlier version of this page
  read that as "dense does not improve"; seeds 1-3 (bases 0.31-0.71) gain 0.22-0.49.
  [INFERENCE] With a 3-digit base near 0, 3-digit groups all score 0 and give no gradient.
- Within each architecture the weakest SFT seed is still the weakest after GRPO.
- The earlier seed-0 numbers (0.08 → 0.06, 0.54 → 0.94, 0.60 → 0.98) came from `grpo`'s built-in
  50-task eval, whose 2- and 3-digit tasks differ from the table's. That eval agrees in direction
  on 11 of 12 runs; the exception is dense seed 0. Logs: `runs/logs/v4_grpo_*`, `v4_gre_*`.

### Do experts specialise by operator?

`reason-pretrain --ops +,- --n-experts 8` → `reason-sft --ops +,-` at seeds 0-3, then the routing
of 300 operand pairs written once with `+` and once with `-` (the same pairs for every seed).
Total-variation distance between the expert shares in the layer where it is largest (a maximum
over 4 layers), against a permutation null (labels swapped at random, 300 times). Every value
below is above its null 95th percentile (0.01-0.09), with p at the permutation floor 0.003.

| Token | Pretrained (equations), s0 / s1 / s2 / s3 | After SFT (traces), s0 / s1 / s2 / s3 |
|---|---|---|
| operator | 0.50 / 0.50 / 0.57 / 0.56 | 0.26 / 0.49 / 0.08 / 0.58 |
| second operand | 0.74 / 0.37 / 0.43 / 0.43 | 0.18 / 0.21 / 0.39 / 0.73 |
| `=` | 0.63 / 0.64 / 0.32 / 0.53 | 0.10 / 0.27 / 0.10 / 0.29 |
| answer / trace | 0.17 / 0.37 / 0.31 / 0.44 | 0.13 / 0.08 / 0.08 / 0.06 |

- The operator token differs between the inputs, so its split is trivial. The second operand
  and `=` are the same tokens in both inputs; the answer and trace tokens are not (different
  digits), so that row mixes operator with content.
- Even on identical tokens the router reads a hidden state that has already attended to the
  operator, so TV measures operator-dependent routing, not a split by token identity.
- On identical tokens routing depends on the operator in every seed and both stages, by an
  amount that varies a lot with the seed: after SFT, second operand 0.18-0.73 and `=` 0.10-0.29.
  An earlier version of this page read seed 0 alone as "only weakly after SFT"; that holds for
  seed 0 and for the answer / trace row (0.06-0.13), not for the second operand.
  [INFERENCE] Part of the pretrained-to-SFT drop may come from scoring different text
  (equations vs traces); not separated.
- The specialisation does not buy accuracy. 3-digit accuracy after SFT (`reason-eval --n 200
  --ops +,-`): MoE 0.105 / 0.235 / 0.180 / 0.050, dense 0.115 / 0.075 / 0.050 / 0.325, means 0.142
  and 0.141; the ranges overlap. Logs: `runs/logs/v4_*_ab{moe,dense}_s*`, `v4_route_abmoe_*`.

## 3. Multi-teacher distillation (MOPD) on the V4 model

`opd` ran unchanged on all-V4 checkpoints (2.31M parameters), with the settings of dense E1 in
[`consolidation.md`](consolidation.md): 500 steps, 8 tasks per step, `--lr 3e-4`, `--ops +,-`,
estimator `full`, seeds 0 and 1.

- Teachers (seed 0): the `+` teacher is `v4_rs_all_s0` plus 200 GRPO steps; the `-` teacher is
  `reason-pretrain --ops -` → `reason-sft --ops -` → 200 GRPO steps. Greedy accuracy at 1 / 2 / 3
  digits (n = 200): `+` 1.00 / 1.00 / 1.00, `-` 1.00 / 1.00 / 0.99; each scores at most 0.06 on
  the other operator.
- Student: `reason-pretrain --ops +,-` → `reason-sft --ops +,- --epochs 1 --optimizer muon`. The
  recipe's 20,000 traces give a student that is already strong on V4 (`+` 1.00 / 1.00 / 0.78), so
  the trace count was swept (200 to 20,000) and fixed by a rule set before the distillation runs:
  mean 1-digit accuracy closest to the dense student's 0.31. That gave 3,000 traces: `+` 0.095 /
  0.010 / 0.005, `-` 0.310 / 0.045 / 0.000 (n = 200).
- Control: the same V4 student distilled on-policy from the dense teachers (`runs/grpo`,
  `runs/sub_grpo`).

Final-step student, greedy accuracy at 1 / 2 / 3 digits (n = 200). KL is the per-token reverse KL
of the student's greedy rollouts against the run's own teachers on 200 held-out tasks.

| Run | Seed | `+` | `-` | KL start → end | Wall (s) |
|---|---|---|---|---|---|
| on-policy, V4 teachers | 0 | 0.405 / 0.015 / 0.000 | 0.745 / 0.090 / 0.000 | 1.296 → 0.801 | 23,468 |
| on-policy, V4 teachers | 1 | 0.280 / 0.005 / 0.000 | 0.765 / 0.030 / 0.000 | 1.320 → 0.875 | 21,443 |
| off-policy, V4 teachers | 0 | 0.840 / 0.305 / 0.070 | 0.950 / 0.305 / 0.050 | 1.296 → 0.437 | 22,426 |
| off-policy, V4 teachers | 1 | 0.660 / 0.235 / 0.020 | 0.885 / 0.380 / 0.025 | 1.320 → 0.480 | 22,676 |
| control: on-policy, dense teachers | 0 | 0.490 / 0.010 / 0.000 | 0.685 / 0.035 / 0.005 | 1.290 → 0.695 | 23,166 |
| control: on-policy, dense teachers | 1 | 0.320 / 0.035 / 0.000 | 0.755 / 0.065 / 0.010 | 1.288 → 0.799 | 20,890 |

Same verdict rule as section 2 (seed ranges do not overlap; gap above 0.05 for accuracy, above
the wider range for KL), here with 2 seeds per row.

- On-policy distillation moves the V4 student, but only at 1 digit: `+` 0.095 → 0.28-0.41, `-`
  0.31 → 0.75-0.77, KL down by 0.45-0.50. 2- and 3-digit accuracy stays within 0.05 of the start.
- Off-policy beats on-policy on both seeds: 1- and 2-digit accuracy on both operators (gaps
  0.12-0.26) and KL (0.44-0.48 vs 0.80-0.88). 3 digits: no measurable difference.
- V4 teachers vs dense teachers, same student: no measurable accuracy difference. The weak
  on-policy result follows the student, not the teachers.
- Against dense E1 (n = 50 per cell, `consolidation.md`): the dense student gains more on `+`
  (1 digit 0.68 / 0.68 vs 0.24-0.32; 2 digits 0.12-0.20 vs 0.02); `-` overlaps. The relative KL
  drop is similar (dense 38-47%, V4 34-38%); absolute KL is not comparable.
- On-policy KL first rises (1.3 → 1.8-2.3 at step 100) in all four on-policy runs, then falls and
  is still falling at step 500. [INFERENCE] `--lr 3e-4` with a fresh AdamW is too large a step
  for a student trained with Muon; not tested (no rate sweep, no `--optimizer muon` on `opd`,
  no longer run). "On-policy is weak on V4" holds for these settings only.
- Cost: 5.8-6.5 h per run, about 29× dense E1 (726-785 s), on shared GPUs. The control costs the
  same, so the student's rollouts dominate: a profile put them at 96% of a train step. A CSA
  decode cache sped generation up only 1.15-1.4×, and batching rollouts by prompt length gave
  1.8-3.6× more train steps per second and about 5× faster greedy eval (`runs/logs/csacache_*`).
  Dense E1 also generated one task at a time, so these recover only part of the 29×; the rest is
  the higher cost of each V4 forward pass, not measured separately. The runs in this table
  predate both changes.
- Limits: one teacher pair and one student (seed 0), so the two seeds cover distillation noise
  only. The student is 2.7× the dense student and matched to it on 1-digit accuracy only. Each
  seed uses its own held-out tasks for KL. Logs: `runs/logs/v4m_*`.

## Verdicts

| Part | Small-scale verdict |
|---|---|
| Aux-loss-free balancing | works: even load, no loss cost |
| MoE (matched active params) | TinyStories: test loss −0.071 (5.0%) on both seeds, at 2.2× the total parameters. Shakespeare: overfits faster. Equations: no measurable gain before or after GRPO (4 seeds); routing on identical tokens depends on the operator on all 4 seeds (TV 0.10-0.73 after SFT) without buying accuracy |
| Multi-token prediction | TinyStories: test −0.006, at the edge of the rule, no difference on val. Shakespeare: no difference at 1000 steps; at 2000 steps it slows overfitting (test 3.76 vs 3.84, smallest gap). Equations: no measurable difference |
| mHC | TinyStories: small gain, test −0.023 (1.6%) on both seeds. No measurable difference on Shakespeare or equations |
| Compressed sparse attention | lower pretrain loss on equations; slightly worse on TinyStories (test +0.014); overfits Shakespeare faster; no speed gain at this context |
| Muon | the largest single gain on equations: alone it lowers val loss 1.73 → 1.46 and lifts 3-digit accuracy 0.39 → 0.96 on all 4 seeds, with a seed spread of 0.02 instead of 0.63. Tuning AdamW (pretrain rate, SFT rate, 2× steps) reaches at most 0.61, so the accuracy gain is not a default-settings artefact; it needs Muon in both stages (one stage: 0.75-0.78). The val-loss gain over tuned AdamW is within AdamW's seed range. TinyStories: test −0.050 (3.5%) at default rates. On Shakespeare it fits faster and overfits sooner (no held-out gain at 1000 steps) |
| All V4 together | lowest equation val loss (1.21); accuracy equal to Muon alone (0.95 vs 0.96). A larger dense control (1.56M) reaches only 0.45, so size does not explain it; without Muon it reaches 0.78. TinyStories: beats dense models with as many parameters as its active ones (6.36M, by 0.05-0.06 test loss) but loses to dense models with as many as its total (13.03M, by 0.035-0.041), with the optimizer matched; 9× their wall time here. Memorises Shakespeare |
| MOPD on all V4 | runs unchanged. On-policy lifts only 1-digit accuracy in 500 steps; off-policy is better on both seeds; V4 and dense teachers give the same result. 29× the dense wall time (no KV cache in CSA) |

Cost: the toy MoE, mHC and CSA are plain PyTorch loops, not fused kernels, so wall time is 2-13×
the dense model; that measures this implementation, not the architecture.

## Limits

- 2 seeds for pretraining and MOPD, 4 for the reasoning pipeline, GRPO and routing.
- Two corpora (Shakespeare, 414k training tokens, where overfitting dominates; a 50 MB TinyStories
  slice, 18.7M tokens, one step budget) and one toy task.
- On TinyStories size is covered by two dense widths at one budget; FLOPs per token were not
  measured. Shakespeare has no size-matched control.
- AdamW was tuned one setting at a time (pretrain rate, SFT rate, steps) plus one combined row,
  not on a joint grid; Muon was not tuned at all.
- Logs: `runs/logs/v4_*` (gitignored).
