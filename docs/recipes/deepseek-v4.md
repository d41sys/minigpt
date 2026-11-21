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
| 2.3.1 Compressed sparse attention | KV compressed every m tokens; a lightning indexer scores compressed blocks and keeps the top-k; each query attends to those plus a causal local window of raw tokens. No KV cache: `generate` recomputes | `--attn-type csa` (`--csa-block`, `--csa-topk`, `--csa-window`) |
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

GRPO, 200 steps from each seed-0 SFT checkpoint (one seed, so indicative only), 3-digit accuracy:
dense 0.08 → 0.06, MoE 0.54 → 0.94, CSA 0.60 → 0.98. RL needs a base that already solves some
tasks (zero-reward groups give no gradient); the dense seed-0 base was too weak.

### Do experts specialise by operator?

`reason-pretrain --ops +,- --n-experts 8` (seed 0), then routing of 300 operand pairs written once
with `+` and once with `-`. Total-variation distance between the expert shares, with a
permutation null (labels swapped at random, 300 times):

| Token | Pretrained, strongest layer | After SFT, strongest layer |
|---|---|---|
| operator | TV 0.50, null p95 0.06 (expert 7 takes 50% of `+` and 0% of `-`) | TV 0.26, null p95 0.05 |
| second operand | TV 0.74, null p95 0.09 | TV 0.18, null p95 0.03 |
| answer / trace | TV 0.17, null p95 0.03 | TV 0.13, null p95 0.02 |

The operator token itself differs between the two inputs, so its routing split is trivial, at
least in layer 0. The informative rows are the tokens that are the same in both inputs (second
operand, answer and trace): their routing differs above the permutation null (p = 0.003) but
after SFT only weakly (TV 0.13-0.18). The specialisation does not buy accuracy here: the
mixed-operator MoE and dense models score 0.11 and 0.12 at 3 digits after SFT (seed 0).

## Verdicts

| Part | Small-scale verdict |
|---|---|
| Aux-loss-free balancing | works: even load, no loss cost |
| MoE (matched active params) | no measurable gain; overfits a small corpus faster; weak operator-dependent routing on shared tokens |
| Multi-token prediction | Shakespeare: no difference at 1000 steps; at 2000 steps it slows overfitting (test 3.76 vs 3.84, smallest gap). Equations: no measurable difference |
| mHC | no measurable difference on either task |
| Compressed sparse attention | lower pretrain loss on equations; overfits Shakespeare faster; no speed gain at this context |
| Muon | the largest single gain on equations: alone it lowers val loss 1.73 → 1.46 and lifts 3-digit accuracy 0.39 → 0.96 on all 4 seeds, with a seed spread of 0.02 instead of 0.63. Tuning AdamW (pretrain rate, SFT rate, 2× steps) reaches at most 0.61, so the accuracy gain is not a default-settings artefact; it needs Muon in both stages (one stage: 0.75-0.78). The val-loss gain over tuned AdamW is within AdamW's seed range. On Shakespeare it fits faster and overfits sooner (no held-out gain at 1000 steps) |
| All V4 together | lowest equation val loss (1.21); accuracy equal to Muon alone (0.95 vs 0.96). A larger dense control (1.56M) reaches only 0.45, so size does not explain it; without Muon it reaches 0.78. Memorises Shakespeare |

Cost: the toy MoE, mHC and CSA are plain PyTorch loops, not fused kernels, so wall time is 2-13×
the dense model; that measures this implementation, not the architecture.

## Limits

- 2 seeds for pretraining, 4 for the reasoning pipeline, 1 for GRPO and routing.
- One tiny corpus (414k training tokens) and one toy task; overfitting dominates the LM results.
- "All V4" is not compute-matched on Shakespeare (6.5M vs 5.1M active). On equations the larger
  dense control (1.56M) covers size; the control has no Shakespeare counterpart.
- AdamW was tuned one setting at a time (pretrain rate, SFT rate, steps) plus one combined row,
  not on a joint grid; Muon was not tuned at all.
- Logs: `runs/logs/v4_*` (gitignored).
