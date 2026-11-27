# Merging specialists: task arithmetic vs multi-teacher OPD vs rejection-sampled SFT

Question: the 2026 literature says multi-teacher on-policy distillation (MOPD) costs 15-23x more
than a weight merge or tuned SFT for about the same result. Does that hold on the repo's
two-op task (`+` and `-`, 1-3 digits)? Everything ran on the Korea server, GPU 1, shared with
other users, so wall times are about 2x slower than an idle card and noisy (see the two GRPO
times below). Toy scale (842,752 parameters); no claim beyond that.

## What was added

- `minigpt merge --base B --expert E1 --expert E2 [--scale L] --out O`:
  `theta = theta_B + L * sum_i (theta_Ei - theta_B)`. Default `L = 1/n` (average of the deltas).
  Refuses experts whose config differs from the base's. Code: `task_arithmetic` in
  `src/minigpt/merge.py`; tests in `tests/test_minigpt_merge.py` (L=0 gives the base, one expert at
  L=1 gives the expert, default scale averages, mismatched configs refused). The LM head is tied
  to the embedding, so the delta is read from the untouched base, not added in place.
- `minigpt reason-rft --sampler` is now repeatable and takes `OP=PATH` (like `opd --teacher`):
  each task is sampled from its operator's expert. A plain `--sampler PATH` works as before.
  Write it as `--sampler=-=PATH` (argparse reads a bare `-=PATH` as an option).
- `docs/recipes/merge_val.py`: picks L (and scores checkpoints) on validation tasks that are not
  test tasks.

## Pre-registered rules (written before any consolidation result)

- Score: `reason-eval --n 200 --seed 0 --ops OP`, greedy, final-step model, per operator.
  "avg" = mean of the six cells (2 ops x 1, 2, 3 digits). Cells have +-0.03 sampling noise.
- Two seeded arms differ measurably when their seed ranges of avg do not overlap and the gap
  between the nearest range ends exceeds 0.05. A merge is one deterministic number: it differs
  from a seeded arm when it lies outside that arm's range by more than 0.05. The merge's own
  noise (experts, validation set) is not measured.
- Forgetting of an expert = base accuracy on the other operator minus the expert's, mean of
  1-3 digits. SFT vs SFT+GRPO forgetting has one training seed each, so it counts only if the gap
  exceeds 0.10 and has the same sign on 1, 2 and 3 digits.
- L is the best of {0.5, 0.75, 1.0, 1.25} by mean validation accuracy (ties keep the smaller).
  RFT lr is the best of {5e-5, 3e-4} the same way, from seed 0 only. OPD took dense-E1 settings
  untouched.

## Setup

- Base: `runs/student_sft` of the dense consolidation setup, weak on both ops (table below).
- Experts, seed 0 only (so expert/teacher noise is not covered): per operator, an SFT expert
  (`reason-sft --ops OP`, defaults: 20,000 traces, 3 epochs, lr 5e-4, from the base) and an
  SFT+GRPO expert (`grpo --base SFT --ops OP --steps 300`, other flags default).
- Consolidation from the base, seeds 0-3, using the SFT+GRPO experts as teachers/samplers:
  - OPD: `opd` with both teachers, 500 steps, 8 tasks per step, lr 3e-4, full estimator.
  - RFT: `reason-rft` with both samplers, 2000 tasks, k 8, keep 2, 2 epochs, lr 5e-5.
  - Merge: one number per expert set; the SFT-only experts are merged too.
- Validation tasks for L and the RFT lr: `Random(1)`, 1-3 digits, up to 200 distinct tasks per
  (op, digits), minus every task `reason-eval --n 200` scores. Sizes: `+` 15 / 200 / 200 and `-`
  2 / 200 / 200 (1-digit has only 100 pairs, 55 for `-`; the test set covers most of them).

## Results

Cells are accuracy on 1 / 2 / 3 digits.

### Experts, both operators

| model | `+` | `-` | avg | GPU s (shared) |
|---|---|---|---|---|
| base | 0.300 / 0.110 / 0.010 | 0.420 / 0.080 / 0.010 | 0.155 | |
| SFT `+` | 0.975 / 0.975 / 0.945 | 0.175 / 0.000 / 0.000 | 0.512 | 136 |
| SFT `-` | 0.020 / 0.000 / 0.000 | 0.990 / 0.985 / 0.960 | 0.492 | 115 |
| SFT+GRPO `+` | 1.000 / 1.000 / 0.995 | 0.205 / 0.000 / 0.000 | 0.533 | 136 + 1045 |
| SFT+GRPO `-` | 0.060 / 0.000 / 0.000 | 1.000 / 1.000 / 0.990 | 0.508 | 115 + 555 |

An expert that is perfect on its op and gone on the other scores about 0.50 here, and a model
that solved both would score 1.0. The two GRPO runs did the same 300 steps and took 1045 s and
555 s: that is the contention on the card, not the task.

### Forgetting: SFT vs SFT+GRPO

| expert | other op, base | other op, SFT | other op, SFT+GRPO | forgetting SFT | forgetting SFT+GRPO |
|---|---|---|---|---|---|
| `+` expert, scored on `-` | 0.420 / 0.080 / 0.010 | 0.175 / 0.000 / 0.000 | 0.205 / 0.000 / 0.000 | 0.112 | 0.102 |
| `-` expert, scored on `+` | 0.300 / 0.110 / 0.010 | 0.020 / 0.000 / 0.000 | 0.060 / 0.000 / 0.000 | 0.133 | 0.120 |

Verdict: no measurable difference. SFT+GRPO forgets 0.010 and 0.013 less (rule: 0.10), and the
sign is the same on 1 digit only (0.03 and 0.04 less); on 2 and 3 digits both are at 0.000.
The base is so weak on the other op that at most 0.17 and 0.14 could be lost, so this probe has
little room to separate the two. It does not test "on-policy RL forgets less" at seed variance.

### Consolidation

| arm | seed | `+` | `-` | avg | GPU min |
|---|---|---|---|---|---|
| base (reference) | | 0.300 / 0.110 / 0.010 | 0.420 / 0.080 / 0.010 | 0.155 | |
| merge, SFT+GRPO experts, L = 0.5 | | 0.075 / 0.000 / 0.000 | 0.060 / 0.015 / 0.000 | 0.025 | 0 (4 CPU s) + 0.5 for the L sweep |
| merge, SFT experts, L = 0.75 | | 0.055 / 0.000 / 0.000 | 0.065 / 0.000 / 0.000 | 0.020 | 0 (5 CPU s) + 0.4 for the L sweep |
| OPD | 0 | 0.430 / 0.110 / 0.015 | 0.515 / 0.150 / 0.015 | 0.206 | 12.5 |
| OPD | 1 | 0.435 / 0.105 / 0.005 | 0.460 / 0.140 / 0.005 | 0.192 | 11.8 |
| OPD | 2 | 0.460 / 0.040 / 0.000 | 0.565 / 0.135 / 0.005 | 0.201 | 12.6 |
| OPD | 3 | 0.290 / 0.120 / 0.025 | 0.485 / 0.075 / 0.000 | 0.166 | 12.7 |
| RFT | 0 | 0.860 / 0.370 / 0.055 | 0.795 / 0.225 / 0.045 | 0.392 | 0.9 |
| RFT | 1 | 0.930 / 0.275 / 0.075 | 0.690 / 0.240 / 0.000 | 0.368 | 0.8 |
| RFT | 2 | 0.945 / 0.290 / 0.050 | 0.810 / 0.205 / 0.015 | 0.386 | 0.8 |
| RFT | 3 | 0.915 / 0.320 / 0.060 | 0.770 / 0.195 / 0.015 | 0.379 | 0.9 |

Ranges of avg: OPD 0.166-0.206 (mean 0.191), RFT 0.368-0.392 (mean 0.381). GPU minutes are the
consolidation step only; the shared cost of the four experts is about 31 GPU min. RFT also spent
about 1.7 min on its lr probe and 0.2 min scoring it; OPD was not tuned.

Merge, every L (test scores, for information; only the L marked "chosen" was selected, on
validation, where every score was at most 0.011 and the SFT sweep's best was about 2 correct tasks):

| experts | L | `+` | `-` | avg | validation mean |
|---|---|---|---|---|---|
| SFT+GRPO | 0.5 (chosen) | 0.075 / 0.000 / 0.000 | 0.060 / 0.015 / 0.000 | 0.025 | 0.011 |
| SFT+GRPO | 0.75 | 0.055 / 0.005 / 0.000 | 0.065 / 0.005 / 0.000 | 0.022 | 0.000 |
| SFT+GRPO | 1.0 | 0.000 / 0.000 / 0.000 | 0.025 / 0.005 / 0.000 | 0.005 | 0.000 |
| SFT+GRPO | 1.25 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |
| SFT | 0.5 | 0.085 / 0.005 / 0.000 | 0.095 / 0.010 / 0.000 | 0.033 | 0.001 |
| SFT | 0.75 (chosen) | 0.055 / 0.000 / 0.000 | 0.065 / 0.000 / 0.000 | 0.020 | 0.002 |
| SFT | 1.0 | 0.005 / 0.000 / 0.000 | 0.000 / 0.005 / 0.000 | 0.002 | 0.001 |
| SFT | 1.25 | 0.000 / 0.000 / 0.000 | 0.000 / 0.000 / 0.000 | 0.000 | 0.000 |

Added after seeing these (post hoc, not selected): smaller L, and one expert alone.

| model | `+` | `-` | avg |
|---|---|---|---|
| merge SFT+GRPO, L = 0.1 | 0.480 / 0.110 / 0.015 | 0.425 / 0.060 / 0.000 | 0.182 |
| merge SFT+GRPO, L = 0.25 | 0.015 / 0.000 / 0.000 | 0.025 / 0.010 / 0.000 | 0.008 |
| merge SFT, L = 0.1 | 0.490 / 0.125 / 0.015 | 0.405 / 0.040 / 0.010 | 0.181 |
| merge SFT, L = 0.25 | 0.015 / 0.000 / 0.000 | 0.035 / 0.000 / 0.000 | 0.008 |
| `+` expert alone, L = 1 (CLI sanity) | 1.000 / 1.000 / 0.995 | 0.205 / 0.000 / 0.000 | = SFT+GRPO `+` |
| `+` expert alone, L = 0.5 | 0.790 / 0.360 / 0.075 | 0.175 / 0.000 / 0.000 | |
| `+` expert alone, L = 0.25 | 0.675 / 0.160 / 0.035 | 0.205 / 0.000 / 0.000 | |

Each expert's delta has norm 27.2 (`+`) or 27.8 (`-`) against a base norm of 80.1, 0.34 and 0.35
of it. The SFT+GRPO delta has the same norm as the SFT delta to three digits, so GRPO added
little to the distance from the base (the norm of GRPO minus SFT was not measured).

## Verdicts

1. Merge vs OPD: measurable, and the merge is worse. 0.020-0.025 lies 0.14 below the OPD range
   (0.166-0.206), and below the base (0.155). The 2026 claim that merging matches MOPD at a
   fraction of the cost does not hold in this setup: the merge costs nothing and gives nothing.
2. RFT vs OPD: measurable, RFT is better. Ranges 0.368-0.392 against 0.166-0.206, gap 0.16, at
   about 1/14 of the GPU time (0.8-0.9 against 11.8-12.7 min). Not a clean comparison: RFT got a
   validation-tuned lr (from two values; the other scored 0.361 on validation against 0.436) and
   OPD kept the untuned E1 settings, and RFT trains on about 2,000 verified samples for 2 epochs
   against OPD's 4,000 student rollouts.
3. OPD vs base: no measurable lift in avg (range 0.166-0.206, base 0.155; nearest gap 0.011). OPD
   raised 1-digit `+` and `-` on most seeds and left 2-3 digits at the base level.
4. Merge with SFT+GRPO experts vs with SFT experts: no measurable difference (0.025 vs 0.020, one
   number each). Both fail the same way at every L >= 0.25.
5. SFT vs SFT+GRPO forgetting: no measurable difference (see above).

Why the merge fails [INFERENCE]: one expert alone at half scale still works (0.79 on 1-digit
`+`), so the straight line from the base towards a single expert is usable, but adding two
deltas that are each a third of the base's weight norm breaks the model at L >= 0.25, and L = 0.1
is just the base again. The base is a weakly trained model, not a pretrained one, and 3,750
SFT steps at lr 5e-4 moved it far; task arithmetic is reported to work for small fine-tunes of
a converged base. A merge of lighter experts (fewer SFT steps, lower lr) was not run.

## Limits

- Experts and teachers are seed 0 only: seed ranges cover consolidation noise, not teacher
  noise. The merge has no seed range at all.
- The L selection was uninformative: validation accuracy is near zero for every L, so the chosen
  L is close to a tie-break (it is 0.5 for one set and 0.75 for the other; test scores favour 0.5
  for both, by 0.013 in the SFT case).
- 1-digit validation groups are tiny (15 and 2 tasks).
- One base, one task family, 842k parameters; the SFT experts are near the ceiling on their own
  op, so GRPO has little to add on top of them (and little forgetting difference to show).
- Wall times come from a shared GPU (the same GRPO job varied 2x). GPU-minute ratios between OPD
  and RFT (about 14x) are large enough to survive that; the merge's GPU cost is the L sweep only.

## Follow-up: light experts (predeclared before running)

The merge above failed with deltas 0.34 of the base norm. Follow-up on the same base, GPU 1:
two kinds of light SFT-only expert per operator (seed 0), each a much smaller move from the base:

- `short`: `reason-sft --ops OP --n 2000` (1/10 of the steps, lr 5e-4 as before).
- `lowlr`: `reason-sft --ops OP --lr 5e-5` (full 20,000 traces, 3 epochs, lr/10).

Reported per expert: both operators' scores, forgetting, and delta norm / base norm. Each kind
is merged (both ops, `L` swept over {0.5, 0.75, 1.0, 1.25} on the same validation tasks, the best
kept, ties to the smaller) and scored on the test tasks as before. Rule: the merge "works" if
(a) its avg beats the base's 0.155 by more than 0.05, and (b) on each operator its mean over 1-3
digits keeps at least half of the single expert's gain over the base on that operator. One seed, so
no seed range; every number is one run.

(results are added below after the run)

## Follow-up: OPD with RFT's tuning budget (predeclared before running)

RFT got two lr candidates scored on validation; OPD ran untuned. Both now get two points. OPD
candidates (seed 0, 500 steps, same teachers and student): AdamW `--lr 1e-4` and
`--optimizer muon --lr 3e-4` (both beat the default in the MOPD tuning of
[deepseek-v4.md](deepseek-v4.md)). The better mean validation accuracy wins (tie: AdamW 1e-4); it
then runs on seeds 1-3, seed 0 being the probe, and is scored on test. The comparison to RFT uses
the same rule as above (seed ranges apart by more than 0.05). The SFT+GRPO experts were deleted
after the first run, so this follow-up retrains them (`sft2_*`, `grpo2_*`, same commands and seed;
their scores are reported next to the first set's, since GPU runs need not repeat bit for bit).
Until this runs, the RFT claim above is only "beats untuned OPD".

## Reproduce

Logs: `runs/logs/merge/*.log` (training `.log`, wall seconds `.time`, `ev_*` = `reason-eval`).

```bash
B=runs/student_sft/ckpt.pt
for op in + -; do                                   # experts, per operator
  minigpt reason-sft --base $B --ops $op --seed 0 --out runs/sft_$op/ckpt.pt
  minigpt grpo --base runs/sft_$op/ckpt.pt --ops $op --steps 300 --seed 0 --out runs/grpo_$op/ckpt.pt
done
G="runs/grpo_+/ckpt.pt"; H="runs/grpo_-/ckpt.pt"
PYTHONPATH=src python docs/recipes/merge_val.py --base $B --expert $G --expert $H --out runs/merge_val/ckpt.pt
minigpt merge --base $B --expert $G --expert $H --scale 0.5 --out runs/merge/ckpt.pt
minigpt opd --teacher=+=$G --teacher=-=$H --student $B --ops +,- --eval-every 100 \
  --eval-tasks 200 --steps 500 --seed 0 --out runs/opd_s0/ckpt.pt
minigpt reason-rft --base $B --sampler=+=$G --sampler=-=$H --ops +,- --lr 5e-5 --seed 0 \
  --out runs/rft_s0/ckpt.pt
minigpt reason-eval --checkpoint runs/merge/ckpt.pt --ops + --n 200   # and --ops -
```
