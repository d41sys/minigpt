# OPD then RL, and the raw-teacher control, at toy scale

Two EMNLP 2026 claims about post-training a small student from RL-trained specialists
("Sequential beats joint"):

1. Running on-policy distillation (OPD) first and RL second beats either alone.
2. The OPD teachers must be RL-trained: distilling from the raw SFT teachers is worse.

This page tests both on the dense setup of [consolidation.md](consolidation.md): a weak mixed
student and two RL specialists (`+`, `-`, 1 to 3 digits). Four seeds per arm, final-step models,
n = 200 tasks per digit and operator. Toy results: the papers give the recipe shape only.

**Outcome: neither claim is supported at this scale.** No pair of arms passed the
pre-registered rule, on accuracy or on pass@8.

## Setup

- Student `runs/student_sft` (842,752 params; greedy `+` 0.30 / 0.11 / 0.01, `-` 0.42 / 0.08 /
  0.01 at n = 200).
- RL teachers: `+` = `runs/grpo`, `-` = `runs/sub_grpo`. Raw teachers (the SFT models those
  were trained from): `+` = `runs/reason_sft`, `-` = `runs/sub_sft`. Same architecture; the raw
  ones are strong but weaker on 2 and 3 digits (references below).
- One shared GPU (RTX 4090 with other jobs running), so wall times are rough.

## What was added

`minigpt opd --patience N`: stop once N consecutive evals fail to beat the best held-out greedy
accuracy so far (the step-0 eval counts); the checkpoint is the stopped model and
`steps trained: N` is printed. `minigpt reason-eval --pass-k K`: also print pass@K for 1 to 3
digits (K samples per task at T = 1, same tasks as the greedy eval). Tests:
`tests/test_minigpt_opd_patience.py`.

## Arms and rules (fixed before any run)

Budget: **600 optimizer steps per arm**. An `opd` step is 8 tasks x 1 sample, a `grpo` step
8 tasks x 8 samples, so the steps are not equal compute; wall time is in the table.

| arm | recipe |
|---|---|
| A | `grpo --ops +,-` 600 steps from the student |
| B | `opd` on-policy (reverse KL, both RL teachers) 600 steps |
| C | `opd` on-policy, RL teachers, cap 300 steps with the switch rule, then `grpo` for the rest |
| D | `opd --off-policy` 100 steps (sequence-level forward KL), then as C, `grpo` for the rest |
| E | as C, but the raw teachers |

Switch rule (C, D, E): OPD evals every 50 steps on 200 held-out mixed-op tasks (greedy
accuracy; tasks drawn from `seed + 10000`, not the scoring tasks). Switch when 2 consecutive
evals fail to beat the best so far (step 0 counts), or at the 300-step cap. `opd` lr 3e-4,
`grpo` default flags, seed `s` for every phase of an arm.

Scoring: `reason-eval --n 200 --ops +` and `--ops -` (seed 0, greedy), plus `--pass-k 8`.
Primary metric **M** = mean greedy accuracy over the 6 cells (2 operators x 1/2/3 digits).
X beats Y only if the 4 seed values of M do not overlap (min of X above max of Y) **and** that
gap exceeds 0.05; otherwise "no measurable difference". The same rule is applied to mean pass@8.
Claim 1 holds if C beats A and C beats B; claim 2 holds if C beats E.

**Post-hoc arms (added after seeing A-E seeds 0-2, before any run of them).** The switch rule
fired early: OPD stopped at step 100 to 200 (table), because greedy accuracy on the held-out
tasks dips from its step-0 value before it recovers, while the KL falls steadily (B, seed 0:
accuracy 0.23, 0.14, 0.15, ... 0.225 at steps 0, 50, 100, ... 600; KL 0.66 to 0.39). So C and E
are mostly GRPO. To give the claims an OPD phase that does something, two more arms ran with the
same scoring and verdict rule: **F** = `opd` RL teachers 300 steps (no early stop) then `grpo`
300 steps; **G** = as F with the raw teachers. Claim 1' = F beats A and B; claim 2' = F beats G.
Seed split: seeds 0-2 of A-E and seeds 0-1 of F/G ran on GPU 2, seed 3 of A-E and seeds 2-3
of F/G on GPU 0 (code and commands identical).

## Results

References (n = 200; greedy accuracy, then pass@8 after the bar) on 1 / 2 / 3 digits:

| model | `+` | `-` |
|---|---|---|
| student_sft | 0.30 / 0.11 / 0.01 &#124; 0.91 / 0.24 / 0.03 | 0.42 / 0.08 / 0.01 &#124; 0.93 / 0.24 / 0.00 |
| RL teacher `+` (grpo) | 1.00 / 1.00 / 0.99 &#124; 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 &#124; 0.09 / 0.01 / 0.00 |
| RL teacher `-` (sub_grpo) | 0.05 / 0.00 / 0.00 &#124; 0.09 / 0.01 / 0.00 | 1.00 / 0.98 / 0.96 &#124; 1.00 / 0.99 / 0.98 |
| raw teacher `+` (reason_sft) | 0.99 / 0.93 / 0.90 &#124; 0.99 / 0.99 / 0.97 | 0.01 / 0.00 / 0.00 &#124; 0.06 / 0.01 / 0.01 |
| raw teacher `-` (sub_sft) | 0.05 / 0.00 / 0.00 &#124; 0.10 / 0.00 / 0.00 | 0.99 / 0.79 / 0.68 &#124; 1.00 / 0.97 / 0.93 |

Per seed. "greedy acc" lists `+` 1/2/3 then `-` 1/2/3; M is their mean; "pass@8" is the mean over
the same six cells; "steps" is OPD (incl. off-policy) / GRPO. Wall seconds sum the phases, on a
shared GPU (the seed-3 runs of A-E and F/G seeds 2-3 ran on GPU 0, which was less loaded).

| arm | seed | steps | greedy acc | M | pass@8 | wall s |
|---|---|---|---|---|---|---|
| A | 0 | 0 / 600 | 0.66 / 0.18 / 0.01 / 0.57 / 0.14 / 0.01 | 0.259 | 0.432 | 1321 |
| A | 1 | 0 / 600 | 0.62 / 0.17 / 0.03 / 0.48 / 0.10 / 0.00 | 0.233 | 0.435 | 936 |
| A | 2 | 0 / 600 | 0.71 / 0.14 / 0.01 / 0.62 / 0.15 / 0.00 | 0.273 | 0.448 | 922 |
| A | 3 | 0 / 600 | 0.54 / 0.12 / 0.00 / 0.45 / 0.04 / 0.01 | 0.192 | 0.410 | 891 |
| B | 0 | 600 / 0 | 0.53 / 0.15 / 0.02 / 0.65 / 0.11 / 0.00 | 0.244 | 0.420 | 334 |
| B | 1 | 600 / 0 | 0.61 / 0.09 / 0.01 / 0.67 / 0.12 / 0.01 | 0.252 | 0.440 | 371 |
| B | 2 | 600 / 0 | 0.44 / 0.13 / 0.01 / 0.48 / 0.10 / 0.03 | 0.198 | 0.423 | 383 |
| B | 3 | 600 / 0 | 0.61 / 0.20 / 0.03 / 0.62 / 0.10 / 0.04 | 0.265 | 0.440 | 313 |
| C | 0 | 100 / 500 | 0.71 / 0.08 / 0.01 / 0.54 / 0.10 / 0.01 | 0.240 | 0.403 | 1269 |
| C | 1 | 100 / 500 | 0.64 / 0.10 / 0.01 / 0.61 / 0.12 / 0.01 | 0.248 | 0.418 | 871 |
| C | 2 | 200 / 400 | 0.60 / 0.12 / 0.01 / 0.66 / 0.17 / 0.01 | 0.263 | 0.440 | 749 |
| C | 3 | 150 / 450 | 0.61 / 0.08 / 0.01 / 0.65 / 0.07 / 0.01 | 0.238 | 0.391 | 718 |
| D | 0 | 200 / 400 (100 off) | 0.73 / 0.17 / 0.01 / 0.64 / 0.12 / 0.00 | 0.281 | 0.439 | 1165 |
| D | 1 | 200 / 400 (100 off) | 0.59 / 0.17 / 0.01 / 0.54 / 0.12 / 0.01 | 0.242 | 0.443 | 696 |
| D | 2 | 200 / 400 (100 off) | 0.74 / 0.23 / 0.02 / 0.48 / 0.10 / 0.00 | 0.263 | 0.451 | 790 |
| D | 3 | 300 / 300 (100 off) | 0.62 / 0.14 / 0.01 / 0.64 / 0.11 / 0.01 | 0.254 | 0.464 | 615 |
| E | 0 | 100 / 500 | 0.84 / 0.14 / 0.01 / 0.57 / 0.13 / 0.01 | 0.286 | 0.426 | 1096 |
| E | 1 | 100 / 500 | 0.74 / 0.11 / 0.02 / 0.63 / 0.07 / 0.01 | 0.265 | 0.423 | 798 |
| E | 2 | 200 / 400 | 0.67 / 0.19 / 0.02 / 0.66 / 0.07 / 0.01 | 0.271 | 0.454 | 715 |
| E | 3 | 150 / 450 | 0.57 / 0.10 / 0.02 / 0.53 / 0.10 / 0.01 | 0.221 | 0.403 | 733 |
| F | 0 | 300 / 300 | 0.76 / 0.13 / 0.04 / 0.72 / 0.07 / 0.01 | 0.286 | 0.461 | 603 |
| F | 1 | 300 / 300 | 0.67 / 0.15 / 0.01 / 0.65 / 0.13 / 0.02 | 0.273 | 0.462 | 613 |
| F | 2 | 300 / 300 | 0.78 / 0.20 / 0.01 / 0.73 / 0.14 / 0.01 | 0.312 | 0.480 | 584 |
| F | 3 | 300 / 300 | 0.55 / 0.16 / 0.01 / 0.69 / 0.09 / 0.01 | 0.252 | 0.438 | 584 |
| G | 0 | 300 / 300 | 0.80 / 0.16 / 0.01 / 0.61 / 0.09 / 0.01 | 0.278 | 0.448 | 580 |
| G | 1 | 300 / 300 | 0.77 / 0.15 / 0.03 / 0.67 / 0.14 / 0.02 | 0.297 | 0.463 | 597 |
| G | 2 | 300 / 300 | 0.79 / 0.17 / 0.01 / 0.80 / 0.15 / 0.01 | 0.323 | 0.475 | 581 |
| G | 3 | 300 / 300 | 0.67 / 0.14 / 0.01 / 0.57 / 0.06 / 0.00 | 0.240 | 0.433 | 599 |

Arms, mean (min..max over 4 seeds):

| arm | M | mean pass@8 |
|---|---|---|
| A GRPO only | 0.239 (0.192..0.273) | 0.431 (0.410..0.448) |
| B OPD only | 0.240 (0.198..0.265) | 0.431 (0.420..0.440) |
| C OPD then GRPO (switch rule) | 0.247 (0.238..0.263) | 0.413 (0.391..0.440) |
| D off-policy, OPD, GRPO | 0.260 (0.242..0.281) | 0.449 (0.439..0.464) |
| E raw-teacher OPD, then GRPO | 0.261 (0.221..0.286) | 0.426 (0.403..0.454) |
| F post-hoc 300 OPD + 300 GRPO | 0.280 (0.252..0.312) | 0.460 (0.438..0.480) |
| G post-hoc, raw teachers | 0.284 (0.240..0.323) | 0.455 (0.433..0.475) |

## Verdicts

Gap = lowest seed of the first arm minus highest seed of the second; a gap above 0.05 is needed.

| comparison | M gap | pass@8 gap | verdict |
|---|---|---|---|
| claim 1: C vs A | -0.035 | -0.057 | no measurable difference |
| claim 1: C vs B | -0.027 | -0.049 | no measurable difference |
| claim 2: C vs E (raw teachers) | -0.048 | -0.063 | no measurable difference |
| claim 1' (post-hoc): F vs A | -0.021 | -0.010 | no measurable difference |
| claim 1' (post-hoc): F vs B | -0.013 | -0.002 | no measurable difference |
| claim 2' (post-hoc): F vs G | -0.071 | -0.037 | no measurable difference |
| D vs C, D vs A | -0.043, -0.031 | -0.073, -0.009 | no measurable difference |

- **Claim 1 (OPD then RL beats either alone): not supported here.** All arms sit at M 0.24 to
  0.28 with seed ranges 0.03 to 0.08 wide. F has a higher mean than A and B (0.280 against 0.239
  and 0.240; pass@8 0.460 against 0.431) but the ranges overlap, so no gain is claimed. Four
  seeds cannot see a 0.04 difference when seeds spread by 0.08.
- **Claim 2 (teachers must be RL-trained): not supported here.** The raw-teacher arms E and G
  score as high as their RL-teacher twins C and F (G 0.284 against F 0.280). The raw teachers
  are strong (references), only weaker on 2 and 3 digits, which the 842k student never learns
  in 600 steps (2- and 3-digit cells are 0.0 to 0.23 in every arm). [INFERENCE] student
  capacity and budget, not the teacher's RL training, are the limit at this scale, so a teacher
  difference could not show up.
- **Coverage (pass@8):** every run is at 0.39 to 0.48 against 0.39 for the SFT student
  (0.91 / 0.24 / 0.03 and 0.93 / 0.24 / 0.00 over the six cells), so training raised coverage
  slightly, mostly on 2-digit tasks. GRPO alone (A, 0.431) does not sit below OPD-then-RL, so
  there is no sign of RL narrowing coverage here, and OPD-then-RL does not widen it. No
  coverage claim is supported or refuted.
- The pre-registered switch rule stopped OPD at 100 to 200 steps: held-out greedy
  accuracy is noisy and starts at its highest value (the student's own 0.23). That is why C
  is mostly GRPO and why F and G were added. They do not change the outcome.

## Limits

- Toy scale: an 842k-parameter student that learns almost only 1-digit answers in 600 steps,
  so most of M is 1-digit accuracy. The claims concern larger models; this run says nothing
  against them.
- Seed spread of M is 0.03 to 0.08 per arm, so the rule only sees gaps above about 0.1.
- One scoring set per cell (n = 200, seed 0), pass@8 from one sampling pass.
- Budgets are in optimizer steps, not compute: a `grpo` step costs 8 tasks x 8 samples and an
  `opd` step 8 tasks x 1 sample. D includes 100 off-policy steps ([consolidation.md](consolidation.md)
  E2 used 1000 and reached far higher accuracy).
- OPD uses the full-vocabulary reverse KL at lr 3e-4, the repo defaults; no tuning.
- Not tested: other switch rules or budgets, a mixed teacher set during RL, larger students.

## Reproduce

Logs: `runs/logs/og_*.log` (training), `og_eval_*_op{+,-}.log` (`reason-eval`), `.time` wall
seconds. `K` is the `steps trained:` line of the OPD phase. Raw teachers: replace the two
teacher paths with `runs/reason_sft` and `runs/sub_sft`.

```bash
T="--teacher=+=runs/grpo/ckpt.pt --teacher=-=runs/sub_grpo/ckpt.pt"
O="--ops +,- --eval-every 50 --eval-tasks 200"; S=runs/student_sft/ckpt.pt
minigpt grpo --base $S --ops +,- --steps 600 --seed 0 --out runs/A/ckpt.pt                     # A
minigpt opd $T --student $S $O --steps 600 --seed 0 --out runs/B/ckpt.pt                       # B
minigpt opd $T --student $S $O --steps 300 --patience 2 --seed 0 --out runs/C_opd/ckpt.pt      # C
minigpt grpo --base runs/C_opd/ckpt.pt --ops +,- --steps $((600 - K)) --seed 0 --out runs/C/ckpt.pt
minigpt opd $T --student $S $O --off-policy --steps 100 --seed 0 --out runs/D_off/ckpt.pt      # D
minigpt opd $T --student runs/D_off/ckpt.pt $O --steps 300 --patience 2 --seed 0 --out runs/D_opd/ckpt.pt
minigpt opd $T --student $S $O --steps 300 --seed 0 --out runs/F_opd/ckpt.pt                   # F
minigpt grpo --base runs/F_opd/ckpt.pt --ops +,- --steps 300 --seed 0 --out runs/F/ckpt.pt
minigpt reason-eval --checkpoint runs/C/ckpt.pt --ops + --n 200 --pass-k 8
```
