# Consolidating specialists into one model, at toy scale

Four labs publish a recipe for merging several RL-trained specialists into one model, and one
for shrinking a strong model into a small one. This page runs a toy version of each on the
repo's two-op task (`+` and `-`, 1 to 3 digits) and reports what moved. Everything is one run
per variant at seed 0, plus seed 1 where a claim needs it. n = 50 tasks per digit, so a
single accuracy is about +-0.07 and, as E1 shows, seeds differ by far more than that.
These are toy results; the sources give the recipe shape only.

Setup (all runs, Korea server, one shared GPU, CUDA):

- Student `runs/student_sft` (842,752 params, greedy `+` 0.20 / 0.08 / 0.00, `-` 0.42 / 0.06 / 0.00
  on 1 / 2 / 3 digits).
- Specialist teachers: `+` = `runs/grpo` (1.00 / 1.00 / 1.00 on `+`, 0.00 on `-`), `-` =
  `runs/sub_grpo` (1.00 / 0.98 / 0.98 on `-`, 0.04 / 0.00 / 0.00 on `+`).
- `opd`: 500 steps, 8 tasks per step, lr 3e-4, 200 eval tasks every 100 steps. `grpo`: default
  flags, 200 steps (300 for the single-op stages). Evaluation of every checkpoint:
  `minigpt reason-eval --checkpoint CKPT --ops +` and `--ops -` (n = 50, greedy).
- Teachers are given with `--teacher=+=PATH --teacher=-=PATH`.

## What was added

`minigpt opd --estimator {full,token}` (default `full`, the old behaviour) and `--r-max`
(default 5). `token` is OPD as RL: the student's sampled token earns the reward
`clip(sg(log p_teacher - log p_student), -r_max, r_max)` and the loss is the policy gradient
`-mean(reward * log p_student)`. One update per batch, so the PPO ratio is 1 and the PPO clip is
omitted. `full` is the full-vocabulary reverse KL. In expectation the token reward is minus the
per-token reverse KL, from one sampled token instead of the whole vocabulary (DeepSeek-V4
section 5.1.2 calls the token estimate high-variance). Code: `token_pg_loss` in
`src/minigpt/opd.py`; tests in `tests/test_minigpt_opd_estimator.py`.

## Sources and what was copied

| Experiment | Source | Toy version |
|---|---|---|
| E1 estimator | Kimi K3, [arXiv 2607.24653](https://arxiv.org/abs/2607.24653) section 4.1.3 eq. 15 (token advantage is the log-ratio of teacher to student) | `opd --estimator full` vs `token` |
| E2 | Microsoft AI, [MAI-Thinking-1](https://microsoft.ai/news/introducing-mai-thinking-1/): specialist RL, trace distillation, final RL | teachers -> `opd --off-policy` (1000 steps) -> `grpo --ops +,-` 200 steps |
| E3 | NVIDIA, [Nemotron 3 Ultra report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Ultra-Technical-Report.pdf) section 3.3, two-iteration MOPD (iteration 2 adds teachers initialised from the iteration-1 model) | round 1 MOPD, refresh each teacher by GRPO from the round-1 student, round 2 MOPD |
| E4 | GLM-5, [arXiv 2602.15763](https://arxiv.org/abs/2602.15763) section 3.5, on-policy cross-stage distillation: earlier-stage checkpoints are the teachers, advantage `sg(log p_teacher - log p_student)`, group size 1 | stage 1 GRPO `+`, stage 2 GRPO `-`, then `opd` toward both stage checkpoints |
| E5 | Qwen3, [arXiv 2505.09388](https://arxiv.org/abs/2505.09388) section 4.5, strong-to-weak: off-policy distillation, then on-policy | 125k-param random-init student, off-policy then on-policy from that checkpoint |

Simplifications: E3 replaces the teachers with the refreshed ones (the paper keeps the first
round's teachers too); E4 uses two stages and two teachers, not the paper's SFT + reasoning RL +
general RL pipeline.

## Results

Cells are greedy accuracy on 1 / 2 / 3 digits. "KL" is the final reverse KL per token on the
eval tasks against the run's own teachers (so only comparable within an experiment). Wall time
is on a GPU shared with other jobs, so it is only a rough guide.

### References

| model | `+` | `-` |
|---|---|---|
| student_sft | 0.20 / 0.08 / 0.00 | 0.42 / 0.06 / 0.00 |
| teacher `+` (grpo) | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 0.00 |
| teacher `-` (sub_grpo) | 0.04 / 0.00 / 0.00 | 1.00 / 0.98 / 0.98 |

### E1: full vs token estimator (500 steps, multi-teacher OPD from student_sft)

| run | seed | lr | `+` | `-` | KL (start 0.66 / 0.68) | wall s |
|---|---|---|---|---|---|---|
| full | 0 | 3e-4 | 0.68 / 0.20 / 0.02 | 0.86 / 0.22 / 0.02 | 0.348 | 785 |
| full | 1 | 3e-4 | 0.68 / 0.12 / 0.00 | 0.32 / 0.10 / 0.00 | 0.423 | 726 |
| full | 0 | 1e-4 | 0.70 / 0.24 / 0.02 | 0.54 / 0.20 / 0.00 | 0.337 | 746 |
| token | 0 | 3e-4 | 0.00 / 0.00 / 0.00 | 0.22 / 0.02 / 0.00 | 1.204 | 680 |
| token | 1 | 3e-4 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 1.525 | 872 |
| token | 0 | 1e-4 | 0.60 / 0.02 / 0.02 | 0.42 / 0.04 / 0.00 | 0.648 | 681 |
| token | 1 | 1e-4 | 0.42 / 0.08 / 0.00 | 0.32 / 0.06 / 0.00 | 0.602 | 762 |

KL every 100 steps, seed 0: full 0.663, 0.496, 0.515, 0.445, 0.416, 0.348; token (lr 3e-4)
0.663, 1.208, 1.227, 1.390, 1.220, 1.204; token (lr 1e-4) 0.663, 0.555, 0.688, 0.725, 0.669, 0.648.

Verdict: the token estimator is worse. At the shared default lr it diverges on both seeds (KL
rises from 0.66 to 1.2 to 1.5, accuracy 0). At lr 1e-4 it stays near the start (0.60 to 0.65)
while the full estimator reaches 0.34 to 0.42 at either lr, a gap well beyond the 0.08 seed
spread of the full runs. This matches the DeepSeek-V4 claim that the one-token estimate is
noisier; here the noise also makes the lr much less forgiving. Accuracy: the full runs lift
1-digit `+` from 0.20 to 0.68 / 0.70 on both seeds and the token runs do not reach that
(0.60, 0.42), but the `-` column swings by 0.54 between full seeds 0 and 1, so no
accuracy difference between full and token at lr 1e-4 is claimed beyond that. The runs are
also not a pure replica of the paper, which uses a PPO-style clipped ratio over many updates per
batch; this repo does one update per batch.

### E2: MAI-Thinking-1 pipeline vs RL alone

| run | seed | `+` | `-` | wall s |
|---|---|---|---|---|
| trace SFT (`opd --off-policy`, 1000 steps from student_sft) | 0 | 0.82 / 0.42 / 0.06 | 0.46 / 0.16 / 0.00 | 1356 |
| trace SFT, then `grpo --ops +,-` 200 steps | 0 | 0.92 / 0.54 / 0.12 | 0.90 / 0.56 / 0.04 | 262 |
| `grpo --ops +,-` 200 steps from student_sft (control) | 0 | 0.50 / 0.16 / 0.00 | 0.32 / 0.02 / 0.00 | 275 |
| trace SFT | 1 | 0.50 / 0.24 / 0.06 | 0.46 / 0.30 / 0.02 | 1373 |
| trace SFT, then RL | 1 | 0.96 / 0.46 / 0.08 | 1.00 / 0.56 / 0.06 | 280 |
| RL alone | 1 | 0.60 / 0.14 / 0.00 | 0.32 / 0.06 / 0.00 | 290 |

Verdict: the full pipeline beats RL alone on both seeds, by 0.3 to 0.7 on 1-digit and about 0.3
to 0.5 on 2-digit for both ops, far beyond the noise. The two RL-alone seeds agree with each
other, and the trace-SFT step alone is not enough (seed 1 `+` is 0.50 / 0.24 / 0.06), so the
final RL climb does real work on top of the distilled student. Not controlled: the pipeline
spends 1000 more off-policy steps than the control, and the comparison to "more RL steps from
student_sft" was not run.

### E3: Nemotron-style two-round MOPD

Round 1 is the E1 full seed 0 student (500 steps). Refreshed teachers are GRPO, 300 steps each
from the round-1 student on one op. Round 2 is 500 more OPD steps from the round-1 student;
control = the same 500 steps against the original teachers.

| model | `+` | `-` | KL (to its own teachers) | wall s |
|---|---|---|---|---|
| round 1 student | 0.68 / 0.20 / 0.02 | 0.86 / 0.22 / 0.02 | 0.348 | 785 |
| refreshed teacher `+` | 0.86 / 0.38 / 0.02 | 0.60 / 0.20 / 0.00 | | 381 |
| refreshed teacher `-` | 0.32 / 0.12 / 0.00 | 0.82 / 0.20 / 0.00 | | 382 |
| round 2, refreshed teachers | 0.72 / 0.12 / 0.00 | 0.74 / 0.18 / 0.00 | 0.048 | 731 |
| control, original teachers | 0.82 / 0.16 / 0.00 | 0.84 / 0.18 / 0.02 | 0.292 | 709 |

Verdict: no measurable difference between round 2 and the control (all gaps are within the seed
noise measured in E1, single seed). The refreshed teachers are weaker specialists than the
originals (0.86 / 0.82 on 1-digit against 1.00), and the round 2 KL of 0.048 is low only
because those teachers start from the student, so it does not show a better student.

### E4: GLM-5 staged RL and cross-stage distillation

Stage 1: `grpo --ops +` 300 steps from student_sft. Stage 2: `grpo --ops -` 300 steps from
stage 1. Cross-stage: `opd` with teachers `+` = stage 1, `-` = stage 2, student = stage 2,
500 steps, run at seeds 0 and 1. Note the token variant uses lr 1e-4 and the full variant the
default 3e-4 (the lr that kept token stable in E1), so the estimator and the lr are confounded.

| model | seed | `+` | `-` | KL | wall s |
|---|---|---|---|---|---|
| stage 1 (`+` only) | 0 | 0.84 / 0.20 / 0.02 | 0.28 / 0.00 / 0.00 | | 422 |
| stage 2 (then `-`) | 0 | 0.46 / 0.16 / 0.00 | 0.62 / 0.06 / 0.00 | | 420 |
| cross-stage, full | 0 | 0.56 / 0.14 / 0.00 | 0.42 / 0.06 / 0.00 | 0.017 | 740 |
| cross-stage, full | 1 | 0.36 / 0.12 / 0.00 | 0.26 / 0.06 / 0.00 | 0.032 | 708 |
| cross-stage, token | 0 | 0.72 / 0.16 / 0.02 | 0.48 / 0.06 / 0.00 | 0.011 | 757 |
| cross-stage, token | 1 | 0.72 / 0.22 / 0.02 | 0.44 / 0.08 / 0.00 | 0.009 | 708 |

Verdict: forgetting is visible only on 1 digit: `+` falls from 0.84 to 0.46 after the `-` stage
(0.38, beyond the +-0.07 eval noise; one seed for the RL stages). Cross-stage distillation with the
full estimator does not recover it (0.56 and 0.36, mean 0.46 = stage 2). With the token
estimator, `+` is 0.72 on both seeds, but `-` drops from 0.62 to 0.48 / 0.44. Summing
1-digit `+` and `-`: stage 2 1.08, full 0.98 / 0.62, token 1.20 / 1.16. Tentative: token
cross-stage keeps more of both ops on 1 digit than full cross-stage, but it is confounded with lr,
and 2- and 3-digit accuracies show no measurable difference. The stage KL (0.01 to 0.03) is small
for both because the student starts at the `-` teacher.

### E5: Qwen3 strong-to-weak, small student

Student `--student-config 64,2,4` (124,800 params, random init), both teachers, 200 eval tasks.
Off-policy: 1000 steps. Then 500 more steps either on-policy (the recipe) or off-policy (equal
compute control), both from the off-policy checkpoint.

| run | `+` | `-` | KL | wall s |
|---|---|---|---|---|
| off-policy 1000 steps | 0.04 / 0.02 / 0.02 | 0.16 / 0.02 / 0.00 | 1.770 (start 14.1) | 1576 |
| then on-policy 500 | 0.06 / 0.00 / 0.00 | 0.18 / 0.04 / 0.00 | 0.954 | 421 |
| then off-policy 500 | 0.32 / 0.00 / 0.00 | 0.28 / 0.00 / 0.00 | 0.965 | 587 |

Verdict: no measurable difference between the on-policy phase and more off-policy training;
both bring KL from 1.77 to about 0.95 and accuracy stays near zero on 2 and 3 digits. The
repo's earlier on-policy-from-random run stayed at 0.00 and here the on-policy phase does
learn (KL 1.77 to 0.95) once an off-policy phase has set the student up, which is what the
paper's ordering predicts, but a 125k student does not turn that into accuracy. One seed.

## Reproduce

Logs are `runs/logs/cs_*.log` (training) and `cs_eval_*_op{+,-}.log` (`reason-eval`); `.time`
files hold wall seconds. For example:

```bash
T="--teacher=+=runs/grpo/ckpt.pt --teacher=-=runs/sub_grpo/ckpt.pt"
O="--ops +,- --eval-every 100 --eval-tasks 200"
minigpt opd $T --student runs/student_sft/ckpt.pt $O --steps 500 --estimator token --lr 1e-4 --out runs/cs_e1_token/ckpt.pt   # E1
minigpt opd $T --student runs/student_sft/ckpt.pt $O --steps 1000 --off-policy --out runs/cs_e2_sft/ckpt.pt                    # E2
minigpt grpo --base runs/cs_e2_sft/ckpt.pt --ops +,- --steps 200 --out runs/cs_e2_rl/ckpt.pt
minigpt grpo --base runs/student_sft/ckpt.pt --ops + --steps 300 --out runs/cs_e4_s1/ckpt.pt                                    # E4
minigpt opd $T --student-config 64,2,4 $O --steps 1000 --off-policy --out runs/cs_e5_off/ckpt.pt                              # E5
```
