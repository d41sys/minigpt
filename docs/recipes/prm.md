# Process reward model vs outcome reward model, best-of-N (OpenAI, toy scale)

Design and verdict rule were committed before the seeds ran (commits `c4bb255` and `7a48d9e`
of PR #49); results, limits and the extra follow.

## Source

| What | Source |
|---|---|
| Outcome-supervised RM (ORM): trained on final-answer correctness, scored at the last token | Lightman et al. 2023, "Let's Verify Step by Step", sec. 2.5 ([arXiv 2305.20050](https://arxiv.org/abs/2305.20050), 2023-05-31) |
| Process-supervised RM (PRM): one correct/incorrect prediction after the last token of each step; labels only up to the first wrong step; solution score = product of step probabilities (the minimum is the alternative in appendix F: 77.6% vs 78.2% solved, neutral labels counted as positive) | same paper, sec. 2.6 and appendix F |
| Evaluation by best-of-N over uniformly sampled solutions from one fixed generator, majority vote as the baseline | same paper, sec. 2.1 and 3 |
| Their large-scale result: best-of-1860 on MATH, PRM 78.2%, ORM 72.4%, majority vote 69.6% | same paper, Figure 3 |
| Why DeepSeek-R1 did not use a PRM: fine-grain steps are hard to define in general reasoning, and a model-based PRM invites reward hacking and needs retraining | DeepSeek-R1, appendix G.2 "Unsuccessful Attempts" ([arXiv 2501.12948](https://arxiv.org/abs/2501.12948), v2 2026-01-04) |

The paper's step labels came from human annotators (PRM800K). Here a program labels every
step, so this is the "supervision is free" corner of the paper's design space, not a
replication of its human-label cost.

## What was built

`src/minigpt/prm.py`, commands `minigpt prm` (train) and `minigpt best-of-n` (select).

- **Steps.** A trace `<think>7+5+0=12 4+8+1=13</think><answer>132</answer>` has one step per
  column plus the answer. A think step is right when it equals the reference step of the same
  index (`Task.solution()`); the answer step is right when every think step was right, none is
  missing and the answer is true. Labels stop at the first wrong step, as in the paper.
  Malformed completions have no step labels.
- **Models.** Both verifiers are the policy's body plus a scalar head
  (`GPTForClassification`, `num_classes=1`), same size, same init, trained with binary
  cross-entropy on the same samples. The PRM reads the head at the last token of every step;
  the ORM reads it at the last token of the sequence and learns `accuracy_reward`.
- **Samples.** `minigpt prm` draws 3000 tasks (1-3 digits) from `--data-seed`, samples 4
  completions each at temperature 1, caches them in `--samples`, and trains both models on
  that cache; the last 10% of samples are held out to report verifier accuracy. The seed
  changes only the head init and the batch order.
- **Selectors** (`best-of-n`): `random` (exact expectation: share correct among the first N),
  `majority` (most common parsed answer, ties to the earliest), `orm` (argmax ORM logit),
  `prm_min` (argmax of the minimum step logit, as assigned), `prm_prod` (argmax of the summed
  step log-probabilities, the paper's main reduction), `oracle` (pass@N, ceiling). Malformed
  completions get PRM score -inf. Ties go to the earliest sample. N=1, 4, 16 are prefixes of the same 16
  samples per task.

## Pre-registered setup and verdict rule

- Policy (fixed): `runs/reason_sft_n15000e2/ckpt.pt`, chosen from the existing SFT checkpoints
  because its 3-digit pass@1 is near the target band (0.2-0.6) and N therefore matters. Measured
  before anything else, n=200 per digit count, temperature 1: pass@1 0.760 / 0.495 / 0.195 on
  1 / 2 / 3 digits (greedy 3-digit 0.375). Candidates not used: `reason_sft` (0.950 / 0.835 /
  0.695, too strong), `reason_sft_n10000e3` (0.490 / 0.340 / 0.135), `reason_sft_n10000e2` and
  weaker ones (3-digit about 0).
- Held-out tasks: 200 per digit count, 1-3 digits (600), `--seed 7`; the verifiers' tasks come
  from another seed. 16 samples each, drawn once and cached, identical for every verifier seed.
  The 100 one-digit problems and some two-digit ones necessarily also occur in the verifiers'
  training tasks; 3-digit overlap is expected to be about 0.
- Verifier seeds 0-3; policy, tasks and samples fixed. Pilot runs with seed 100 (in no table)
  chose lr `1e-4` and 8 epochs from the verifiers' held-out BCE alone; the best-of-N tasks were
  not used for tuning. Pilot held-out BCE (PRM / ORM) at 3 epochs: lr 3e-4 0.443 / 0.571,
  1e-4 0.393 / 0.553, 3e-5 0.379 / 0.536; at 6 epochs: 1e-4 0.286 / 0.514, 3e-5 0.323 / 0.502.
  All were still falling at 6 epochs; 8 is a budget choice, not a tuned optimum.
- Primary cell: **3-digit accuracy of the selected answer, N=16** (n=200 tasks). Everything else
  (1, 2 digits, N=4, all-digit mean, `prm_prod`) is descriptive.
- Claim 1: PRM (`prm_min`) beats ORM at N=16. Claim 2: PRM (`prm_min`) beats majority vote at
  N=16.
- Verdict: **measurable** only if the 4 PRM seeds' range does not overlap the comparator's range
  (PRM min > comparator max; majority vote is one deterministic number) AND the gap of means
  is > 0.05. Otherwise **not measurable**, and a reversed pattern is stated as such. A cell at n=200 has a
  binomial standard error of about 0.03 at p=0.5.

## Commands

```bash
P=runs/reason_sft_n15000e2/ckpt.pt
for s in 0 1 2 3; do
  minigpt prm --sft $P --samples runs/prm/samples.json --seed $s --lr 1e-4 --epochs 8 --out runs/prm_s$s
  minigpt best-of-n --policy $P --prm runs/prm_s$s/prm.pt --orm runs/prm_s$s/orm.pt \
    --samples runs/prm/eval_samples.json --json-out runs/logs/bon_s$s.json
done
```

`tests/test_minigpt_prm.py` covers the step labels, token positions, selectors, both
verifiers' training and the sample cache.

## Results: best-of-N (verifier seeds 0-3, 200 tasks per cell)

Logs: `runs/logs/prm/` (`prm_s*.log` training, `bon_s*.log` and `.json` selection, `diag.log`).
The policy's pass@1 on these tasks, which is also every selector's accuracy at N=1, is
0.735 / 0.510 / 0.175 on 1 / 2 / 3 digits; pass@16 (`oracle`) is 1.000 / 0.975 / 0.800.
Held-out verifier quality after 8 epochs (last 10% of the 12000 samples, seeds 0-3): PRM step
accuracy 0.892 / 0.923 / 0.874 / 0.912 (73.8% of the 24847 labeled steps are right), ORM
outcome accuracy 0.747 / 0.776 / 0.756 / 0.743 (47.3% of the samples are right).

**Primary cell: 3 digits, N=16, accuracy of the selected answer**

| seed | random | majority | ORM | PRM (min) | PRM (product) | oracle |
|---|---|---|---|---|---|---|
| 0 | 0.199 | 0.355 | 0.100 | 0.340 | 0.320 | 0.800 |
| 1 | 0.199 | 0.355 | 0.140 | 0.365 | 0.380 | 0.800 |
| 2 | 0.199 | 0.355 | 0.110 | 0.245 | 0.255 | 0.800 |
| 3 | 0.199 | 0.355 | 0.100 | 0.365 | 0.370 | 0.800 |
| mean | | | 0.113 | 0.329 | 0.331 | |

`random`, `majority` and `oracle` do not depend on the verifier seed.

**Verdicts (rule above)**

- Claim 1, PRM beats ORM at N=16: **measurable**. PRM-min 0.245-0.365 (mean 0.329) against
  ORM 0.100-0.140 (mean 0.113): ranges do not overlap, gap +0.216. The same holds for the
  product reduction (0.255-0.380).
- Claim 2, PRM beats majority vote at N=16: **not measurable, and not supported**. PRM-min
  mean 0.329 against 0.355 (gap -0.026); the PRM range contains the majority value. The
  product reduction gives the same picture (0.331).

**All cells, per seed (s0 s1 s2 s3)**; N=1 is the policy's pass@1 for every selector.

| N | digits | random | majority | oracle | ORM | PRM (min) | PRM (product) |
|---|---|---|---|---|---|---|---|
| 4 | 1 | 0.801 | 0.875 | 0.985 | 0.870 0.875 0.830 0.845 | 0.955 0.935 0.905 0.915 | 0.950 0.935 0.910 0.920 |
| 4 | 2 | 0.496 | 0.635 | 0.850 | 0.500 0.495 0.530 0.555 | 0.700 0.745 0.700 0.740 | 0.710 0.730 0.705 0.745 |
| 4 | 3 | 0.190 | 0.210 | 0.495 | 0.180 0.220 0.215 0.200 | 0.330 0.310 0.250 0.305 | 0.340 0.325 0.260 0.365 |
| 16 | 1 | 0.782 | 0.945 | 1.000 | 0.680 0.735 0.585 0.645 | 0.860 0.870 0.815 0.830 | 0.850 0.870 0.795 0.820 |
| 16 | 2 | 0.485 | 0.710 | 0.975 | 0.280 0.270 0.380 0.345 | 0.630 0.665 0.585 0.640 | 0.595 0.605 0.565 0.620 |
| 16 | 3 | 0.199 | 0.355 | 0.800 | 0.100 0.140 0.110 0.100 | 0.340 0.365 0.245 0.365 | 0.320 0.380 0.255 0.370 |

Mean over the three digit counts at N=16 (descriptive): ORM 0.353-0.382, PRM-min 0.548-0.633,
PRM-product 0.538-0.618, majority 0.670, random 0.488, oracle 0.925.

Post-hoc readings of cells that were not pre-registered (the same range-and-gap rule applied
after seeing the table, so treat them as hypotheses): at N=4 the PRM beats majority vote at
1, 2 and 3 digits (e.g. 3 digits 0.250-0.330 against 0.210), while at N=16 majority vote beats
the PRM at 1 and 2 digits (0.945 against 0.815-0.870; 0.710 against 0.585-0.665). The PRM's
lead over majority vote therefore shrinks as N grows, unlike the paper's large-scale trend
(sec. 3: the PRM's lead over ORM and majority vote widens with N). The ORM is above `random`
only at 1 digit at N=4 (0.830-0.875 against 0.801), level with it at 2 and 3 digits, and
worse than `random` at N=16 in every digit count: taking its top score among 16 hurts.

**Why (post-hoc, one script, `docs/recipes/prm_diag.py`, 3-digit tasks with a right and a wrong
sample among the 16: 160 of 200).** Within-task AUC, i.e. how often a right sample outscores a
wrong one of the same task: ORM 0.548 / 0.625 / 0.585 / 0.565 (seeds 0-3), PRM-min 0.797 / 0.802 /
0.714 / 0.792, PRM-product 0.830 / 0.833 / 0.746 / 0.835. The picks of the PRM are 100%
well-formed; the ORM's are 94.0-97.0% (`diag.log`). [INFERENCE] The ORM ranks only slightly
better than chance inside a task, and the argmax of 16 scores selects the tail of that score,
where it is least reliable; this was not tested further. The PRM's step labels tell it where
the arithmetic went wrong, which the outcome label does not.

## Extra: GRPO with a PRM-shaped reward (not part of the two claims)

`minigpt grpo --prm CKPT --prm-weight W` adds `W * P(every step right)` (the PRM's step
probabilities multiplied, 0 for a malformed completion) to the usual reward (accuracy +
0.1 format). Pre-registered before it ran: same policy as above, default GRPO settings (200
steps, 8 tasks x 8 completions, lr 5e-5), seeds 0-3 for both arms, `--prm-weight 1`, the PRM
of verifier seed 0 for all four PRM-arm runs (so the PRM arm differs from the outcome arm only
by the reward; the PRM is frozen and was trained on the starting policy's samples, so it can
drift off distribution as the policy moves). Metric: final-step greedy accuracy on 200 fixed
tasks per digit count (`reason-eval --n 200`), primary cell 3 digits. Same verdict rule: a
difference is measurable only if the two arms' 4-seed ranges do not overlap and the gap of
means is > 0.05. One PRM, so the PRM-arm seeds do not cover verifier variance.

```bash
minigpt grpo --base $P --seed $s --out runs/grpo_outcome_s$s/ckpt.pt
minigpt grpo --base $P --seed $s --prm runs/prm_s0/prm.pt --out runs/grpo_prm_s$s/ckpt.pt
minigpt reason-eval --checkpoint runs/grpo_<arm>_s$s/ckpt.pt --n 200
```
