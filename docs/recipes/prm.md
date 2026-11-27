# Process reward model vs outcome reward model, best-of-N (OpenAI, toy scale)

Status: design and verdict rule written before the four seeds ran; results below are filled in
after the run.

## Source

| What | Source |
|---|---|
| Outcome-supervised RM (ORM): trained on final-answer correctness, scored at the last token | Lightman et al. 2023, "Let's Verify Step by Step", sec. 2.5 ([arXiv 2305.20050](https://arxiv.org/abs/2305.20050), 2023-05-31) |
| Process-supervised RM (PRM): one correct/incorrect prediction after the last token of each step; labels only up to the first wrong step; solution score = product of step probabilities (the minimum is the alternative in appendix F: 77.6% vs 78.2% best-of-1860) | same paper, sec. 2.6 and appendix F |
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
- Held-out tasks: 200 per digit count, 1-3 digits (600), `--seed 7`, never seen by the
  verifiers' sampler seed; 16 samples each, drawn once and cached, identical for every seed.
  1-digit and some 2-digit tasks necessarily repeat prompts from the verifiers' training set (only 100 one-digit problems exist);
  3-digit overlap is expected to be about 0.
- Verifier seeds 0-3, policy, tasks and samples fixed. Pilot runs with seed 100 (not in any
  table) chose lr `1e-4` and 8 epochs from verifier held-out BCE only; the best-of-N held-out
  tasks were not used for tuning. In the pilot, both verifiers' held-out BCE was still falling
  at 6 epochs.
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
