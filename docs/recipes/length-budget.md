# Length-control reward shaping at toy scale

Three length-control rewards from other labs, added to `minigpt grpo` and run on the toy
addition task (1-3 digit questions, 5-digit questions are out of range for this model).
All run from the same SFT checkpoint with the default GRPO settings, 200 steps. One run per
cell, so read this as a smoke test of each recipe, not a benchmark.

## Sources (recipe shape only)

| Flag | Recipe | Source |
|---|---|---|
| `--budget-tau T1,T2,..` | Per-problem budget `b0(x)` from the cold-start model; a trajectory with more than `tau * b0(x)` thinking tokens gets task reward `-1` even if correct; `tau` is annealed in stages | Kimi K3 report, "Reasoning Effort RL" ([arXiv 2607.24653](https://arxiv.org/abs/2607.24653)) |
| `--length-reward W` `--length-warmup N` | In-group length reward `lambda = 0.5 - (len - min)/(max - min)`; correct answers get `lambda`, wrong ones `min(0, lambda)`; weight `W`, off for the first `N` steps | Kimi K1.5 section 2.3.3 ([arXiv 2501.12599](https://arxiv.org/abs/2501.12599)) |
| `--overlong-cache L` | Soft penalty ramping from 0 to -1 over the last `L` of `--max-new-tokens` (80), -1 at the cap | DAPO section 3.4 ([arXiv 2503.14476](https://arxiv.org/abs/2503.14476)) |

Toy-scale readings: `b0` is one number per (effort level, digit count), the mean thinking
tokens of the SFT model over 8 tasks x 8 samples, not one per problem (a per-problem
budget would double the rollouts). The tokenizer is bytewise, so characters = tokens. The
shaping runs in a group-level hook (`make_shaper` in `src/minigpt/grpo.py`) after the
existing effort penalty; the loss is untouched and every flag is off by default.
`tests/test_minigpt_length.py` covers the reward functions, the composition order and
the hook in `train_grpo`.

## Setup

- Machine: Korea server, one shared CUDA GPU (GPU 3), other jobs on it, so wall times are noisy.
- Medium base: `runs/reason_sft_n10000e3/ckpt.pt` (greedy accuracy 0.68 / 0.62 / 0.30 on
  1 / 2 / 3 digits, think 7.5 / 16.0 / 24.9 tokens). Effort base: `runs/effort_sft/ckpt.pt`
  (SFT with `e1:`/`e9:` prefixes and non-think traces mixed in).
- Eval: greedy, 50 fixed tasks per digit count (a cell is about +-0.07).
- Baseline: default `grpo --steps 200` from the medium base, seeds 0/1/2, run once for all
  recipes (`runs/logs/grpo_mid_s{0,1,2}.log`). The effort baseline (`--efforts 1,9
  --effort-k0 0`, so no effort penalty) is the `eff_base` row.
- Logs: `runs/logs/lb_<name>.log` (`_s1` suffix = seed 1). Wall-clock is the last line.

```bash
B=runs/reason_sft_n10000e3/ckpt.pt   # medium base
E=runs/effort_sft/ckpt.pt            # effort base
minigpt grpo --base $B --steps 200 --budget-tau 1.5,1.25,1.0 --out runs/lb_k3_anneal/ckpt.pt
minigpt grpo --base $B --steps 200 --budget-tau 1.25         --out runs/lb_k3_t125/ckpt.pt
minigpt grpo --base $B --steps 200 --budget-tau 1.25,0.8     --out runs/lb_k3_stress/ckpt.pt
minigpt grpo --base $B --steps 200 --length-reward 0.5 --length-warmup 50 --out runs/lb_k15/ckpt.pt
minigpt grpo --base $B --steps 200 --overlong-cache 20       --out runs/lb_dapo20/ckpt.pt
minigpt grpo --base $B --steps 200 --overlong-cache 40       --out runs/lb_dapo40/ckpt.pt
# effort-conditioned (same flags plus):
minigpt grpo --base $E --steps 200 --efforts 1,9 --effort-k0 0 [--budget-tau 1.5,1.0 | --length-reward 0.5 --length-warmup 50]
# seed 1: add --seed 1
```

`--budget-tau` gives one multiplier per equal stage of `--steps`; measured `b0` (think tokens,
medium base) is 7.7 / 16.6 / 24.4 for 1 / 2 / 3 digits.

## Results: medium base

Greedy accuracy and mean thinking tokens per digit count after RL; `tok` and `think` are the
means over all five digit counts (1-5), so they include the out-of-range 4-5 digit questions.
SFT start: acc 0.68 / 0.62 / 0.30, think 7.5 / 16.0 / 24.9, overall tok 49.5, think 14.3.

| Run | seed | acc 1/2/3 digits | think 1/2/3 | overall acc | tok | think | wall |
|---|---|---|---|---|---|---|---|
| baseline | 0 | 1.00 / 0.96 / 0.76 | 7.5 / 16.0 / 24.8 | 0.544 | 51.1 | 15.5 | 273 s |
| baseline | 1 | 1.00 / 0.92 / 0.74 | 7.5 / 16.0 / 24.8 | 0.532 | 51.6 | 15.3 | 274 s |
| baseline | 2 | 1.00 / 0.90 / 0.74 | 7.5 / 16.0 / 24.8 | 0.528 | 50.3 | 14.6 | 272 s |
| K3 tau 1.25 | 0 | 1.00 / 0.98 / 0.72 | 7.5 / 16.0 / 24.7 | 0.540 | 51.3 | 15.9 | 371 s |
| K3 tau 1.25 | 1 | 1.00 / 0.96 / 0.78 | 7.5 / 16.0 / 24.8 | 0.548 | 50.3 | 15.1 | 391 s |
| K3 anneal 1.5,1.25,1.0 | 0 | 0.60 / 0.54 / 0.12 | 6.2 / 14.9 / 15.6 | 0.252 | 49.4 | 11.5 | 379 s |
| K3 anneal 1.5,1.25,1.0 | 1 | 0.58 / 0.80 / 0.30 | 6.6 / 15.5 / 21.4 | 0.336 | 49.8 | 14.3 | 543 s |
| K3 stress 1.25,0.8 | 0 | 0.94 / 0.72 / 0.60 | 7.5 / 16.0 / 24.8 | 0.452 | 51.2 | 15.7 | 351 s |
| K3 stress 1.25,0.8 | 1 | 0.58 / 0.50 / 0.00 | 0.0 / 3.5 / 11.5 | 0.216 | 48.1 | 7.3 | 467 s |
| K1.5 length 0.5 | 0 | 1.00 / 0.92 / 0.66 | 7.5 / 16.0 / 24.8 | 0.516 | 50.7 | 15.4 | 367 s |
| K1.5 length 0.5 | 1 | 0.98 / 0.88 / 0.68 | 7.5 / 16.0 / 24.8 | 0.508 | 50.7 | 15.5 | 373 s |
| DAPO overlong 20 | 0 | 0.98 / 0.88 / 0.70 | 7.5 / 16.0 / 24.8 | 0.512 | 50.6 | 14.5 | 388 s |
| DAPO overlong 20 | 1 | 1.00 / 0.88 / 0.86 | 7.5 / 16.0 / 24.8 | 0.548 | 51.3 | 15.3 | 364 s |
| DAPO overlong 40 | 0 | 1.00 / 0.96 / 0.00 | 7.5 / 16.0 / 8.5 | 0.392 | 36.5 | 9.7 | 342 s |
| DAPO overlong 40 | 1 | 1.00 / 0.94 / 0.82 | 7.5 / 16.0 / 24.8 | 0.552 | 51.7 | 15.5 | 433 s |

Baseline spread over three seeds: overall 0.528-0.544, 3 digits 0.74-0.76, think unchanged at
7.5 / 16.0 / 24.8. Wall time of the baseline is not comparable (it ran on another GPU and
load); do not read the 270 s vs 350-540 s gap as the cost of shaping.

## Results: effort-conditioned base

Same eval, per effort level (`e1` low, `e9` high). `--effort-k0 0` means no effort length penalty,
so only the new shaping can separate the levels. SFT start: e1 acc 0.86 / 0.74 / 0.58, think
6.1 / 14.4 / 19.4; e9 acc 0.94 / 0.78 / 0.60, think 6.6 / 14.7 / 19.4.

| Run | seed | level | acc 1/2/3 digits | think 1/2/3 | overall acc | tok | think | wall |
|---|---|---|---|---|---|---|---|---|
| eff baseline | 0 | e1 | 1.00 / 1.00 / 1.00 | 7.5 / 16.0 / 24.8 | 0.600 | 48.8 | 13.1 | 687 s |
| eff baseline | 0 | e9 | 1.00 / 1.00 / 0.98 | 7.5 / 16.0 / 24.8 | 0.596 | 48.4 | 13.3 | |
| eff baseline | 1 | e1 | 1.00 / 1.00 / 1.00 | 7.5 / 16.0 / 24.8 | 0.600 | 49.4 | 13.9 | 845 s |
| eff baseline | 1 | e9 | 1.00 / 1.00 / 1.00 | 7.5 / 16.0 / 24.8 | 0.600 | 49.2 | 14.0 | |
| eff K3 1.5,1.0 | 0 | e1 | 0.98 / 0.10 / 0.02 | 0 / 0 / 0 | 0.220 | 35.0 | 0.0 | 635 s |
| eff K3 1.5,1.0 | 0 | e9 | 0.90 / 0.14 / 0.02 | 0 / 0 / 0 | 0.212 | 34.9 | 0.0 | |
| eff K3 1.5,1.0 | 1 | e1 | 1.00 / 0.10 / 0.02 | 0 / 0 / 0 | 0.224 | 34.9 | 0.0 | 784 s |
| eff K3 1.5,1.0 | 1 | e9 | 1.00 / 0.12 / 0.02 | 0 / 0 / 0 | 0.228 | 34.7 | 0.0 | |
| eff K1.5 0.5 | 0 | e1 | 1.00 / 0.98 / 0.90 | 7.5 / 16.0 / 24.8 | 0.576 | 48.5 | 13.2 | 732 s |
| eff K1.5 0.5 | 0 | e9 | 1.00 / 0.98 / 0.90 | 7.5 / 16.0 / 24.8 | 0.576 | 48.3 | 13.3 | |
| eff K1.5 0.5 | 1 | e1 | 1.00 / 1.00 / 1.00 | 7.5 / 16.0 / 24.8 | 0.600 | 49.8 | 14.0 | 745 s |
| eff K1.5 0.5 | 1 | e9 | 1.00 / 1.00 / 1.00 | 7.5 / 16.0 / 24.8 | 0.600 | 49.4 | 14.0 | |

(In the effort base, `b0` per (level, digits) was 5.9 / 11.4 / 14.3 for e1 and 4.6 / 9.8 / 14.8
for e9 at seed 0; 3-digit budgets are 40-45% below the 24.8 tokens the model learns in RL.)

## Verdicts

- **K3 budget, tau 1.25: no measurable difference.** Overall 0.540 / 0.548 vs baseline
  0.528-0.544, think unchanged. At 1.25x the mean, the budget rarely binds: the SFT traces
  have almost fixed length per digit count, so there is little over-budget mass to cut.
- **K3 budget, anneal to tau 1.0 or 0.8: worse, both seeds, and it does not make the model think
  less in any useful way.** Accuracy falls (overall 0.25 / 0.34 for the anneal, 0.45 / 0.22
  for stress vs 0.53-0.54) while think shrinks only where the model stops thinking or
  formatting (stress s1: think 0.0 on 1 digit, format 0.00). With `b0` set at the mean, tau 1.0 puts
  roughly half of the correct samples at reward -1, and a wrong short answer (reward 0)
  then beats a correct long one (-1). [INFERENCE, not checked sample by sample.]
- **K3 budget on the effort base: collapses to no thinking, both seeds.** Think drops to 0 at both
  levels, 1-digit stays at 0.90-1.00 but 2 / 3 digits fall to 0.10-0.14 / 0.02 (baseline 1.00).
  Levels do not separate. It is the same "wrong-and-short beats correct-and-long" effect, with
  tighter budgets (14.3 vs 24.4 at 3 digits). In the paper the budgets come per problem
  and `tau` is tuned by hand per domain; here neither holds.
- **K1.5 in-group length reward: no measurable difference.** Overall 0.516 / 0.508 vs 0.528-0.544
  is a gap of 1-2 tasks over 250 and 3-digit accuracy 0.66 / 0.68 vs 0.74-0.76 is within the
  +-0.07 grid; think is unchanged. Likely cause: correct traces have the same length in a
  group (hi == lo gives zero reward), so there is nothing to rank. Effort base: also no
  measurable difference (seed 0 3-digit 0.90 vs 1.00 does not reproduce at seed 1).
- **DAPO overlong, L=20: no measurable difference** (0.512 / 0.548, think unchanged). The soft
  zone starts at 60 tokens; only 3-digit completions (61 on average) reach it, and the
  penalty there is small (about -0.07 at the mean length).
- **DAPO overlong, L=40: unstable, not a clear loss.** Seed 0 collapsed on 3 digits (acc 0.00,
  format 0.00, tokens 16.5) while seed 1 matched the baseline (0.552). One collapsed seed
  out of two is not a verdict either way; a third seed would settle it. The zone starts at 40,
  below the 42-61 token traces, so every sample is penalised and a degenerate short
  completion can win. [INFERENCE]
- **Overall: none of the three shortens thinking at no accuracy cost on this task.** The
  column-addition traces are already minimal for their digit count (think 7.5 / 16.0 / 24.8
  after SFT and after every neutral run), so there is no overthinking to remove; the only
  way to cut tokens is to cut the answer's accuracy. The recipes target long free-form
  reasoning where lengths vary within a group. A null result here says nothing about them there.

## Limits

- Medium base: seed 0 and 1 for each variant, baseline seeds 0-2. Effort base: seeds 0 and 1.
  Per-cell eval noise is +-0.07, the overall column is about +-0.03.
- `b0` is per difficulty bucket, not per problem; `T(y)` is thinking tokens only.
- Wall times are from a shared GPU with other jobs and are only indicative.
