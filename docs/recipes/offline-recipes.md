# Offline recipes at toy scale: Llama 3, Tülu 3, R1, R1-Zero, Phi-4-reasoning

Other labs' post-training recipes, re-run on the same toy problem as the rest of this repo:
multi-digit addition with a `<think>` format, a ~3 MB GPT, verifier reward. Every row starts from
the same medium SFT checkpoint (`reason_sft_n10000e3`, greedy 0.68 / 0.62 / 0.30 on 1 / 2 / 3
digits) unless the row says otherwise. Papers give the **shape** of each recipe only; the numbers
below are one-run toy results, not a replication of any lab's claim.

## What was added

- `reason-rft` (`src/minigpt/rft.py`): sample K completions per task from the model (or from a
  fixed teacher with `--sampler`), keep correct, well-formed ones (`--select verifier`) or the
  highest-scored by a reward model (`--select rm`, Llama 2 best-of-K), SFT on them. `--rounds`
  repeats; the best round (by a held-out greedy eval) seeds the next.
- `reason-dpo`: DPO on (correct, incorrect) pairs of the model's own samples, reusing the loss and
  reference handling of `dpo.py`, plus Llama 3's NLL term on the chosen answer (`--nll-alpha`).

## Method

- Accuracy is greedy decoding on 50 fixed tasks per digit count (about ±0.07 per cell). The
  4/5-digit columns are 0.00 for every model and left out.
- Baseline: default `grpo`, 200 steps, from the same base, seeds 0/1/2 (`grpo_mid_s*`), run once
  and shared with the other slices.
- A recipe "beats" or "loses to" the baseline only if seeds 0 and 1 agree and the gap exceeds the
  baseline's own seed spread; otherwise "no measurable difference". All variants ran at seeds 0
  and 1. With 50 tasks, even a 0.12 gap is only about 1.5 standard errors.
- Rollouts = completions sampled for training data. Baseline GRPO: 200 steps x 8 tasks x 8
  completions = 12.8k rollouts. One RFT or DPO sampling pass here is 2000 tasks x 8 = 16k.
- Everything ran on one shared GPU (CUDA). Greedy numbers can differ by a task or two when the
  same command is re-run.

## Results (greedy accuracy, 1 / 2 / 3 digits)

| Row | Seed 0 | Seed 1 | Rollouts |
| --- | --- | --- | --- |
| Medium SFT base | 0.68 / 0.62 / 0.30 | same model | 0 |
| **Baseline: GRPO 200 steps** (seeds 0 / 1 / 2) | 1.00 / 0.96 / 0.76 | 1.00 / 0.92 / 0.74 (s2: 1.00 / 0.90 / 0.74) | 12.8k |
| Llama 3: 3 x (RS, SFT, DPO) | 0.96 / 0.96 / **0.88** | 1.00 / 0.94 / **0.88** | 96k |
| Llama 3 without DPO: 3 x (RS, SFT) | 1.00 / 0.94 / 0.82 | 0.90 / 0.92 / 0.82 | 48k |
| Tülu 3: DPO, then GRPO 200 | 1.00 / 1.00 / 0.74 | 1.00 / 0.92 / 0.64 | 29k |
| (Tülu 3 after DPO, before RL) | 1.00 / 0.90 / 0.24 | 1.00 / 0.88 / 0.22 | 16k |
| R1: RS-SFT of the GRPO model, then GRPO 100 | 1.00 / 0.94 / 0.76 | 1.00 / 0.96 / 0.70 | 35k incl. baseline |
| (R1 after RS-SFT, before RL) | 1.00 / 0.96 / 0.80 | 1.00 / 0.98 / 0.82 | |
| Phi-4: SFT on teacher traces, then GRPO 100 | 1.00 / 1.00 / **0.88** | 1.00 / 0.98 / **0.98** | 22k + teacher |
| (Phi-4 after SFT, before RL) | 1.00 / 0.92 / 0.88 | 1.00 / 0.98 / 0.92 | 16k + teacher |
| R1-Zero: GRPO 200 from the pretrain base | 0.00 / 0.00 / 0.00 | not run | 12.8k |
| RS-SFT, verifier selection, 1 round | 1.00 / 0.78 / 0.50 | 1.00 / 0.90 / 0.66 | 16k |
| RS-SFT, reward-model selection, 1 round | 0.80 / 0.50 / 0.24 | 0.62 / 0.36 / 0.30 | 16k |

Notes on the rows:

- Seed 1 of the 1-round selection rows is the `reason-rft` run with `--seed 1`; the round-1 data
  of the Llama 3 and R1 chains differs by seed, so seeds are not paired.
- R1 "RS-SFT" samples correct traces from the baseline GRPO model of the same seed
  (`grpo_mid_s0` / `_s1`) and fine-tunes the medium base on them, as R1 retrains from the base.
- Phi-4's teacher is `runs/grpo/ckpt.pt`, an earlier GRPO model of this repo (97% of its samples
  are correct); its training compute is not in the rollout column.
- R1-Zero was run at seed 0 only: its failure (below) does not depend on the seed.

### Llama 3 per round (greedy, after each step)

| Round | Step | Seed 0 | Seed 1 |
| --- | --- | --- | --- |
| 1 | RS, SFT | 1.00 / 0.90 / 0.66 | 0.96 / 0.84 / 0.58 |
| 1 | DPO | 0.90 / 0.88 / 0.56 | 1.00 / 0.98 / 0.66 |
| 2 | RS, SFT | 1.00 / 1.00 / 0.74 | 1.00 / 0.96 / 0.74 |
| 2 | DPO | 1.00 / 0.92 / 0.82 | 1.00 / 0.94 / 0.82 |
| 3 | RS, SFT | 1.00 / 0.98 / 0.94 | 1.00 / 1.00 / 0.92 |
| 3 | DPO | 0.96 / 0.96 / 0.88 | 1.00 / 0.94 / 0.88 |

Sample accuracy of the model that is sampled: 0.33, 0.52-0.53, 0.58-0.64 (rounds 1, 2, 3).

### DPO settings, from the round-1 RS-SFT checkpoint of seed 0

Chosen on a different eval seed (1) than the one reported in the table above (0). The
checkpoint before any DPO scores 0.94 / 0.96 / 0.78 on eval seed 1 and 1.00 / 0.90 / 0.66 on
eval seed 0.

| lr | NLL weight | epochs | Eval seed 0 | Eval seed 1 |
| --- | --- | --- | --- | --- |
| 1e-5 (default) | 0.2 | 2 | 0.62 / 0.76 / 0.32 | 0.56 / 0.60 / 0.36 |
| 5e-6 | 0.2 | 1 | 0.58 / 0.40 / 0.14 | 0.74 / 0.40 / 0.34 |
| 5e-6 | 1.0 | 1 | 0.78 / 0.86 / 0.48 | 0.92 / 0.78 / 0.58 |
| 2e-6 | 0.2 | 1 | 0.84 / 0.82 / 0.44 | 0.96 / 0.86 / 0.54 |
| 2e-6 | 1.0 | 1 | 0.90 / 0.88 / 0.56 | 1.00 / 0.92 / 0.72 |
| 1e-6 | 0.2 | 1 | 1.00 / 0.90 / 0.62 | 1.00 / 0.96 / 0.74 |
| 1e-6 | 1.0 | 1 | 1.00 / 0.90 / 0.62 | 1.00 / 0.96 / 0.74 |

The chains above use lr 2e-6, NLL weight 1.0, 1 epoch. An earlier sweep of this slice (MPS,
lost with that run) picked it; on Korea the sweep above only says that gentler is better and
that 1e-6 is nominally best, within noise of 2e-6.

## Verdicts

1. **DPO as run here does not help, and its default hurts.** With the default setting
   (lr 1e-5, NLL 0.2, 2 epochs) greedy 3-digit accuracy falls 0.66 to 0.32 and format
   breaks (0.92 on 3 digits). No gentler setting beats the checkpoint it started from on 3
   digits; started from: 0.66 (eval seed 0), 0.78 (eval seed 1). In the 3-round chain, the DPO
   step moves 3-digit accuracy by
   -0.10, +0.08, -0.06 (seed 0) and +0.08, +0.08, -0.04 (seed 1): no consistent sign.
   [INFERENCE] The sequence-summed log-probability over ~50-token completions makes a small lr
   move the policy a lot, which is why the NLL term helps.
2. **Llama 3 beats the GRPO baseline on 3 digits, and the gain comes from the RS rounds.** 0.88 /
   0.88 against 0.76 / 0.74 / 0.74 (both seeds agree, gap 0.12-0.14, baseline spread 0.02; still
   only about 1.5 standard errors). The same chain without DPO reaches 0.82 / 0.82: no measurable
   difference from the DPO chain. It costs 4x to 7.5x the baseline's rollouts, but the RS rounds
   are batched generation plus a short SFT, about a minute each on the shared GPU against 8 to 17
   minutes per 200 GRPO steps. Each round's samples are more accurate (0.33, 0.52, 0.60), which is
   the expert-iteration effect.
3. **Tülu 3 (SFT, DPO, RLVR): no measurable difference from the baseline.** DPO from the base
   lifts 2 digits (0.62 to 0.90 / 0.88) but wrecks 3 digits (0.30 to 0.24 / 0.22); GRPO then
   repairs it to 0.74 / 0.64 against 0.76 / 0.74. Seed 0 is level with the baseline, seed 1 is
   0.10 below, so the seeds disagree.
4. **R1 (cold SFT, RL, RS-SFT, RL): no measurable difference from the baseline.** The RS-SFT
   stage on the RL model's correct samples gets 0.80 / 0.82 on 3 digits, level with or a little
   above the RL model it distils (0.76 / 0.74), in one cheap SFT. The final RL stage then does not
   improve it (0.76 / 0.70).
5. **R1-Zero fails on this base.** From the pretrain base the policy never produces `<think>`:
   format 0.00, reward 0.00 and drop rate 1.0 at every step (all groups have equal rewards, so
   every group is dropped and there is no gradient); 200 steps change nothing, and all samples run
   to the 80-token cap. Greedy output of the pretrain model for `12+34=` is `101\n4+4=1\n4+4=10...`:
   it continues arithmetic lines. R1-Zero needs a base that emits the format with non-zero
   probability; this one does not.
6. **Phi-4-reasoning (SFT on verifier-filtered teacher traces, then short RL) beats the baseline on 3
   digits**: 0.88 / 0.98 against 0.76 / 0.74 / 0.74, both seeds, gaps 0.12 and 0.24. Almost all of
   it comes from the SFT step (0.88 / 0.92); the 100 RL steps add 0.00 and +0.06. This is
   distillation from a stronger, already RL-trained teacher, so it is not comparable on compute
   with the baseline: the same SFT with its own samples as the data (RS-SFT, row 1-round
   verifier) reaches only 0.50 / 0.66.
7. **Reward-model selection loses to verifier selection by a large margin.** With RM best-of-K
   (K=8, keep 2), 26% of the kept completions are correct although 33% of all samples are, so the
   RM picks slightly worse than a random pick (about 0.33) on this task. SFT on them ends at
   0.50 / 0.24 and 0.36 / 0.30 on 2 / 3 digits (seeds 0 / 1), at or below the base (0.62 /
   0.30), while verifier selection gives 0.78 / 0.50 and 0.90 / 0.66. Both seeds agree. One
   confound: RM selection keeps 2 per task (about 3960 examples) whereas the verifier keeps
   about 1960 (many tasks have fewer than 2 distinct correct samples), so the RM rows also got
   twice the SFT steps.

## Commands

All from `reason_sft_n10000e3` as `$M`, `$R` = the runs directory, defaults unless shown.

```bash
# Llama 3, round r = 1..3 (prev starts at $M; seed = r for seed 0, 10 + r for seed 1)
minigpt reason-rft --base $prev --select verifier --seed $s --out $R/llama3_r${r}_rft/ckpt.pt
minigpt reason-dpo --base $R/llama3_r${r}_rft/ckpt.pt --seed $s \
  --lr 2e-6 --nll-alpha 1.0 --epochs 1 --out $R/llama3_r${r}/ckpt.pt

# Tülu 3
minigpt reason-dpo --base $M --lr 2e-6 --nll-alpha 1.0 --epochs 1 --seed $s --out $R/tulu_dpo/ckpt.pt
minigpt grpo --base $R/tulu_dpo/ckpt.pt --steps 200 --seed $s --out $R/tulu/ckpt.pt

# R1-Zero
minigpt grpo --base $R/reason_pretrain/ckpt.pt --steps 200 --out $R/r1zero/ckpt.pt

# R1 (stage 1-2 = baseline grpo_mid_s$s)
minigpt reason-rft --base $M --sampler $R/grpo_mid_s$s/ckpt.pt --select verifier --seed $s \
  --out $R/r1_rft/ckpt.pt
minigpt grpo --base $R/r1_rft/ckpt.pt --steps 100 --seed $s --out $R/r1/ckpt.pt

# Phi-4-reasoning
minigpt reason-rft --base $M --sampler $R/grpo/ckpt.pt --select verifier --seed $s \
  --out $R/phi4_sft/ckpt.pt
minigpt grpo --base $R/phi4_sft/ckpt.pt --steps 100 --seed $s --out $R/phi4/ckpt.pt

# RM vs verifier selection
minigpt reason-rft --base $M --select verifier --seed $s --out $R/rft_ver/ckpt.pt
minigpt reason-rft --base $M --select rm --rm $R/rm_v3/ckpt.pt --seed $s --out $R/rft_rm/ckpt.pt
```

Logs of every run in this doc: `runs/logs/off_*.log` (checkpoints are not kept).

## Sources (recipe shape only)

- Llama 3: arXiv 2407.21783 section 4 (rounds of rejection sampling, SFT, DPO with an NLL term).
- Llama 2: arXiv 2307.09288 section 3.2.3 (best-of-K by reward model).
- Tülu 3: arXiv 2411.15124; OLMo 3: arXiv 2512.13961 (SFT, DPO, RLVR).
- DeepSeek-R1 / R1-Zero: arXiv 2501.12948 sections 2.2 and 2.3.
- Phi-4-reasoning: arXiv 2504.21318 (SFT on curated teacher traces, then short RL).
