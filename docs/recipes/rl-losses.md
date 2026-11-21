# RL policy-loss recipes from other labs, at toy scale

GRPO (DeepSeek) is the repo's RL baseline. This page adds the loss / advantage / reference
choices that other labs published on top of it, each behind a flag of `minigpt grpo`, and runs
each at seeds 0 and 1 on the same task. All flags default to the old behaviour.

Setup for every run: the addition task (1 to 3 digits), start from the
medium SFT base `runs/reason_sft_n10000e3` (greedy 0.68 / 0.62 / 0.30 on 1 / 2 / 3 digits),
8 tasks x 8 completions per step, 200 steps, lr 5e-5, temperature 1.0, `--eval-n 50`, one GPU
(CUDA). Only the flags listed below differ.

## Methods and flags

| Recipe | Source | What changes | Flags |
|---|---|---|---|
| GRPO baseline | DeepSeekMath, [arXiv 2402.03300](https://arxiv.org/abs/2402.03300) | per-token PPO clip 0.2, per-sequence token mean, `(r-mean)/std`, k3 KL beta 0.04 | defaults |
| GSPO | Qwen, [arXiv 2507.18071](https://arxiv.org/abs/2507.18071) eq. 5-7 | one importance ratio per sequence `s = exp(mean_t log pi/pi_old)`, clipped per sequence at 3e-4 / 4e-4 (paper section 5.1), group-normalised advantage. KL is not discussed ("omitted for brevity"), so beta stays 0.04 | `--policy-loss gspo --clip-eps 3e-4 --clip-eps-high 4e-4` |
| CISPO | MiniMax-M1, [arXiv 2506.13585](https://arxiv.org/abs/2506.13585) section 3.1 eq. 4-5 | `sg(clip(r)) * A * log pi`: a clipped token keeps a gradient scaled by the clipped weight, instead of dropping it. Token-level normalisation, no lower IS bound, no KL, dynamic sampling. eps_high is "tuned" and not given: 0.28 here | `--policy-loss cispo --loss-agg token-mean --clip-eps 1.0 --clip-eps-high 0.28 --beta 0 --fill-batch` |
| DAPO | ByteDance Seed, [arXiv 2503.14476](https://arxiv.org/abs/2503.14476) | clip-higher (eps_low 0.2, eps_high 0.28), token-level loss: sum over all tokens of the batch / total tokens (eq. 8), dynamic sampling, no KL. Overlong reward shaping is not used (completions are short) | `--clip-eps-high 0.28 --loss-agg token-mean --beta 0 --fill-batch` |
| Magistral | Mistral, [arXiv 2506.10910](https://arxiv.org/abs/2506.10910) section 2.1 | no KL, clip-higher (0.28), loss = sum of token losses / total length of the group, advantage `r - mean` (no group std) then divided by the minibatch std, groups with equal rewards filtered (already always done here) | `--beta 0 --clip-eps-high 0.28 --loss-agg group-mean --advantage batch` |
| RLOO / REINFORCE | Ahmadian et al., [arXiv 2402.14740](https://arxiv.org/abs/2402.14740) (also Kimi k1.5) | advantage `r_i - mean(r_-i)`, no std, no clipping. KL kept (beta 0.04) | `--advantage rloo --mu 1` |
| ProRL | NVIDIA, [arXiv 2505.24864](https://arxiv.org/abs/2505.24864) section 2.3.1 | every N steps hard-reset the KL reference to the current policy and re-initialise the optimizer; KL kept. The paper resets when the KL term dominates, with no fixed interval: N = 50 here | `--ref-reset-every 50` |

Implementation: `grpo_loss` (`--policy-loss`, `--clip-eps-high`, `--loss-agg token-mean|group-mean`),
`group_advantages` (`--advantage group|batch|rloo`) and `train_grpo` (`--ref-reset-every`, which
copies the reference, so the "SFT start" evaluation printed after training is still the base).
One test per function in `tests/test_minigpt_rl_losses.py`.

Two honest notes on what can differ at all:

- With `--mu 1` the importance ratio is exactly 1 in value, so no clip ever fires and GSPO
  reduces to GRPO (same gradient: `A/|y| * sum grad log pi`), and CISPO to token-mean
  REINFORCE. The clipping variants only mean something with `--mu > 1`, so the experiment uses
  `--mu 4` (four optimizer steps per batch) for everything except RLOO. That makes the shared
  default baseline (`mu 1`) a different setting, so a `mu 4` GRPO baseline is run as well.
- The recipes change several things at once (KL, normalisation, sampling). Each row is the lab's
  recipe as a bundle, not an ablation of one idea.

## Results

Greedy accuracy on the fixed eval tasks after RL (50 tasks per digit count, so about +-0.07 per
cell). Every cell shows seed 0 / seed 1; the shared default baseline also has seed 2. The base
starts at 0.68 / 0.62 / 0.30.

- `tokens`: mean greedy completion length over all 1 to 5 digit eval tasks (the base is 49.5).
- `train acc`: mean sampled-completion accuracy over the logged steps 100 to 190.
- `KL`: mean k3 KL to the reference over the logged steps 10 to 190. The step-0 line is left
  out: it is large only for the `--beta 0` runs and is not meaningful.
- `wall`: seconds on one shared, busy GPU, so only roughly comparable.

| Recipe | 1-digit | 2-digit | 3-digit | tokens | train acc | KL | wall (s) |
|---|---|---|---|---|---|---|---|
| GRPO default (`mu 1`, shared baseline), seeds 0 / 1 / 2 | 1.00 / 1.00 / 1.00 | 0.96 / 0.92 / 0.90 | 0.76 / 0.74 / 0.74 | 51.1 / 51.6 / 50.3 | 0.58 / 0.58 / 0.56 | 0.021 / 0.022 / 0.018 | 273 / 274 / 272 |
| GRPO `mu 4` | 1.00 / 1.00 | 0.86 / 0.92 | 0.68 / 0.60 | 50.4 / 49.3 | 0.51 / 0.60 | 0.031 / 0.025 | 322 / 292 |
| GSPO | 1.00 / 0.94 | 0.96 / 0.94 | 0.66 / 0.66 | 50.6 / 51.2 | 0.58 / 0.62 | 0.020 / 0.019 | 284 / 284 |
| CISPO | 0.80 / 0.74 | 0.84 / 0.28 | 0.34 / 0.10 | 50.5 / 51.6 | 0.36 / 0.34 | 0.095 / 0.116 | 519 / 537 |
| DAPO | 0.98 / 1.00 | 0.82 / 0.86 | 0.50 / 0.52 | 50.3 / 49.5 | 0.58 / 0.55 | 0.057 / 0.039 | 479 / 478 |
| Magistral | 0.94 / 0.90 | 0.86 / 0.88 | 0.44 / 0.52 | 50.5 / 51.3 | 0.52 / 0.57 | 0.058 / 0.042 | 293 / 356 |
| RLOO (`mu 1`) | 0.90 / 1.00 | 0.84 / 0.94 | 0.76 / 0.78 | 51.7 / 50.9 | 0.56 / 0.60 | 0.018 / 0.016 | 272 / 289 |
| ProRL | 1.00 / 1.00 | 0.88 / 0.80 | 0.54 / 0.66 | 50.4 / 50.5 | 0.53 / 0.56 | 0.020 / 0.021 | 316 / 288 |

Verdict rule: a recipe "beats" or "loses to" its baseline only if both of its seeds are on the
same side of both baseline seeds and the seed-mean gap is above 0.07 (the n = 50 noise).
Otherwise: no measurable difference. The baseline is `GRPO default` for RLOO (same `mu 1`) and
`GRPO mu 4` for everything else (same `mu 4`). The 3-digit column is the hardest and carries
most of the signal; 1 and 2 digits are at or near the ceiling.

| Recipe | Verdict vs its baseline | Evidence |
|---|---|---|
| `mu 4` vs `mu 1` | loses (3-digit, weakly) | 3-digit 0.68 / 0.60 vs 0.76 / 0.74 / 0.74 (mean gap 0.11, just above the noise); 1 and 2 digits within noise. This is why the other recipes are compared to `mu 4` |
| GSPO | no measurable difference | 3-digit 0.66 vs 0.64, 2-digit 0.95 vs 0.89 (gap 0.06, below 0.07); KL and train acc unchanged |
| CISPO | loses | 3-digit 0.34 / 0.10 vs 0.68 / 0.60, train acc 0.36 / 0.34 vs 0.51 / 0.60, KL 4x higher, 2-digit collapses on seed 1 (0.28) |
| DAPO | loses (3-digit) | 3-digit 0.50 / 0.52 vs 0.68 / 0.60, 2-digit 0.82 / 0.86 vs 0.86 / 0.92; train acc matches the baseline; 1.6x wall time from the extra sampling |
| Magistral | loses (3-digit) | 3-digit 0.44 / 0.52 vs 0.68 / 0.60, 1-digit 0.94 / 0.90 vs 1.00 / 1.00 |
| RLOO | no measurable difference | 3-digit 0.76 / 0.78 vs 0.76 / 0.74; 1-digit seed 0 is 0.90 vs 1.00 (noise, seed 1 is 1.00) |
| ProRL | no measurable difference | 3-digit 0.54 / 0.66 vs 0.68 / 0.60; KL and train acc equal to the baseline |

What this does and does not show:

- The three recipes that remove the KL term (CISPO, DAPO, Magistral) are the three that lose
  at 3 digits, and their KL is 1.6 to 4.6 times the `mu 4` baseline's. The next section tests
  whether the missing KL term is the cause.
- CISPO never zeroes a gradient, so with `mu 4` and no KL it keeps pushing on every token of
  every inner step. [INFERENCE] This is the likely cause of its drift, not tested.
- One tiny task (3 digits are the only column with room), one base model, two seeds per recipe
  (three for the default baseline), 200 steps. These are one-run toy results. They do not say
  anything about how the labs' recipes behave at their own scale. The recipe shapes
  follow the papers cited above; the hyperparameters (eps_high of CISPO, `N = 50` of ProRL) are
  guesses where the paper gives none.

### KL isolation

Four more runs change only `--beta`: GRPO `mu 4` with the KL term removed, and the three no-KL
recipes with `--beta 0.04` restored. Same base, 200 steps, seeds 0 / 1, same verdict rule.

| Run | 1-digit | 2-digit | 3-digit | KL | Verdict (3-digit) |
|---|---|---|---|---|---|
| GRPO `mu 4`, `--beta 0` | 1.00 / 0.96 | 0.68 / 0.64 | 0.44 / 0.52 | 0.052 / 0.034 | loses to GRPO `mu 4` with KL (0.68 / 0.60); 2-digit also loses (0.86 / 0.92) |
| DAPO + KL | 1.00 / 1.00 | 0.92 / 0.92 | 0.82 / 0.62 | 0.028 / 0.028 | beats DAPO (0.50 / 0.52); no measurable difference to GRPO `mu 4` |
| Magistral + KL | 1.00 / 1.00 | 0.88 / 0.90 | 0.60 / 0.38 | 0.035 / 0.028 | no measurable difference to Magistral (0.44 / 0.52); 1-digit recovers from 0.94 / 0.90 |
| CISPO + KL | 0.82 / 0.86 | 0.60 / 0.54 | 0.36 / 0.42 | 0.042 / 0.047 | beats CISPO (0.34 / 0.10), seed 0 only by 0.02; still loses to GRPO `mu 4` |

- The KL term matters here: removing it from GRPO costs 0.16 at 3 digits and 0.23 at 2 digits,
  and restoring it in DAPO closes DAPO's whole gap to GRPO.
- It is not the whole story. CISPO with KL still loses to GRPO, and Magistral with KL is not
  measurably better at 3 digits. Their other changes (unclipped-gradient weighting, batch-std
  advantage, group-length normalisation) are not isolated one by one.
- Two seeds, n = 50 per cell, one base model. DAPO + KL seed 0 (0.82) is the single highest
  3-digit cell in this file; one cell is not evidence that DAPO + KL beats GRPO.

```bash
minigpt grpo $B --mu 4 --beta 0 --out runs/kl_grpo_nokl/ckpt.pt
minigpt grpo $B --mu 4 --clip-eps-high 0.28 --loss-agg token-mean --beta 0.04 --fill-batch \
  --out runs/kl_dapo_kl/ckpt.pt
minigpt grpo $B --mu 4 --beta 0.04 --clip-eps-high 0.28 --loss-agg group-mean \
  --advantage batch --out runs/kl_magistral_kl/ckpt.pt
minigpt grpo $B --mu 4 --policy-loss cispo --loss-agg token-mean --clip-eps 1.0 \
  --clip-eps-high 0.28 --beta 0.04 --fill-batch --out runs/kl_cispo_kl/ckpt.pt
```

`$B` is defined in the Reproduce block below. Logs: `runs/logs/kl_*.log`.

Reproduce (from the repo root, base checkpoint in `runs/`):

```bash
B="--base runs/reason_sft_n10000e3/ckpt.pt --steps 200 --seed 0"
minigpt grpo $B --mu 4 --out runs/rl_base_mu4/ckpt.pt
minigpt grpo $B --mu 4 --policy-loss gspo --clip-eps 3e-4 --clip-eps-high 4e-4 --out runs/rl_gspo/ckpt.pt
minigpt grpo $B --mu 4 --policy-loss cispo --loss-agg token-mean --clip-eps 1.0 \
  --clip-eps-high 0.28 --beta 0 --fill-batch --out runs/rl_cispo/ckpt.pt
minigpt grpo $B --mu 4 --clip-eps-high 0.28 --loss-agg token-mean --beta 0 --fill-batch \
  --out runs/rl_dapo/ckpt.pt
minigpt grpo $B --mu 4 --beta 0 --clip-eps-high 0.28 --loss-agg group-mean --advantage batch \
  --out runs/rl_magistral/ckpt.pt
minigpt grpo $B --advantage rloo --mu 1 --out runs/rl_rloo/ckpt.pt
minigpt grpo $B --mu 4 --ref-reset-every 50 --out runs/rl_prorl/ckpt.pt
```

Seed 1 runs add `--seed 1` (the later flag wins); the shared baseline is the plain `minigpt grpo
--base ...` run with seeds 0, 1, 2. Logs: `runs/logs/rl_*.log`, `runs/logs/grpo_mid_s*.log`.
