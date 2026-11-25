# RL policy-loss recipes from other labs, at toy scale

GRPO (DeepSeek) is the repo's RL baseline. This page adds the loss / advantage / reference
choices that other labs published on top of it, each behind a flag of `minigpt grpo`, and runs
each at seeds 0 to 3 on the same task. All flags default to the old behaviour.

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
cell). Seeds 0 to 3 for every recipe; the shared `mu 1` default baseline has seeds 0 to 2. The
base starts at 0.68 / 0.62 / 0.30.

- `KL`: mean k3 KL to the reference over the logged steps 10 to 190, averaged over seeds. The
  step-0 line is left out: it is large only for the `--beta 0` runs and is not meaningful.
- Mean greedy completion length (49.3 to 52.3 tokens on every run, against the base's 49.5) and
  train accuracy (seed means 0.53 to 0.58 in this table, except CISPO's 0.35) do not separate the
  recipes and are not shown.

| Recipe | Mean 1 / 2 / 3 digits | 3-digit, seeds 0 / 1 / 2 / 3 | KL | Verdict vs its baseline |
|---|---|---|---|---|
| GRPO default (`mu 1`, shared baseline) | 1.00 / 0.93 / 0.75 | 0.76 / 0.74 / 0.74 (3 seeds) | 0.020 | baseline for RLOO |
| GRPO `mu 4` | 0.98 / 0.88 / 0.57 | 0.68 / 0.60 / 0.50 / 0.50 | 0.029 | baseline for the rest; loses to `mu 1` at 3 digits (gap 0.18) |
| GSPO | 0.98 / 0.92 / 0.67 | 0.66 / 0.66 / 0.70 / 0.66 | 0.020 | no measurable difference |
| CISPO | 0.78 / 0.51 / 0.19 | 0.34 / 0.10 / 0.16 / 0.16 | 0.102 | loses at 2 and 3 digits |
| DAPO | 0.98 / 0.84 / 0.54 | 0.50 / 0.52 / 0.62 / 0.52 | 0.048 | no measurable difference |
| Magistral | 0.91 / 0.86 / 0.53 | 0.44 / 0.52 / 0.66 / 0.48 | 0.066 | no measurable difference |
| RLOO (`mu 1`) | 0.98 / 0.92 / 0.73 | 0.76 / 0.78 / 0.64 / 0.72 | 0.017 | no measurable difference |
| ProRL | 0.98 / 0.80 / 0.58 | 0.54 / 0.66 / 0.54 / 0.56 | 0.021 | no measurable difference |

Verdict rule, fixed before seeds 2 and 3 were run: a recipe "beats" or "loses to" its baseline
in a column only if none of its seed values overlaps any baseline seed value (ties count as
overlap) and the seed-mean gap is above 0.07 (the n = 50 noise). Otherwise: no measurable
difference. The baseline is `GRPO default` for RLOO (same `mu 1`) and `GRPO mu 4` for everything
else (same `mu 4`). The 3-digit column carries most of the signal; 1 and 2 digits are at or near
the ceiling.

What this does and does not show:

- Only CISPO loses (2-digit 0.28-0.84 vs 0.86-0.92, 3-digit 0.10-0.34 vs 0.50-0.68), with 3.5
  times the baseline's KL. CISPO never zeroes a gradient, so with `mu 4` it keeps pushing on
  every token of every inner step. [INFERENCE] This is the likely cause of its drift, not tested.
- The first version of this table used seeds 0 and 1 only and also had DAPO and Magistral losing
  at 3 digits. Seeds 2 and 3 of GRPO `mu 4` scored 0.50 / 0.50 against 0.68 / 0.60 for seeds 0
  and 1, so that pair was the baseline's luckiest; with 4 seeds DAPO and Magistral overlap it.
  A single cell varies by about 0.2 at 3 digits from the seed alone.
- GSPO comes closest to a win: 3-digit mean gap +0.10 with the tightest seed range of any cell
  (0.66-0.70), but baseline seed 0 (0.68) sits above three GSPO seeds. Its KL (0.019-0.020) is
  below every GRPO `mu 4` seed. [INFERENCE] Its tiny clip range (3e-4 / 4e-4) may throttle the
  four inner steps so it behaves more like `mu 1`; the clip fraction is not logged.
- One tiny task, one base model, 200 steps. These are toy results and say nothing about how the
  labs' recipes behave at their own scale. The recipe shapes follow the papers cited above; the
  hyperparameters (eps_high of CISPO, `N = 50` of ProRL) are guesses where the paper gives none.

### KL isolation

Four more cells change only `--beta`: GRPO `mu 4` with the KL term removed, and the three no-KL
recipes with `--beta 0.04` restored. Same base, 200 steps, seeds 0 to 3, same rule.

| Run | Mean 1 / 2 / 3 digits | 3-digit, seeds 0 / 1 / 2 / 3 | KL | Verdict |
|---|---|---|---|---|
| GRPO `mu 4`, `--beta 0` | 0.98 / 0.72 / 0.48 | 0.44 / 0.52 / 0.56 / 0.38 | 0.050 | no measurable difference to GRPO `mu 4` with KL |
| DAPO + KL | 1.00 / 0.91 / 0.63 | 0.82 / 0.62 / 0.48 / 0.58 | 0.029 | no measurable difference to DAPO or to GRPO `mu 4` |
| Magistral + KL | 0.99 / 0.83 / 0.55 | 0.60 / 0.38 / 0.56 / 0.66 | 0.029 | no measurable difference to Magistral or to GRPO `mu 4` |
| CISPO + KL | 0.88 / 0.63 / 0.36 | 0.36 / 0.42 / 0.34 / 0.30 | 0.043 | no measurable difference to CISPO; loses to GRPO `mu 4` at 2 and 3 digits and to DAPO + KL in every column |

- The KL term does what it is for: mean KL is higher without it in all four pairs and on every
  seed (descriptive, computed after the runs, not part of the rule).
- No accuracy effect of KL is measurable at 4 seeds. The point estimates move by 0.03-0.17 at 3
  digits, inside the seed spread. The 2-seed version of this section read "removing KL costs
  0.16 at 3 digits" and "restoring KL rescues DAPO"; neither survives seeds 2 and 3.
- Closest calls: DAPO + KL vs DAPO at 2 digits has no overlap but a gap of 0.065; Magistral + KL
  vs Magistral at 1 digit has a gap of 0.075 with one tie (0.94).
- CISPO + KL still loses to DAPO + KL, so CISPO's collapse is not the missing KL term.

```bash
minigpt grpo $B --mu 4 --beta 0 --out runs/kl_grpo_nokl/ckpt.pt
minigpt grpo $B --mu 4 --clip-eps-high 0.28 --loss-agg token-mean --beta 0.04 --fill-batch \
  --out runs/kl_dapo_kl/ckpt.pt
minigpt grpo $B --mu 4 --beta 0.04 --clip-eps-high 0.28 --loss-agg group-mean \
  --advantage batch --out runs/kl_magistral_kl/ckpt.pt
minigpt grpo $B --mu 4 --policy-loss cispo --loss-agg token-mean --clip-eps 1.0 \
  --clip-eps-high 0.28 --beta 0.04 --fill-batch --out runs/kl_cispo_kl/ckpt.pt
```

### Which part of CISPO and Magistral matters

All cells keep `--beta 0.04`, so KL is on; same base, 200 steps, same rule.

- `--clip-eps 1.0` sets the lower importance bound to 0, which can never fire. So CISPO + KL
  differs from DAPO + KL in two ways: no lower clip and the CISPO gradient rule. A PPO control
  (`--clip-eps 1.0 --clip-eps-high 0.28 --loss-agg token-mean --fill-batch`) splits them: it
  differs from CISPO + KL only in `--policy-loss` and from DAPO + KL only in `--clip-eps`. PPO and
  CISPO then differ only on tokens whose ratio exceeds 1.28.
- Magistral's three changes (clip-higher, `--loss-agg group-mean`, `--advantage batch`) were
  added one at a time to GRPO `mu 4` (4 seeds) and removed one at a time from Magistral + KL
  (seeds 0 and 1 only).

| Cell | Mean 1 / 2 / 3 digits | 3-digit, seeds 0 / 1 / 2 / 3 | Verdict (3-digit) |
|---|---|---|---|
| PPO control (no lower clip) | 0.92 / 0.75 / 0.47 | 0.54 / 0.48 / 0.40 / 0.46 | no measurable difference to DAPO + KL (gap -0.15) or to CISPO + KL (gap +0.12) |
| GRPO `mu 4` + clip-higher | 0.98 / 0.86 / 0.56 | 0.56 / 0.50 / 0.60 / 0.56 | no measurable difference |
| GRPO `mu 4` + group-mean | 0.98 / 0.84 / 0.57 | 0.56 / 0.52 / 0.62 / 0.58 | no measurable difference |
| GRPO `mu 4` + batch advantage | 0.92 / 0.81 / 0.56 | 0.40 / 0.58 / 0.64 / 0.62 | no measurable difference |
| Magistral + KL minus clip-higher / group-mean / batch advantage | — | seeds 0 / 1: 0.56 / 0.58; 0.66 / 0.44; 0.50 / 0.46 | no measurable difference |

- CISPO: the two steps from DAPO + KL (0.63) to the PPO control (0.47) to CISPO + KL (0.36) are
  each about 0.12-0.15 at 3 digits, and neither is measurable alone at 4 seeds; together they
  are. [INFERENCE] Both the missing lower clip and the CISPO gradient rule likely contribute.
- Magistral: none of its changes, alone or removed, has a measurable effect; nor does the full
  bundle. GRPO + batch advantage at 2 digits (0.78-0.84 vs 0.86-0.92) has no overlap but a gap of
  exactly 0.07, so it does not pass.
- A rerun of GRPO `mu 4` seed 0 reproduced its log line for line, so cells from different
  sessions are comparable and differences between cells come from flags and seeds.
- Not measured: how often the clip fires (not logged); seeds 2 and 3 of the leave-one-out cells.

`$B` is defined in the Reproduce block below. Logs: `runs/logs/kl_*.log`, `runs/logs/ri*_*.log`.

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

Seeds 1 to 3 add `--seed N` (the later flag wins); the shared baseline is the plain `minigpt grpo
--base ...` run with seeds 0, 1, 2. Logs: `runs/logs/rl_*.log`, `runs/logs/grpo_mid_s*.log`,
`runs/logs/ri4_*`, `runs/logs/ri5_*`, `runs/logs/ri6_*`.
