# One view of every training loss in minigpt

Every post-training method here optimises `E[policy term] - beta * regulariser`. They differ in three
places only:

1. **Where the samples come from**: a dataset, the current policy, a frozen teacher, or the policy
   as it was at the start of the step.
2. **What weights each sample**: a label (weight 1), a preference pair, a reward turned into an
   advantage, or a teacher's token probabilities.
3. **What keeps the policy near a reference**: a KL term (which form, against which model), a
   ratio clip, a low-rank update, or nothing.

The table lists each loss with those three answers. The formulas below it are what the code
computes, not what the papers print; where they differ, the line says so. Toy results are quoted
from the README and `docs/recipes/`, with their seed counts. Nothing here is re-derived or new.

## The table

Code paths are relative to `src/minigpt/`. "Ref" is a frozen copy of the starting model. "Toy
result" is the repo's own verdict, quoted; `n` seeds means the seeds that were run, and the verdict
rule (range overlap and a minimum gap) is in the linked document.

| ID | Method (command / flag) | Samples from | Learning signal | Regulariser / anchor | Models in memory | Code | Toy result |
|---|---|---|---|---|---|---|---|
| S1 | Next-token LM (`pretrain`, `reason-pretrain`) | dataset | the next token (label) | none | 1 | `model.py:GPTModel.forward`, `train.py:train` | test loss 3.872-3.879, seeds 0-2 ([README](../README.md#pretraining-overfitting-and-a-selection-biased-headline)) |
| S2 | Auxiliary: multi-token prediction (`--mtp-depth`), MoE balance (`--n-experts`) | dataset | extra next-token heads; expert-load penalty | weight `mtp_weight`; `moe_balance_alpha` | 1 | `model.py:GPTModel._aux_losses` | MTP: no measurable difference on equations, 4 seeds ([deepseek-v4](recipes/deepseek-v4.md#verdicts)) |
| S3 | SFT (`sft`, `reason-sft`, `cai-train --stage sl`) | dataset (prompt tokens masked) | the answer tokens (label) | none; small lr | 1 | `sft.py:train_sft` | `reason-sft` seed 0: 0.98 / 0.94 / 0.86 on 1 / 2 / 3 digits ([README](../README.md#results-one-run-on-an-m-series-mac-mps-seed-0)) |
| S4 | LoRA (`--lora-rank` on `reason-sft`, `grpo`) | as the host loss | as the host loss | update confined to rank r, B = 0 at start | as the host, plus adapters | `lora.py:apply_lora`, `lora.py:merge_lora` | SFT: worse than full, 4 seeds (frozen new-token embeddings, `[INFERENCE]`); GRPO: no measurable difference, 4 seeds ([lora](recipes/lora.md)) |
| S5 | Classifier head (`finetune-cls`) | dataset | class label | trains last block, final LN, head only | 1 | `finetune_cls.py:GPTForClassification` | test accuracy 0.940, 1 run ([README](../README.md#results)) |
| S6 | Task arithmetic (`merge`) | none (weights only) | expert minus base | scale `L` | base + experts | `merge.py:task_arithmetic` | worse than OPD (0.020-0.025 vs 0.166-0.206), experts seed 0 ([merging](recipes/merging.md#verdicts)) |
| P1 | DPO (`dpo`, `cai-train --stage rl`; `--nll-alpha`) | dataset or judge pairs (fixed, offline) | chosen beats rejected | implicit: `beta` times the log-ratio to ref; optional NLL on chosen | 2 (policy, ref) | `dpo.py:dpo_loss`, `dpo.py:train_dpo` | no detectable difference from SFT, beta 0.05 / 0.1 / 0.5, beta 0.1 at 3 seeds ([README](../README.md#results)) |
| P2 | Bradley-Terry reward model (`reward-model`) | current SFT policy, k per prompt, one strict pair | chosen scores above rejected | none | 1 (SFT body + scalar head) | `rlhf.py:bradley_terry_loss`, `rlhf.py:train_reward_model` | held-out pair accuracy 0.500 (232 pairs) and 0.827 (1,908 pairs), 1 run each ([README](../README.md#rlhf-reward-model--ppo)) |
| P3 | PPO + GAE + learned RM (`ppo`) | current policy (old policy per step) | RM score at the last token, GAE advantage, value head | per-token `-beta * (logp - ref_logp)` inside the reward; ratio clip | 3 (policy + value head, ref, RM) | `rlhf.py:train_ppo`, `rlhf.py:shape_rewards`, `rlhf.py:gae`, `rlhf.py:ppo_policy_loss` | gold accuracy fell at beta 0.02 / 0.1 / 0.3, 1 seed ([README](../README.md#results-one-run-on-an-m-series-mac-mps-seed-0)) |
| P4 | DPO on own samples (`reason-dpo`) | current model, k per task; verifier picks best and worst | correct beats incorrect | as P1 | 2 | `rft.py:preference_pairs`, `dpo.py:train_dpo` | default hurts (3-digit 0.66 to 0.32), seeds 0 / 1 ([offline-recipes](recipes/offline-recipes.md#verdicts)) |
| P5 | Constitutional AI data (`cai-data`, `cai-eval`) | current model; an LLM judge critiques, revises, or picks | revision as label (SL); judge's pick as pair (RL) | as S3 / P1 | 1 + external judge (Ollama, not ours) | `cai.py:critique_and_revise`, `cai.py:label_pairs` | adherence sft 1.00, SL 1.25, RL 1.45; judge picked slot A 86.1%; 1 run, 20 prompts ([README](../README.md#constitutional-ai-caipy)) |
| G1 | GRPO (`grpo`) | current policy, G per task, `old` taken before the step | `(r - mean) / (std + 1e-4)` per group | k3 KL to ref; clip `[1-eps, 1+eps]` | 2 (policy, ref) | `grpo.py:grpo_loss`, `grpo.py:group_advantages`, `grpo.py:kl_k3` | 3-digit 0.74-0.76, seeds 0-2 ([rl-losses](recipes/rl-losses.md#results)) |
| G2 | Dr. GRPO (`--no-std-norm`, `--loss-agg token-sum`) | as G1 | `r - mean`, no std; sum over tokens / constant `max_len` | as G1 | 2 | `grpo.py:group_advantages`, `grpo.py:grpo_loss` | `--no-std-norm` with G3: no measurable effect, seed 0; `token-sum` alone: no result reported ([README](../README.md#results-one-run-on-an-m-series-mac-mps-seed-0)) |
| G3 | Unbiased KL (`--unbiased-kl`) | as G1 | as G1 | k3 times the ratio `pi / pi_old` | 2 | `grpo.py:grpo_loss` | no measurable effect, seed 0 (with G2) |
| G4 | GSPO (`--policy-loss gspo`) | as G1 | as G1 | one clipped ratio per sequence, k3 KL kept | 2 | `grpo.py:grpo_loss` | no measurable difference, 4 seeds ([rl-losses](recipes/rl-losses.md#results)) |
| G5 | CISPO (`--policy-loss cispo`) | as G1 | as G1 | clipped weight is a stop-gradient scale; no drop; no lower clip with `--clip-eps 1.0` | 2 | `grpo.py:grpo_loss` | loses at 2 and 3 digits, 4 seeds; weight cap 2 / 4 / 8 does not fix it ([rl-losses](recipes/rl-losses.md#cispos-eps_high)) |
| G6 | DAPO pieces (`--clip-eps-high`, `--loss-agg token-mean`, `--fill-batch`, `--overlong-cache`) | as G1; resample until groups have mixed rewards | as G1; overlong penalty added to reward | asymmetric clip; `--beta 0` in the recipe | 2 | `grpo.py:grpo_loss`, `grpo.py:train_grpo`, `grpo.py:overlong_penalty` | recipe: no measurable difference, 4 seeds; overlong L = 20 none, L = 40 unstable, 2 seeds ([rl-losses](recipes/rl-losses.md#results), [length-budget](recipes/length-budget.md#verdicts)) |
| G7 | Magistral (`--loss-agg group-mean`, `--advantage batch`) | as G1 | group-centred, divided by the batch std | `--beta 0`, clip-higher | 2 | `grpo.py:grpo_loss`, `grpo.py:group_advantages` | no measurable difference, 4 seeds; each change alone also none ([rl-losses](recipes/rl-losses.md#which-part-of-cispo-and-magistral-matters)) |
| G8 | RLOO (`--advantage rloo`) | as G1 | `r_i - mean(r_-i)`, no std | k3 KL kept | 2 | `grpo.py:group_advantages` | no measurable difference, 4 seeds ([rl-losses](recipes/rl-losses.md#results)) |
| G9 | ProRL (`--ref-reset-every N`) | as G1 | as G1 | ref becomes the current policy every N steps; optimizer reset | 3 (policy, caller's ref, reset copy) | `grpo.py:train_grpo` | no measurable difference, 4 seeds ([rl-losses](recipes/rl-losses.md#results)) |
| G10 | Effort length penalty (`--efforts`, `--effort-k0`) | as G1, one group per (task, effort) | reward minus a think-length penalty that decays with the effort level | as G1 | 2 | `grpo.py:effort_coef`, `grpo.py:length_penalty` | inconclusive, seed 0 ([README](../README.md#controllable-reasoning-effort)) |
| G11 | Length shaping (`--length-reward` Kimi K1.5, `--budget-tau` Kimi K3) | as G1 | reward plus in-group length reward; or -1 above a think budget | as G1 | 2 | `grpo.py:kimi_length_rewards`, `grpo.py:budget_override`, `grpo.py:make_shaper` | K1.5 none; K3 tau 1.25 none, annealed tau worse; 2 seeds ([length-budget](recipes/length-budget.md#verdicts)) |
| G12 | PRM-shaped reward (`grpo --prm`) | as G1 | reward plus `w * P(every step right)` from a frozen PRM | as G1 | 3 (policy, ref, PRM) | `prm.py:prm_shaped_reward` | not measurable vs outcome reward, 4 seeds ([prm](recipes/prm.md#extra-grpo-with-a-prm-shaped-reward-not-part-of-the-two-claims)) |
| G13 | Lenient grader (`hack-demo`) | as G1 | substring match on the answer | as G1 | 2 | `monitor.py:lenient_reward` | strong SFT: hack rate 0.003 to 0.002; weak SFT: 0.077 to 0.197, 1 run each ([README](../README.md#cot-monitor-and-reward-hacking)) |
| D1 | OPD, reverse KL, full vocabulary (`opd`) | current student | teacher's whole next-token distribution at each prefix | the KL is the objective; teacher frozen | 2 (student, teacher) | `opd.py:token_kl`, `opd.py:opd_loss` | round 1, 4 seeds: all arms (OPD, GRPO, OPD then GRPO) at M 0.24-0.28, ranges overlap; with Muon, OPD beats GRPO alone (gap +0.086, not predeclared) ([opd-then-rl](recipes/opd-then-rl.md#verdicts)) |
| D2 | OPD, forward KL (`--kl forward`) | current student | as D1, other direction | as D1 | 2 | `opd.py:token_kl` | no forward-vs-reverse result is reported ([README](../README.md#on-policy-distillation-opd)) |
| D3 | OPD as RL (`--estimator token`) | current student | teacher minus student log-prob of the sampled token, clipped, as reward | as D1 | 2 | `opd.py:token_pg_loss` | worse than full; diverges at lr 3e-4 on both seeds ([consolidation](recipes/consolidation.md#e1-full-vs-token-estimator-500-steps-multi-teacher-opd-from-student_sft)) |
| D4 | Off-policy distillation (`opd --off-policy`) | frozen teacher | the teacher's sampled tokens (label) | none | 2 | `opd.py:train_opd` | not distinguishable from on-policy on overall accuracy, 1 seed ([README](../README.md#multi-teacher-opd-addition--subtraction)) |
| D5 | Multi-teacher OPD (`--teacher OP=PATH`) | current student | each prompt's own teacher | token-weighted mean over teachers | 1 + one per distinct teacher | `opd.py:routed_opd_loss` | no measurable lift over the base, 4 seeds; tuned OPD ties RFT ([merging](recipes/merging.md#verdicts)) |
| O1 | Rejection-sampling SFT (`reason-rft --select verifier`) | current model (or `--sampler` teachers), k per task | verifier keeps correct, well-formed, distinct samples (label) | none | 1 (+ sampler) | `rft.py:select_completions`, `rft.py:rft_dataset` | 3 rounds beat GRPO at 3 digits (0.88 / 0.88 vs 0.74-0.76), seeds 0 / 1; ties tuned OPD, 4 seeds ([offline-recipes](recipes/offline-recipes.md#verdicts), [merging](recipes/merging.md#verdicts)) |
| O2 | Same, selected by a learned RM (`--select rm`) | as O1 | the RM's top-scored samples, right or not | none | 2 (model, RM) | `rft.py:rm_scores`, `rft.py:select_completions` | RM picks worse than random; 3-digit 0.24 / 0.30 vs 0.50 / 0.66 for the verifier, seeds 0 / 1 ([offline-recipes](recipes/offline-recipes.md#verdicts)) |
| O3 | SFT on a teacher's samples (`reason-rft --sampler`) | a stronger, RL-trained model | verified teacher samples (label) | none | 2 (model, sampler) | `rft.py:sample_by_op`, `rft.py:rft_dataset` | beats GRPO at 3 digits (0.88 / 0.98), seeds 0 / 1; the gain is the distilled teacher ([offline-recipes](recipes/offline-recipes.md#verdicts)) |
| V1 | PRM and ORM (`prm`, `best-of-n`) | cached policy samples | per-step label (PRM), outcome label (ORM), as BCE | none | 1 verifier (policy body + scalar head) | `prm.py:batch_loss`, `prm.py:best_of_n` | PRM beats ORM at N = 16, not majority vote, 4 seeds ([prm](recipes/prm.md#results-best-of-n-verifier-seeds-0-3-200-tasks-per-cell)) |
| V2 | Guards (`guard`): linear probe, classifier, cascade | templated prompts; the policy's activations or answers | forbidden-or-allowed label (BCE or CE) | weight decay (probe); none (classifier) | policy (frozen) + classifier | `guard.py:fit_probe`, `guard.py:fit_classifier`, `guard.py:cascade` | probe and cascade claims hold, 4 seeds ([constitutional-classifiers](recipes/constitutional-classifiers.md#verdicts)) |

## The formulas

Notation. Prompt $x$, completion $y = (y_1, \dots, y_T)$, completion mask $m_t \in \lbrace 0, 1 \rbrace$ (prompt
and padding are 0). $\ell_t = \log \pi_\theta(y_t \mid x, y_{<t})$, and $\ell_t^{\text{ref}}$,
$\ell_t^{\text{old}}$ are the same under the reference and under the policy at the start of the
step. $\rho_t = \exp(\ell_t - \ell_t^{\text{old}})$. $\sigma$ is the sigmoid.

### Supervised: S1, S3, S5

$$\mathcal{L} = -\frac{\sum_t m_t \, \ell_t}{\sum_t m_t}$$

Cross-entropy with `ignore_index=-100`, so the mean runs over every unmasked token of the batch
(`model.py:GPTModel.forward`). S1 masks nothing; S3 masks the prompt; S5 is a two-class
cross-entropy on one logit vector per sequence.

### Auxiliary: S2

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{mtp}} \cdot \frac{1}{D}\sum_{d=1}^{D} \mathrm{CE}_d + \sum_{\text{layers}} \mathcal{L}_{\text{balance}}$$

$\mathrm{CE}_d$ is the cross-entropy of MTP module $d$ on the token $d + 1$ positions ahead
(`_aux_losses`). The balance term is a quoted line, not re-derived:
`self.balance_loss = self.alpha * (frac * prob).sum(dim=-1).mean()`. Both exist only while
`self.training`.

### LoRA and merging: S4, S6

$$W' = W + \frac{\alpha}{r} B A, \qquad B = 0 \text{ at init}, \qquad \theta = \theta_0 + L \sum_i (\theta_i - \theta_0), \quad L = 1/n \text{ by default}$$

The loss is whatever the host command uses; LoRA changes which parameters train (`lora.py`).
`merge_lora` folds $\frac{\alpha}{r} B A$ into $W$.

### DPO: P1, P4

$$\mathcal{L} = -\log \sigma\Big(\beta \big[(\log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w)) - (\log \pi_\theta(y_l) - \log \pi_{\text{ref}}(y_l))\big]\Big) + \alpha \Big(-\frac{\log \pi_\theta(y_w)}{n_w}\Big)$$

$\log \pi(y)$ is the sum of token log-probabilities over the completion, not the mean
(`compute_sequence_log_probs`). The last term is the length-normalised NLL of the chosen answer,
$n_w$ its token count; $\alpha = 0$ by default for `dpo` and 0.2 for `reason-dpo`. The policy runs
with dropout off so the loss starts at $\log 2$.

### Reward model: P2

$$\mathcal{L} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)$$

$r_\phi$ is the scalar head at the last non-pad token. This is P1 with the implicit reward
$\beta \log(\pi_\theta / \pi_{\text{ref}})$ replaced by a separate network's output.

### PPO with GAE: P3

Per-token reward, with the reward-model score $s_\phi$ added on the last completion token $T$:

$$r_t = -\beta\,(\ell_t - \ell_t^{\text{ref}})\,m_t + s_\phi \cdot \mathbb{1}[t = T]$$

$\hat A_t$ is GAE over $r_t$ and the value head's $V_t$ ($\gamma = 1$, $\lambda = 0.95$), then
whitened over the masked tokens: $\hat A \leftarrow (\hat A - \mu) / \sqrt{\mathrm{var} + 10^{-8}}$.
The return $R_t = \hat A_t + V_t$ is taken before whitening.

$$\mathcal{L} = -\frac{\sum_t m_t \min\big(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1 \pm \epsilon)\hat A_t\big)}{\sum_t m_t} + c_v \cdot \tfrac12\,\overline{\max\big((V_t - R_t)^2,\ (V^{\text{old}}_t + \mathrm{clip}(V_t - V^{\text{old}}_t, \pm\epsilon_v) - R_t)^2\big)}$$

Here $\beta = 0.02$, $\epsilon = \epsilon_v = 0.2$, $c_v = 0.1$, two epochs of minibatches of 16. The
KL is a reward (the single-sample estimator $\ell_t - \ell_t^{\text{ref}}$), not a loss term. The
value head shares the policy's body.

### GRPO: G1 to G9

Advantage, per group of $G$ rewards ($\mathrm{std}$ is the population std):

$$A_i = \frac{r_i - \bar r}{\mathrm{std}(r) + 10^{-4}}$$

Loss, with $k_3 = e^{d} - d - 1$ and $d = \ell_t^{\text{ref}} - \ell_t$ (non-negative, 0 when the
policy equals ref):

$$\mathcal{L} = -\mathrm{agg}_{i,t}\Big[\, m_t \big( S_t - \beta \, k_3 \big) \Big], \qquad S_t = \min\big(\rho_t A_i,\ \mathrm{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon_{h})\, A_i\big)$$

The surrogate $S_t$ and the aggregation `agg` are the flags:

| Flag value | Surrogate $S_t$ or aggregation |
|---|---|
| `--policy-loss ppo` | $S_t$ as above |
| `--policy-loss gspo` | one ratio per sequence, $s_i = \exp\big(\frac{1}{\sum_t m_t}\sum_t m_t (\ell_t - \ell_t^{\text{old}})\big)$, in $\min(s_i A_i,\ \mathrm{clip}(s_i, 1-\epsilon, 1+\epsilon_h) A_i)$ |
| `--policy-loss cispo` | $S_t = \mathrm{sg}\big(\mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon_h)\big)\, A_i\, \ell_t$ |
| `--loss-agg seq-mean` | $\frac{1}{N}\sum_i \frac{1}{\lvert y_i \rvert}\sum_t(\cdot)$, $N$ = rows |
| `--loss-agg token-sum` | $\frac{1}{N}\sum_i \frac{1}{L_{\max}}\sum_t(\cdot)$, $L_{\max}$ = `max_new_tokens` |
| `--loss-agg token-mean` | $\sum_{i,t}(\cdot) \big/ \sum_{i,t} m_t$ over the batch |
| `--loss-agg group-mean` | per group of $G$ rows: $\sum_{i,t}(\cdot)$ over the group's total length, then the mean over groups |
| `--advantage batch` | $A_i = (r_i - \bar r) / (\mathrm{std}_{\text{batch}}(r - \bar r) + 10^{-4})$, std over all centred values of the batch |
| `--advantage rloo` | $A_i = \frac{G}{G-1}(r_i - \bar r) = r_i - \mathrm{mean}(r_{-i})$ |
| `--no-std-norm` | $A_i = r_i - \bar r$ |
| `--unbiased-kl` | $k_3 \leftarrow \rho_t\, k_3$, with $\rho_t$ **not** detached (the code multiplies by `ratio`, which carries gradient) |
| `--clip-eps-high` | sets $\epsilon_h$; default $\epsilon_h = \epsilon$ |
| `--ref-reset-every N` | every $N$ steps `ref.load_state_dict(policy.state_dict())` and a fresh optimizer |

Groups whose rewards are all equal are dropped before the loss (zero advantage); `--fill-batch`
resamples until `tasks_per_step` groups survive. `--mu` inner steps reuse the same `old` log-probs,
so the clip only fires from the second inner step on.

### Reward sources for GRPO: G10 to G13

Base reward $r = \mathbb{1}[\text{answer correct}] + 0.1\,\mathbb{1}[\text{format exact}]$
(`grpo.py:default_reward`). Each shaper adds to it:

$$\text{G10: } -\min\!\Big(1,\ \frac{k(b)\, n_{\text{think}}}{30}\Big),\quad k(b) = k_0\, e^{-(b - b_{\min})/(0.25\,\overline{\Delta b})}$$

$$\text{G11, K1.5: } w\,\lambda_i,\ \ \lambda_i = \tfrac12 - \tfrac{\lvert y_i \rvert - \min}{\max - \min}\ (\text{correct}),\ \ \min(0, \lambda_i)\ (\text{wrong}),\ \ 0 \text{ if all lengths tie}$$

$$\text{G11, K3: } r_i \leftarrow -1 \text{ if } n_{\text{think},i} > \tau(\text{step})\, b_0 \text{, correct or not}; \qquad \text{G6 overlong: } \max\!\Big(-1,\ \frac{L_{\max} - L_{\text{cache}} - \lvert y_i \rvert}{L_{\text{cache}}}\Big) \text{ above } L_{\max} - L_{\text{cache}}$$

$$\text{G12: } w \prod_{\text{steps}} \sigma(z_s) \quad (z_s = \text{the PRM's logit at step } s;\ 0 \text{ if malformed}), \qquad \text{G13: } \mathbb{1}[\text{answer string} \subset \text{completion}]$$

### On-policy distillation: D1 to D5

Position-wise, with $p_s, p_t$ the student and teacher next-token distributions at the same
prefix (the student's own sample), and the mean over completion positions:

$$\text{D1: } \sum_v p_s(v)\,[\log p_s(v) - \log p_t(v)], \qquad \text{D2: } \sum_v p_t(v)\,[\log p_t(v) - \log p_s(v)]$$

The gradient goes through $p_s$ at each prefix, not through the sampling of the prefix. D3 replaces
the sum over $v$ by the one sampled token $y_t$:

$$\mathcal{L}_{\text{D3}} = -\frac{1}{\sum m}\sum_t m_t \,\mathrm{clip}\big(\mathrm{sg}(\log p_t(y_t) - \ell_t),\ \pm r_{\max}\big)\,\ell_t, \qquad r_{\max} = 5$$

D4 is plain cross-entropy (S1 on completions) on the teacher's samples. D5 splits the batch by
operator, computes D1 against that operator's teacher, and weights each part by its share of
completion tokens, so the result equals the per-token mean over the whole batch.

### Offline selection and verifiers: O1 to O3, V1, V2

O1 to O3 are S3 on a filtered set: `select_completions` keeps up to `keep` distinct completions
per task, in verifier order (O1: correct and well-formed) or by descending RM score (O2). P4 builds
`(best, worst)` pairs with `strict_pair`, by `labeler_score` (accuracy + format, 0 to 2), then
applies P1.

$$\text{V1: } -\tfrac{1}{n}\sum \big[\, y \log \sigma(z) + (1 - y)\log(1 - \sigma(z)) \,\big]$$

$z$ is the head's logit at the last token of each step (PRM, label: that step is right; steps after
the first error are unlabeled) or at the last token of the sequence (ORM, label: the answer is
right). V2's probe is the same BCE on a standardised linear layer over pooled hidden states
(Adam, weight decay $10^{-3}$); its classifier is cross-entropy over two classes. The cascade has
no loss: the probe decides outside a probability band, the output classifier inside it.

## Where the code differs from the paper's formula

- **GRPO std.** The std is the population std with `+ 1e-4`, not an unspecified constant (`group_advantages`).
- **`--unbiased-kl`.** The ratio multiplying $k_3$ keeps its gradient. The line is
  `kl = ratio * kl`; whether the DeepSeek-V3.2 paper stops that gradient is not checked here.
- **CISPO.** The paper's loss has token-level normalisation, no lower bound and no KL. Here the
  normaliser is whatever `--loss-agg` says, the lower bound is $1 - \epsilon$ unless you pass
  `--clip-eps 1.0`, and $\beta k_3$ is subtracted unless you pass `--beta 0`.
- **GSPO.** The paper omits the KL; here `--beta` still applies per token. The sequence ratio's
  clip range is `--clip-eps` / `--clip-eps-high`, so the paper's values must be passed.
- **PPO (P3).** The value head shares the policy's body with `vf_coef` 0.1 instead of a separate
  value model, and there is no pretraining-mix term.
- **DPO (P1, P4).** The log-probability is a sum, as in the paper. The NLL term is an addition
  (Llama 3), off by default for `dpo`.
- **OPD token estimator (D3).** There is one update per batch, so no PPO ratio and no clip on the
  ratio; the log-ratio is clipped to $\pm r_{\max}$. Each token is weighted by its own reward only,
  not a sum of later rewards.
- **Dr. GRPO.** `token-sum` divides by `max_new_tokens`, a constant, which is how the code reads
  the paper's constant normaliser; the paper's exact constant is not checked.

## How they relate

- **GRPO is PPO (P3) with the critic replaced by a group baseline and the KL moved from the reward
  into the loss.** P3's $\hat A$ comes from GAE over a value head; G1's $A_i$ from the group's
  own rewards. P3 puts $-\beta(\ell - \ell^{\text{ref}})$ in $r_t$; G1 subtracts $\beta k_3$ from the
  surrogate.
- **G4, G5 and G6 to G9 change one slot of G1 each.** The clip rule (G4, G5, G6), the
  normaliser (G2, G6, G7), the baseline (G7, G8) or the anchor (G9, G3). Every flag defaults to G1.
- **CISPO keeps the gradient of clipped tokens.** In G1 a token with $\rho_t$ outside the clip
  range and the matching sign of $A_i$ gets zero gradient (the `min` picks the constant); in G5 it
  gets the capped weight times $A_i \nabla \ell_t$.
- **D3 is a policy gradient with the teacher's log-ratio as a dense per-token reward, and D1 is
  its exact-expectation form.** Per position, $\mathbb{E}_{y \sim p_s}[(\log p_s - \log p_t)\nabla \log p_s]
  = \nabla \sum_v p_s (\log p_s - \log p_t)$, because $\sum_v \nabla p_s = 0$; the code's D3
  docstring states the same in expectation.
- **DPO and the reward model are one loss on different rewards, and O1 is REINFORCE with an
  indicator reward.** P1 and P2 are both $-\log\sigma(r_w - r_l)$, with $r = \beta\log(\pi_\theta/\pi_{\text{ref}})$
  or the head's output. SFT on the verifier-kept samples has gradient $\sum \mathbb{1}[\text{correct}]\nabla\ell$:
  policy gradient with reward 1 for correct, no baseline, samples from a fixed old policy.
