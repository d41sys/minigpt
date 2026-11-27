# LoRA against full fine-tuning on the equation task

Enterprise adaptation in 2026 is mostly LoRA: a frozen base plus a low-rank update per linear
layer, merged into the weights for serving. This adds it to `reason-sft` and `grpo` and measures
it against full fine-tuning on the toy equation pipeline. All runs on the Korea server (RTX 4090,
shared with other jobs), never on a laptop. Toy scale: 0.84M parameters, one task (addition).

## What was built

`src/minigpt/lora.py`; flags on `reason-sft` and `grpo` (`hack-demo` shares the `grpo` arguments).

| Flag | Meaning |
|---|---|
| `--lora-rank R` | 0 (default) = full fine-tuning. R > 0: freeze the whole model, wrap the target linears with `W x + (alpha / R) B A x`; A is `R x in` (uniform ±1/√in), B is `out x R` and starts at zero, so training starts from the unchanged model |
| `--lora-alpha` | default 16, so the scale is `16 / R` (R = 8 gives 2, R = 1 gives 16). Fixed alpha, as in "LoRA without regret", keeps the best learning rate roughly independent of the rank |
| `--lora-targets all\|attn` | `all`: attention `c_attn`, `c_proj` + MLP `c_fc`, `c_proj` (default); `attn`: attention projections only |
| `--lora-embed` | also train the token embeddings and the tied LM head (PEFT's `modules_to_save`; needed here, see below) |

The embeddings, layer norms, biases and (without `--lora-embed`) the tied LM head stay frozen.
The command prints `lora rank=… trainable=N of M`. On save the update is folded into the base
weights (`W += (alpha / R) B A`) and the wrappers are removed, so the checkpoint has the plain GPT
layout and every downstream command works unchanged. `--lora-rank` with `--ref-reset-every` is
rejected (the reference snapshot would need the LoRA keys). Dense models only: it wraps the
`attn` / `mlp` linears named above, so MoE experts and CSA's extra linears are not wrapped.
`tests/test_minigpt_lora.py`: B = 0 gives the base output; merged forward equals LoRA forward; only
A and B (and the embedding with `--lora-embed`) receive gradients; the merged checkpoint loads as
a plain model.

## Experiment

Verdict rule, written before the runs: 4 seeds per arm (0-3); a LoRA arm differs measurably from
full fine-tuning only when the two seed ranges (min-max) do not overlap and the gap between them
exceeds 0.05 accuracy (3 digits, `reason-eval --n 200`, final-step model). Otherwise "no measurable
difference". Per-seed numbers are description only.

```
minigpt reason-pretrain --name rp_sS --seed S --optimizer muon       # 1500 steps, 0.84M dense
minigpt reason-sft --base runs/rp_sS/ckpt.pt --n 10000 --epochs 3 --seed S \
    [--lora-rank 8 --lora-targets all|attn] --lr LR --out runs/sft_ARM_sS/ckpt.pt
minigpt grpo --base runs/sft_full_sS/ckpt.pt --seed S [--lora-rank R] --lr LR --out runs/g_ARM_sS/ckpt.pt
minigpt reason-eval --checkpoint CKPT --n 200
```

The base is Muon-pretrained (Muon makes the SFT seeds stable; `deepseek-v4.md` section 2). SFT
uses the default AdamW so there is headroom; Muon in SFT would sit at its 0.97 ceiling. GRPO runs
the default 200 steps from each seed's full-FT SFT checkpoint (default lr 5e-5, and 10x = 5e-4 for
the LoRA arms; all arms were fixed before the runs, not chosen after looking).

### 1. SFT: full against LoRA (equation traces, 3-digit greedy accuracy)

Params = trainable parameters (full fine-tuning: 842,752). Seed order s0 / s1 / s2 / s3.

| Arm | Trainable | 3 digits, s0 / s1 / s2 / s3 | Range | 1 / 2 / 3 digits, mean | Final SFT loss, s0-s3 | Wall s | Verdict |
|---|---|---|---|---|---|---|---|
| full FT, lr 5e-4 | 842,752 | 0.805 / 0.635 / 0.65 / 0.895 | 0.635-0.895 | 0.98 / 0.87 / 0.75 | 0.010 / 0.014 / 0.014 / 0.011 | 44 / 47 / 52 / 55 | baseline |
| LoRA all, r = 8, lr 5e-4 | 65,536 | 0.02 / 0.01 / 0.01 / 0.015 | 0.01-0.02 | 0.51 / 0.09 / 0.01 | 2.09 / 2.25 / 2.10 / 2.13 | 63 / 56 / 63 / 52 | worse (gap 0.62) |
| LoRA all, r = 8, lr 5e-3 (10x) | 65,536 | 0 / 0.01 / 0.105 / 0.005 | 0-0.105 | 0.33 / 0.09 / 0.03 | 2.29 / 2.26 / 2.04 / 2.19 | 53 / 62 / 53 / 68 | worse (gap 0.53) |
| LoRA attn, r = 8, lr 5e-3 | 24,576 | 0 / 0 / 0.005 / 0.005 | 0-0.005 | 0.28 / 0.03 / 0.00 | 2.21 / 2.41 / 2.21 / 2.27 | 47 / 50 / 51 / 56 | worse (gap 0.63) |

All three assigned LoRA arms fail: the SFT loss stalls near 2.1-2.4 against 0.01 for full
fine-tuning. The 10x learning rate does not help (0.01-0.02 vs 0-0.105 overlap: no measurable
difference between the two rates, and neither works); attention-only is no better.

Why: the SFT traces contain letters and tags (`<think>`, `<answer>`) that the equation-only base
never saw. The tokenizer is byte-level, so those bytes' embeddings (and the tied LM-head rows) are
untouched random initialisation, and with frozen embeddings LoRA can only steer hidden states
towards random output rows. Diagnostic arm, `--lora-embed` (98,560 trainable: LoRA all r = 8 plus
the 256 x 128 embedding), same seeds:

| Arm | Trainable | 3 digits, s0 / s1 / s2 / s3 | Range | 1 / 2 / 3 digits, mean | Final SFT loss | Wall s | Verdict |
|---|---|---|---|---|---|---|---|
| LoRA all r = 8 + embed, lr 5e-4 | 98,560 | 0.65 / 0.51 / 0.25 / 0.855 | 0.25-0.855 | 0.98 / 0.89 / 0.57 | 0.029 / 0.052 / 0.084 / 0.015 | 68 / 66 / 74 / 67 | none (overlaps full) |
| LoRA all r = 8 + embed, lr 5e-3 | 98,560 | 0.02 / 0.205 / 0.135 / 0.015 | 0.015-0.205 | 0.65 / 0.37 / 0.09 | 0.100 / 0.074 / 0.075 / 0.115 | 69 / 71 / 68 / 45 | worse (gap 0.43) |

[INFERENCE] The frozen new-token embeddings, not the rank, are what breaks LoRA here: the loss
drops from 2.1 to 0.02-0.08 once they train, and 1-digit accuracy reaches full fine-tuning's.
Rank, layer norms and the 3-digit gap that remains (mean 0.57 vs 0.75, worst seed 0.25 vs 0.635)
were not separated; the seed ranges overlap, so this is "no measurable difference", not a tie.
The 10x rate hurts once embeddings train (AdamW at 5e-3 on the embedding is too large). A real
base model has seen its whole vocabulary; this is a property of the toy base.

### 2. GRPO: full against LoRA (from each seed's full-FT SFT checkpoint)

3-digit greedy accuracy after 200 steps, `reason-eval --n 200`. The SFT start is the full-FT row
above (0.805 / 0.635 / 0.65 / 0.895). LoRA rank 1 and rank 8 on all linears, embeddings frozen
(the SFT model already knows the format). Gain = after minus SFT start, per seed.

| Arm | Trainable | 3 digits after, s0 / s1 / s2 / s3 | Range | Gain, s0 / s1 / s2 / s3 | 1 / 2 / 3 digits, mean | Wall s | Verdict |
|---|---|---|---|---|---|---|---|
| full FT, lr 5e-5 | 842,752 | 0.98 / 0.975 / 0.98 / 0.95 | 0.95-0.98 | +0.18 / +0.34 / +0.33 / +0.06 | 1.00 / 1.00 / 0.97 | 251 / 469 / 249 / 243 | baseline |
| LoRA r = 1, lr 5e-5 | 8,192 | 0.945 / 0.985 / 0.945 / 0.93 | 0.93-0.985 | +0.14 / +0.35 / +0.30 / +0.04 | 1.00 / 1.00 / 0.95 | 466 / 615 / 459 / 444 | none |
| LoRA r = 1, lr 5e-4 (10x) | 8,192 | 0.975 / 0.985 / 0.94 / 0.94 | 0.94-0.985 | +0.17 / +0.35 / +0.29 / +0.05 | 1.00 / 1.00 / 0.96 | 472 / 624 / 454 / 451 | none |
| LoRA r = 8, lr 5e-5 | 65,536 | 0.965 / 0.97 / 0.965 / 0.935 | 0.935-0.97 | +0.16 / +0.34 / +0.32 / +0.04 | 1.00 / 1.00 / 0.96 | 457 / 588 / 456 / 453 | none |
| LoRA r = 8, lr 5e-4 (10x) | 65,536 | 1 / 0.99 / 0.915 / 0.955 | 0.915-1 | +0.20 / +0.36 / +0.27 / +0.06 | 1.00 / 1.00 / 0.96 | 473 / 463 / 450 / 450 | none |

- No LoRA arm differs measurably from full fine-tuning: every range overlaps the full range
  (0.95-0.98). Rank 1 (8,192 parameters, 1% of the model) gains as much as full fine-tuning.
- The default learning rate already moves: with alpha fixed at 16 the update scale is `16 / R`
  (rank 1: 16), so the 10x rule was not needed here, and it changes nothing measurable either.
- This does not show LoRA is better: all arms sit at 0.92-1.0 on 3 digits, near the ceiling of
  200 tasks and of the 1-3 digit RL range (4-5 digits stay 0 for every arm). Differences under
  0.05 cannot be resolved.

## Limits

- One task, a 0.84M model, 200 GRPO steps, 4 seeds; seeds change initialisation, data order and
  rollouts but not the task. SFT start accuracy varies 0.635-0.895 across seeds and the 3-digit
  metric is saturated after GRPO.
- The SFT failure is specific to a base that never saw the output vocabulary (equations-only
  pretraining), and `--lora-embed` is a diagnostic that was not part of the assigned arms. One
  rank (8) and two rates were run for SFT; no rank sweep, no alpha sweep, no Muon on A/B.
- Wall times come from a GPU shared with other users' jobs (full GRPO seed 1 took 469 s against
  243-251 s for the other seeds); the LoRA layers are unfused (an extra matmul per wrapped linear,
  also during generation), so LoRA arms ran slower here (SFT about 1.2-1.4x, GRPO about 1.8x on
  the unloaded seeds). Memory was not measured. Compare ratios, not seconds.
- Logs: `runs/logs/lora_*` (gitignored).
