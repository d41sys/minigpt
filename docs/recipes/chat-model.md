# A chat model, and a block ablation to choose its architecture

minigpt's other results are arithmetic or TinyStories at 0.8-5M parameters. This recipe
(1) adds the modern block (RMSNorm, SwiGLU, grouped-query attention; RoPE already existed) as
flags, (2) tests it against the GPT-2 block at equal parameters, (3) trains one English chat
model and (4) scores it on four standard multiple-choice benchmarks, with standard errors.

New flags (defaults unchanged, so every old checkpoint loads): `pretrain --norm rmsnorm`,
`--mlp-type swiglu`, `--num-kv-heads K`, `--d-ff N`; `generate --chat`; global `--tokenizer PATH`
(before the command); `sft --dataset dolly --out PATH`; `pretrain --corpus chatmix`;
`data chatmix|dolly`; `train-tokenizer --sample-chars N`; `benchmark`.

## 1. Block ablation: pre-registered rule

Written and committed before any ablation seed ran.

Setup: the 5M TinyStories configuration of
[`deepseek-v4.md`](deepseek-v4.md#1b-language-model-pretraining-on-tinystories) (first 50 MB of
TinyStories, the existing 1024-token BPE, 6 layers, d_model 256, 8 heads, context 256, batch 32,
4000 steps, lr 1e-3, dropout 0.1, AdamW, `--final`: the model as training left it). Seeds 0-3.

| Arm | Flags | Block |
|---|---|---|
| A | none | LayerNorm, GELU MLP (d_ff 1024), learned positions, 8 MHA heads |
| B | `--norm rmsnorm --mlp-type swiglu --pos-encoding rope` | RMSNorm, SwiGLU (d_ff 704), RoPE, 8 MHA heads |
| C | B plus `--num-kv-heads 4 --d-ff 784` | B with 8 query / 4 key-value heads; d_ff widened so the parameters match |

Parameters must agree within 2% (A 5.067M, B 5.098M, C reported with the results).

Metric: held-out loss of the final-step model, test half (primary) and val half (check), per
token, so comparable across arms here and not with other corpora.

Verdict for a pair X, Y: **X is better** only if the four per-seed ranges of X and Y do not
overlap and the gap between them (min of Y minus max of X) exceeds the wider of the two ranges.
It must hold on the test and the val half; if only one half passes, the verdict is "mixed".
Otherwise: **no measurable difference**.

Decision, fixed now:

- A vs B decides the block of the chat model: B better, or no measurable difference, means the
  modern block; A better means the GPT-2 block.
- C vs B decides grouped-query attention (only if the modern block is chosen): C better, or no
  measurable difference, means 8 query / 4 key-value heads; B better means plain multi-head.
  GQA is smaller in the KV cache at equal parameters, so a tie goes to it.

### Result (4 seeds, final-step model)

| Arm | Params | Test loss s0 / s1 / s2 / s3 | Test range | Val loss s0 / s1 / s2 / s3 | Val range | Train loss s0 / s1 / s2 / s3 |
|---|---|---|---|---|---|---|
| A GPT-2 block | 5,066,752 | 1.4182 / 1.4200 / 1.4198 / 1.4164 | 1.4164-1.4200 | 1.5035 / 1.5090 / 1.5065 / 1.5055 | 1.5035-1.5090 | 1.4307 / 1.4341 / 1.4338 / 1.4311 |
| B modern | 5,098,496 (+0.63%) | 1.3424 / 1.3342 / 1.3317 / 1.3290 | 1.3290-1.3424 | 1.4324 / 1.4238 / 1.4238 / 1.4209 | 1.4209-1.4324 | 1.3489 / 1.3409 / 1.3393 / 1.3351 |
| C modern + GQA 8q/4kv, d_ff 784 | 5,073,344 (+0.13%) | 1.3375 / 1.3327 / 1.3326 / 1.3391 | 1.3326-1.3391 | 1.4288 / 1.4240 / 1.4252 / 1.4310 | 1.4240-1.4310 | 1.3434 / 1.3387 / 1.3401 / 1.3458 |

Arm A at seed 0 reproduces the dense row of the DeepSeek-V4 recipe (test 1.418, val 1.504, train
1.431), so the refactored model code leaves the GPT-2 block numerically unchanged.

| Pair | Test: gap vs wider range | Val: gap vs wider range | Verdict |
|---|---|---|---|
| A vs B | 1.4164 − 1.3424 = 0.0740 vs 0.0134 | 1.5035 − 1.4324 = 0.0711 vs 0.0115 | **B better** (mean test 1.3343 vs 1.4186, −0.084, −5.9%) |
| C vs B | ranges overlap | ranges overlap | **no measurable difference** (mean test +0.001 for C) |
| A vs C (not a pre-registered pair, same rule) | 1.4164 − 1.3391 = 0.0773 vs 0.0065 | 1.5035 − 1.4310 = 0.0725 vs 0.0070 | C better |

Decision by the rule above: the chat model uses the modern block, and, because C vs B is a tie,
8 query / 4 key-value heads.

Limits: the arms differ in three parts at once (norm, MLP, positions), so which part carries
the −0.084 is not separated. [INFERENCE] RoPE and SwiGLU are the likelier sources; nothing here
tests it. One corpus, one size (5M), 4000 steps (1.75 epochs). Wall times of the 12 runs (123-177 s)
are not comparable: the GPUs were shared with other jobs.
