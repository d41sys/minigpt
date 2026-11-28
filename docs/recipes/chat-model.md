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
