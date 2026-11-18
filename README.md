# minigpt — GPT from scratch

[![CI](https://github.com/d41sys/minigpt/actions/workflows/ci.yml/badge.svg)](https://github.com/d41sys/minigpt/actions/workflows/ci.yml)

A small GPT built step by step in PyTorch: a byte-level BPE tokenizer, the
transformer, pretraining, loading real GPT-2 weights, classifier fine-tuning, SFT
and DPO. Only `torch.nn` primitives are used (Linear, Embedding, LayerNorm). The
tokenizer, attention, training loop, loss masking and DPO loss are all written here.

## Problem

Each stage of a modern LM pipeline is shown as a small, testable module:

| Stage | Module | What it shows |
|---|---|---|
| Tokenizer | `bpe.py` | byte-level BPE trainer/encoder/decoder, GPT-2 pre-tokenisation regex, JSON save/load, special tokens |
| Data | `data.py` | sliding-window `(x, y)` next-token dataset |
| Model | `model.py` | token + learned position embeddings (or RoPE via `--pos-encoding rope`), own causal MHA (optional `F.scaled_dot_product_attention`), pre-LN blocks, GELU MLP, weight-tied head, `generate(temperature, top_k)` with a per-layer KV cache (O(T) per new token instead of O(T²); falls back to crop-and-recompute past the context length) |
| Training | `train.py` | AdamW (decay on 2-D weights only), linear warmup + cosine LR, clip 1.0, periodic eval, best-val checkpoint, device mps → cuda → cpu |
| Pretrained | `pretrained.py` | HF GPT-2 124M safetensors → `GPTConfig(50257, 1024, 768, 12, 12)`, Conv1D transposes |
| Classifier | `finetune_cls.py` | head swap to 2 classes; train last block + final LN + head on UCI SMS Spam (balanced, 70/10/20) |
| SFT | `sft.py` | Alpaca prompt template, prompt tokens masked with `-100` |
| DPO | `dpo.py` | DPO loss (β = 0.1) against a frozen reference; pairs = dataset answer (chosen) vs SFT sample (rejected) |
| Constitutional AI | `cai.py` | judge-LLM critique/revise (SL) and A/B preference labels (RL) under random constitution principles; Ollama judge behind an injectable `ChatJudge` protocol |
| Eval | `evaluate.py` | perplexity, exact match, token F1, likelihood ranking, `Judge` protocol with an offline `MockJudge` and an Ollama LLM judge (1-5), weighted `EvalReport` |

## Architecture

![minigpt model architecture: BPE, embeddings, N pre-LN blocks, final LayerNorm, tied LM head; spam head on the last non-pad token](docs/diagrams/minigpt-model.svg)

![minigpt training pipeline: pretrain, spam classifier, SFT, DPO on 300 sampled rejected answers, eval of SFT and DPO](docs/diagrams/minigpt-pipeline.svg)

## Data and licences

Datasets are downloaded at run time from their sources. The only redistributed data are the Alpaca rows in `docs/judge-eval/` (`dpo_pairs.json`, `rows.json`, `rows_nll.json`), under CC BY-NC 4.0 with attribution to [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca).

| Data | Source | Licence |
|---|---|---|
| tiny-Shakespeare | [karpathy/char-rnn](https://github.com/karpathy/char-rnn) | public-domain text |
| Alpaca (SFT, DPO) | [tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca) | CC BY-NC 4.0, research use only; so are the SFT and DPO checkpoints derived from it |
| SMS Spam Collection | [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | CC BY 4.0 |
| GPT-2 124M weights | [Hugging Face `gpt2`](https://huggingface.co/openai-community/gpt2) | OpenAI's modified MIT licence |

## Quickstart

Run from the repo root. Downloads go to `data/raw/`, outputs to `runs/` (both gitignored).

```bash
uv run minigpt train-tokenizer                     # BPE, vocab 1024, on tiny-Shakespeare
uv run minigpt pretrain --steps 2000              # random-offset windows; --sampling aligned = old recipe
uv run minigpt pretrain --batch-size 16 --grad-accum 2 --bf16   # 2 micro-batches of 16 per step; bf16 autocast, no GradScaler
uv run minigpt pretrain-eval                       # val-half and test-half loss of runs/pretrain/ckpt.pt
uv run minigpt generate --prompt "ROMEO:"
uv run minigpt data spam && uv run minigpt finetune-cls
uv run minigpt data alpaca && uv run minigpt sft
uv run minigpt dpo
uv run minigpt cai-data --stage sl && uv run minigpt cai-train --stage sl   # Constitutional AI (judge on :11434)
uv run minigpt cai-data --stage rl && uv run minigpt cai-train --stage rl
uv run minigpt eval
uv run minigpt eval --n 100 --paired-vs sft --checkpoints sft=runs/sft/ckpt.pt,dpo=runs/dpo/ckpt.pt
```

Tests: `uv run pytest -q` (CI runs ruff, mypy and pytest on pushes to `main` and on every PR). Release: push a tag `vX.Y.Z` matching `pyproject.toml`'s version; CI reruns, then the wheel and sdist are attached to a GitHub Release. The GPT-2 parity test
needs internet and the extras:
`uv run --extra hf --extra test pytest -m network`.

## Results

All numbers come from one run on an Apple-silicon Mac (`device=mps`), seed 0, with the
CLI defaults. The model is 6 layers, d=256, 8 heads, context 256, 5.07M params, BPE vocab 1024.

All checkpoints were retrained from scratch after the repo split: pretrain, SFT and the spam classifier reproduce the earlier numbers exactly; DPO's epoch-2 loss and reward accuracy differ slightly (0.2427 / 0.940 vs 0.2314 / 0.95 before), cause not identified (same-seed reruns reproduce every metric and greedy response, but their weights differ by about 1e-5, so the cause may be run-to-run floating-point noise on MPS).

Published tokenizer stats and every result below predate the fix that stopped the BPE pre-tokenizer from dropping `_` (`snake_case` used to decode as `snakecase`). Retraining the tokenizer changes its merges, so those numbers do not reproduce on current code until the server runs are redone. The eval's Shakespeare perplexity also used to skip the last partial batch and now includes it.

| Experiment | Metric | Result |
|---|---|---|
| BPE on tiny-Shakespeare | vocab / compression | 1024 tokens, 2.41 bytes/token (17 s to train) |
| Pretrain, 2000 steps (about 6 min), aligned windows | test loss / perplexity of the val-selected checkpoint (step 800), 3 seeds | **3.875 / 48.2** (seeds 3.872 to 3.879); the earlier selection-biased "val 3.648 / ppl 40.27" is superseded |
| Pretrain, random-offset windows (new default) | same, 3 seeds | **3.767 / 43.2** (Δ −0.109, per-seed CIs exclude 0), see [overfitting section](#pretraining-overfitting-and-a-selection-biased-headline) |
| SMS Spam, head swap (790,786 trainable params) | test accuracy / F1 (n=300) | **0.940** / 0.939 (precision 0.965, recall 0.914) |
| SFT, Alpaca 900 rows × 3 epochs | train loss | 4.286 → 3.453 |
| DPO, 300 pairs, β=0.1, 2 epochs | loss at step 1 / epoch 2 mean | 0.6931 (= log 2) / 0.2427; reward accuracy 0.940 |
| GPT-2 124M parity (network test) | greedy 10 tokens vs HF `transformers` | identical (test passes) |
| Unit tests | `pytest` | 90 passed, 1 deselected (network) in 9 s on CI |

### Evaluation pipeline

`minigpt eval` runs four evals on every checkpoint and combines them into one weighted score:

- **Perplexity** on the held-out Alpaca responses (prompt tokens masked), mapped to [0, 1] as `1 − ln(ppl) / ln(vocab)` (1 = perfect, 0 = uniform guessing). Shakespeare perplexity is printed alongside but not aggregated.
- **Exact match** (SQuAD normalisation) on held-out Alpaca rows whose reference has ≤ 5 words (12 rows).
- **Token F1** on the first 50 held-out Alpaca rows by default; `--n 100` scores all 100 held-out rows.
- **Judge** (1–5) on the same rows. The default `MockJudge` maps ROUGE-L F1 onto the rubric so the eval runs offline; it duplicates token F1, so it is printed but **not aggregated**. `--judge ollama` swaps in a real LLM judge (`OllamaJudge`, see "Real LLM judge" below), which is aggregated by default.
- **Likelihood ranking** (printed, **not aggregated**): for each of the 100 held-out rows, does the model give the row's own reference a higher log-probability than the reference of row `(i+37) % 100` (the distractor scheme of [`judge_check.py`](docs/judge-eval/judge_check.py)) as the continuation of prompt `i`? Reported as accuracy with a Wilson 95% CI (chance 0.5), twice: summed log-probability, which favours short answers, and per-token (length-normalised). The gap between the two is the length bias. Built on the SFT example builder and `dpo.compute_sequence_log_probs`.
- **Stop rate** (printed, not aggregated): `stopped k/n`, how many of the greedy responses (at most 60 new tokens) ended with the end-of-text token instead of running to the limit.

The aggregate is the unweighted mean of perplexity score, EM and token F1 (`--w-ppl --w-em --w-f1` default 1, `--w-judge` default 0). `minigpt eval --checkpoints pretrain=runs/pretrain/ckpt.pt,sft=...`. Add `--paired-vs sft` for a paired bootstrap CI of each other checkpoint's per-row token F1 minus SFT (95% percentile CI, 10,000 row resamples, bootstrap p Holm-adjusted across the compared checkpoints) and `--rows-out PATH` to save per-row F1 and responses:

| Model | Alpaca ppl (held-out) | Shakespeare ppl | EM (n=12) | Token F1 (n=50) | Mock judge (ROUGE-L, 1-5) | Aggregate |
|---|---|---|---|---|---|---|
| pretrain | 409.23 | 40.27 (full validation split, on which this checkpoint was selected; test half: 48.20) | 0.000 | 0.048 | 1.00 | 0.060 |
| sft | 39.45 | 93.31 | 0.000 | 0.057 | 1.10 | 0.176 |
| dpo β=0.05 | 41.46 | 90.12 | 0.000 | 0.035 | 1.04 | 0.166 |
| dpo β=0.1 | 40.81 | 89.33 | 0.000 | 0.054 | 1.08 | 0.173 |
| dpo β=0.5 | 39.33 | 88.60 | 0.000 | 0.058 | 1.14 | 0.176 |

Reading, as measured:
- SFT cuts held-out Alpaca perplexity tenfold (409 → 39) and doubles Shakespeare perplexity (40 → 93): it learns the instruction format and forgets the pretraining domain. Almost all of the aggregate gain comes from this perplexity term.
- A 5M-parameter model trained on 900 instructions gets no exact match and near-floor F1; the generation metrics barely separate the checkpoints.
- An earlier version of this table measured perplexity on Shakespeare only, which ranked pretrain first. That compared the models on a domain SFT is meant to leave, so it was replaced.

#### Real LLM judge: Qwen 3.6 27B

`minigpt eval --judge ollama --judge-url http://HOST:11434 --judge-model qwen3.6:27b-ctx4k` scores each response against its reference with a fixed 1–5 rubric (`JUDGE_RUBRIC` in `evaluate.py`). The call uses Ollama's native `/api/chat` with temperature 0, seed 0, thinking off and a JSON schema that only admits an integer 1–5; any other reply raises instead of being guessed.

Run: the whole pipeline above retrained on one RTX 4090 (`device=cuda`, torch 2.13 cu129), then scored on all 100 held-out rows with `qwen3.6:27b-ctx4k` (Ollama model ID `7d4e56429f65`, Q4). Script, environment, per-row scores and every response: [`docs/judge-eval/`](docs/judge-eval/).

| Model | Alpaca ppl | Shakespeare ppl | EM (n=12) [95% CI] | Token F1 (n=100) [95% CI] | Qwen judge (n=100) [95% CI] | Aggregate (4 parts) |
|---|---|---|---|---|---|---|
| pretrain | 404.97 | 40.00 | 0.000 [0.000, 0.242] | 0.042 [0.034, 0.050] | 1.00 [1.00, 1.00] | 0.044 |
| sft | 39.92 | 93.41 | 0.000 [0.000, 0.242] | 0.046 [0.036, 0.057] | 1.00 [1.00, 1.00] | 0.129 |
| dpo β=0.05 | 42.51 | 90.51 | 0.000 [0.000, 0.242] | 0.062 [0.050, 0.075] | 1.00 [1.00, 1.00] | 0.130 |
| dpo β=0.1 | 41.61 | 90.16 | 0.000 [0.000, 0.242] | 0.057 [0.046, 0.068] | 1.00 [1.00, 1.00] | 0.130 |
| dpo β=0.5 | 39.50 | 91.04 | 0.000 [0.000, 0.242] | 0.061 [0.050, 0.073] | 1.00 [1.00, 1.00] | 0.133 |

CIs (printed by `eval` since this run): EM uses a Wilson interval over the 12 rows, token F1 and the judge a 10,000-resample row bootstrap of the mean. 0 of 12 exact matches is compatible with a true EM rate of up to 24%, so EM cannot separate these models at all. The judge's [1.00, 1.00] is a degenerate bootstrap (100 identical scores leave nothing to resample), not certainty; the informative bound is the share of responses scoring above 1: 0 of 100, Wilson 95% CI [0, 0.037] per checkpoint. A second scoring run of the same checkpoints (with the CI code) reproduced every response and every judge score exactly ([`rows.json`](docs/judge-eval/rows.json)). The 300 DPO preference pairs of this run are in [`dpo_pairs.json`](docs/judge-eval/dpo_pairs.json).

**The judge gives 1 to all 500 responses.** Two checks show this is a verdict, not a stuck judge ([`judge_check.json`](docs/judge-eval/judge_check.json)):

| Check (100 held-out rows) | Expected | Mean score | Share |
|---|---|---|---|
| Reference answer judged against itself | 5 | 4.88 | 97% score 5 |
| Another row's reference judged against this row's | 1 | 1.00 | 100% score 1 |

What it changes:
- **Token F1 differences are filler-word overlap.** The five highest-F1 responses (F1 ≈ 0.25) are text like "The collollolly of the collollos of the 10000…". Their overlap with the reference is "the", "of", "is".
- **The token-F1 verdict flips with the hardware.** On this CUDA run the paired test calls DPO β=0.05 and β=0.5 *better* than SFT (Δ +0.016 and +0.015, Holm p 0.049 and 0.007); on the earlier MPS run every DPO variant was at or below SFT. Same seeds and datasets; the floating-point paths differ, and so do the preference pairs sampled from each SFT model. The sign flips. A metric whose significant result reverses across devices is measuring noise in gibberish, so neither verdict says anything about quality. This CUDA run was not pre-registered, so the pre-registered MPS verdict above (no detectable difference) stands; the flip is reported as a replication discrepancy.
- **All five checkpoints are at the judge's floor.** SFT and DPO change perplexity and the answer format, not answer quality, at 5M parameters trained on Shakespeare. Rank agreement between judge and token F1 is undefined because every judge score is the same.
- Training on CUDA reproduced the MPS run closely: SFT loss 4.292 → 3.474 (MPS 4.286 → 3.453), DPO β=0.1 epoch-2 loss 0.2412 and reward accuracy 0.940 (MPS 0.2427, 0.940), pretrain test loss 3.844 (MPS 3.875).
- **Contamination check** ([`contamination.py`](docs/judge-eval/contamination.py), [`contamination.json`](docs/judge-eval/contamination.json)): 4 of the 100 held-out rows (12, 17, 26, 87) share at least one lowercase word 8-gram (instruction + input + answer) with the 900 SFT/DPO training rows, 0 with tiny-Shakespeare, and no held-out instruction appears verbatim in training. Dropping the 4 rows moves every checkpoint's mean token F1 by at most 0.0015 (e.g. SFT 0.046 → 0.048, DPO β=0.1 0.057 → 0.055), so overlap does not drive any number above. Whether the 4 overlaps are leaks or shared stock phrases was not inspected.

Pretrain log (excerpt; seed 0, `--sampling aligned --dropout 0.1`, the recipe that produced `runs/pretrain/ckpt.pt`; val = first half of the held-out 10%):

```text
device=mps train_tokens=414,007 val_tokens=23,000 test_tokens=23,001
params=5,066,752
step     0 train 6.9923 val 6.9426 lr 1.00e-05
step   400 train 3.5398 val 3.7873 lr 9.46e-04
step   800 train 2.9612 val 3.5264 lr 7.31e-04
step  1200 train 2.3985 val 3.6315 lr 4.40e-04
step  1999 train 1.8529 val 3.9659 lr 1.00e-04
val loss step0=6.9964 best=3.5264 drop=49.6%
val-selected checkpoint (step 800): val loss=3.5264 ppl=34.00 (n=89 windows)  test loss=3.8754 ppl=48.20 (n=89 windows)
```

`minigpt generate --prompt "ROMEO:"`:

```text
ROMEO:
It's name, my wilt bend?

ROMEO:
Ay, the dead; and I would thou wilt swear,
I fear thy sound not that thou shouldst wilt say it thee.

ROMEO:
Then, sin me, thou art a man that thou art a torment,
To have thee in thee to death; let
```

SFT sample on a held-out instruction (greedy):

```text
### Instruction: Calculate the cost of 4 apples at $2.50 each.
### Model: The calcon is 1000.
### Reference: The cost of 4 apples at $2.50 each is $10.00.
```

How to read these results:
- **Overfitting.** About 414k training tokens is small for 5M params. Val loss bottoms out at step 800 and rises after that while train loss keeps falling. The checkpoint keeps the best-val weights, so every downstream stage starts from step 800 of the aligned-window run. Random-offset windows reduce but do not remove this (see the pretraining section below).
- **SFT.** The model learns the response *format*: short declarative answers that reuse words from the instruction. It cannot learn facts or arithmetic, because it was pretrained only on Shakespeare.
- **DPO.** It learns the preference: reward accuracy rises with β and the loss always starts at exactly log 2. Held-out quality does not move detectably. All runs train on the **same 300 preference pairs** (sampled once, saved to `runs/dpo/pairs.json`, reused via `--pairs-file`), so only β or the seed differs:

  | Checkpoint | DPO loss (epoch 2) | Reward acc. | Token F1 (n=50) | Mock judge (1-5) |
  |---|---|---|---|---|
  | sft | – | – | 0.057 | 1.10 |
  | β=0.05, seed 0 | 0.386 | 0.923 | 0.035 | 1.04 |
  | β=0.1, seed 0 | 0.243 | 0.940 | 0.054 | 1.08 |
  | β=0.1, seed 1 | 0.243 | 0.940 | 0.048 | 1.10 |
  | β=0.1, seed 2 | 0.236 | 0.940 | 0.042 | 1.02 |
  | β=0.5, seed 0 | 0.051 | 0.977 | 0.058 | 1.14 |

  **Paired test on all 100 held-out rows.** The n=50 point values above carry no uncertainty, so every existing DPO checkpoint was re-scored on all 100 held-out rows (greedy, 60 new tokens) and compared with SFT by a paired bootstrap on per-row token F1 (10,000 row resamples, seed 0). The rule was fixed before scoring ([`docs/protocol-dpo-paired-bootstrap.md`](docs/protocol-dpo-paired-bootstrap.md)): a variant is **different** from SFT only if its 95% CI excludes 0 **and** its Holm-adjusted p (Holm over all 10 existing DPO checkpoints) is below 0.05; otherwise it is **no detectable difference**. SFT token F1 on the 100 rows is 0.069 (0.057 on the first 50).

  | Checkpoint | F1 (n=100) | F1 (first 50) | Δ F1 vs SFT | 95% CI | p | p (Holm) | Verdict |
  |---|---|---|---|---|---|---|---|
  | β=0.05, seed 0 | 0.054 | 0.035 | -0.0150 | [-0.0292, -0.0010] | 0.0364 | 0.364 | no detectable difference |
  | β=0.1, seed 0 | 0.068 | 0.054 | -0.0013 | [-0.0156, +0.0133] | 0.8685 | 1.000 | no detectable difference |
  | β=0.1, seed 1 | 0.063 | 0.048 | -0.0060 | [-0.0197, +0.0075] | 0.3916 | 1.000 | no detectable difference |
  | β=0.1, seed 2 | 0.055 | 0.042 | -0.0139 | [-0.0283, +0.0004] | 0.0550 | 0.440 | no detectable difference |
  | β=0.5, seed 0 | 0.065 | 0.058 | -0.0037 | [-0.0146, +0.0068] | 0.4896 | 1.000 | no detectable difference |

  Ten checkpoints were scored (`runs/dpo`, `runs/dpo_b*`, `runs/fx_*`), but they are five distinct models: β=0.1 seed 0 exists four times (`dpo`, `dpo_b0.1`, `fx_b0.1_a`, `fx_b0.1_b`), β=0.05 and β=0.5 twice each (`dpo_b*`, `fx_b*_a`). Same-seed reruns differ in weights by about 1e-5 (so the pre-registration's "none bit-identical, none merged" check passed), yet every one of the 100 greedy responses is identical within each group, so each group gives one row above and the same CI. The Holm adjustment was applied over the 10 pre-registered rows, as written. Merging the duplicates (**post-hoc**, 5 comparisons) gives Holm p of 0.182 (β=0.05), 0.220 (β=0.1 seed 2) and 1.0 for the rest, so the verdicts do not change.

  No variant is different from SFT, and every point estimate is below SFT. "No β improves on SFT" holds, but with 100 rows the CIs are about ±0.014 F1 wide, so a small gain or loss is not excluded: β=0.05 has a raw 95% CI that excludes 0 on the negative side (and β=0.1 seed 2 only just includes it), which does not survive Holm over 10 comparisons. The rows are the only source of uncertainty here: the 300 pairs and each seed's training are fixed, so the CIs do not cover variation across training seeds or pair samples (the seed spread above is a separate, rough look at that). The rejected samples come from the same weak model, so DPO mostly learns to move away from its own outputs: it optimises likelihood margins, not answer quality. Reported as measured.

  **NLL regulariser (Llama 3), exploratory, not pre-registered.** `minigpt dpo --nll-alpha 0.2` adds 0.2 × the length-normalised NLL of the chosen answers to the DPO loss (Llama 3 paper, section 4.1.4), and every run now logs the policy's per-token log-prob of chosen and rejected answers against the reference. Same 300 pairs ([`dpo_pairs.json`](docs/judge-eval/dpo_pairs.json)), β=0.1, 2 epochs, CUDA ([logs](docs/judge-eval/), [`rows_nll.json`](docs/judge-eval/rows_nll.json)):

  | Checkpoint | Epoch-2 DPO loss | Reward acc. | Chosen log-prob/token vs ref | Rejected vs ref | Alpaca ppl | Token F1 Δ vs SFT [95% CI], Holm p | Qwen judge |
  |---|---|---|---|---|---|---|---|
  | DPO | 0.2412 | 0.940 | **−0.054** | −0.694 | 41.61 | +0.010 [−0.004, +0.025], 0.145 | 1.00 |
  | DPO + NLL (α=0.2) | 0.2337 | 0.960 | **+0.068** | −0.476 | 39.69 | +0.011 [+0.001, +0.021], 0.074 | 1.00 |

  Plain DPO lowers the likelihood of the answers it is told to prefer; the NLL term reverses that and keeps held-out Alpaca perplexity at SFT level (SFT 39.92). Held-out quality does not move: both are no detectable difference from SFT on token F1 (Holm over 2), and the judge scores every response 1. The plain-DPO row is a rerun of the β=0.1 checkpoint above and reproduces its eval row exactly.

### Pretraining: overfitting and a selection-biased headline

The earlier headline (val loss 3.648, perplexity 40.27) picked the best of 11 validation evaluations and reported that same minimum, so it is optimistically biased. The model also overfits: the train set is 414k tokens in non-overlapping windows (about 40 epochs in 2000 steps) and val loss turns up after step 800. The held-out 10% is now cut deterministically into contiguous **val** (first 23,000 tokens, selects the checkpoint) and **test** (last 23,001 tokens, reported only). Protocol, written before any run: [`docs/protocol-pretrain-regularization.md`](docs/protocol-pretrain-regularization.md). Raw per-run JSON and `summary.json`: `docs/pretrain-regularization/`. Reproduce with `minigpt pretrain --name pre_<variant>_s<seed> --seed <seed> [--sampling aligned] [--dropout 0.2] --results-out ...` and `minigpt pretrain-compare`.

Headline, corrected: the val-selected baseline checkpoint has **test loss 3.875 (perplexity 48.2)**, mean over seeds 0, 1, 2 (range 3.872 to 3.879). The seed-0 rerun reproduces the checkpoint used downstream: same selected step (800) and the existing `runs/pretrain/ckpt.pt` scores the same on both halves under `minigpt pretrain-eval` (val 3.5264, test 3.8754). So 3.648 / 40.27 understated the loss by about 0.23 nats; the halves differ in difficulty (val 3.53 vs test 3.88), so part of that gap is the split and not only selection.

Tokenizer-independent: `pretrain-eval` also prints **bits per byte** (loss in bits times tokens per byte of that half). The test half has 2.298 bytes/token, so test loss 3.875 is **2.433 bits/byte** (random-offset mean 3.767: 2.365). Measured directly on the CUDA retrain (test loss 3.844): 2.414. For scale, nanoGPT's character-level Shakespeare baseline reports val loss 1.4697 nats/char, about 2.12 bits/char; that model is larger and uses a different split, so this is an order-of-magnitude comparison, not a ranking.

Three variants, seeds 0, 1, 2, same 2000-step budget, one factor changed at a time, selected on val, scored on test. `base`: aligned windows, dropout 0.1. `randoff`: random-offset windows (stride-1 windows, shuffled). `drop02`: aligned, dropout 0.2. Δ is test loss minus the same-seed baseline; the CI is a paired bootstrap over the 89 test windows (10,000 resamples), so it reflects test-set sampling noise only.

| Variant | seed | selected step | test loss (ppl) | Δ test vs base [95% CI] |
|---|---|---|---|---|
| base | 0 / 1 / 2 | 800 / 800 / 800 | 3.8754 (48.20) / 3.8785 (48.35) / 3.8720 (48.04) | n/a |
| drop02 | 0 | 1200 | 3.8282 (45.98) | −0.047 [−0.064, −0.030] |
| drop02 | 1 | 1400 | 3.8539 (47.17) | −0.025 [−0.041, −0.008] |
| drop02 | 2 | 1200 | 3.8408 (46.56) | −0.031 [−0.047, −0.016] |
| randoff | 0 | 1200 | 3.7408 (42.13) | −0.135 [−0.153, −0.116] |
| randoff | 1 | 1400 | 3.7646 (43.14) | −0.114 [−0.135, −0.093] |
| randoff | 2 | 1200 | 3.7946 (44.46) | −0.077 [−0.101, −0.053] |

Pre-registered rule: adopt a variant only if, on every seed, its test loss is lower than the baseline by at least 0.05 **and** the per-seed 95% CI excludes 0. `randoff` meets it on all three seeds (mean test loss 3.767, Δ −0.109); `drop02` does not (its gain is 0.025 to 0.047 on every seed, below the 0.05 margin). **Decision: random-offset sampling is the new default; dropout stays 0.1.** `--sampling aligned` restores the old windows. Neither variant removes the overfitting: val loss still rises after its minimum (`randoff` ends at val 3.41 to 3.42 vs minimum 3.35 to 3.37). The test half is 23k tokens of one contiguous passage adjacent to the validation half, so this is a within-corpus hold-out. The 0.05 margin and the choice of variants were fixed before the runs; nothing in this section is post-hoc.

The default change affects only new runs. SFT, DPO, the spam classifier and the eval tables in this README were trained from the old aligned-window checkpoint (`runs/pretrain/ckpt.pt`, not overwritten) and are **not retrained**; a bare `minigpt pretrain` no longer reproduces that checkpoint (use `--sampling aligned`).

## Post-training stages

Small-scale copies of post-training stages that labs describe publicly, run on verifiable addition tasks ([`reason.py`](src/minigpt/reason.py)). The sources are cited for the stage's shape only; nothing here claims to reproduce their results. No stage has been run yet, so no results are reported.

### On-policy distillation (OPD)

The student samples its own completions and, at every completion token, minimises the full-vocabulary KL between its next-token distribution and a frozen teacher's on that same prefix ([`opd.py`](src/minigpt/opd.py)). Mirrors the last stage of [DeepSeek-V4.1](https://arxiv.org/html/2609.19969) (§5.2.4, full-vocabulary OPD); background: [MiniLLM](https://arxiv.org/abs/2306.08543) (Gu et al. 2024) and [Thinking Machines, "On-policy distillation"](https://thinkingmachines.ai/blog/on-policy-distillation/) (2025).
Run: `minigpt opd` (teacher `runs/grpo/ckpt.pt`, student `runs/reason_sft/ckpt.pt` or `--student-config 64,2,4`; writes `runs/opd/ckpt.pt`). `--kl forward` swaps the direction; `--off-policy` is the baseline where the teacher samples and the student does plain SFT on those samples (`runs/opd/off_policy/ckpt.pt`).
Lessons to test: reverse KL is mode-seeking (the student commits to one teacher mode) and forward KL is mode-covering; training on the student's own prefixes is meant to fix exposure bias, since an off-policy student never sees its own mistakes corrected. The history logs reverse KL per token, greedy accuracy and mean length on one fixed seeded task set for both modes.

### Constitutional AI ([`cai.py`](src/minigpt/cai.py))

Not on addition tasks: this stage uses the Alpaca chat model from `runs/sft` and a judge LLM (Ollama) instead of a verifiable reward.

- **What:** the judge critiques the SFT model's answers against a random principle from a 7-line [constitution](docs/cai/constitution.md) and rewrites them; SFT on the rewrites is SL-CAI. The SL-CAI model then samples 2 answers per prompt, the judge picks the better one under a random principle (A/B order randomised, position-bias rate recorded), and DPO on those pairs is RL-CAI. 128 PG prompts in [`prompts.json`](docs/cai/prompts.json); the last 20 are held out for `cai-eval`.
- **Mirrors:** [Bai et al. 2022, Constitutional AI](https://arxiv.org/abs/2212.08073), the public method behind the Claude Opus 5.5 system card's statement that post-training aligns the model "with the values described in Claude's constitution". The constitution text is original, inspired by [Anthropic's](https://www.anthropic.com/constitution). One departure: DPO replaces the paper's preference model + RL.
- **Run** (needs `runs/sft` and the judge, e.g. `ssh -L 11434:localhost:11434 Korea`):

```bash
uv run minigpt cai-data --stage sl && uv run minigpt cai-train --stage sl   # writes runs/cai_sl
uv run minigpt cai-data --stage rl && uv run minigpt cai-train --stage rl   # writes runs/cai_rl
uv run minigpt cai-eval   # judge scores held-out prompts 1-5: sft vs cai_sl vs cai_rl
```

- **Lesson to test:** the labels are only as good as the judge reading a text file; `runs/cai/pairs_stats.json` shows how often it just picked the slot shown first.

## Design decisions

- **Byte-level BPE with GPT-2 regex pre-tokenisation.** Byte-level means no unknown tokens and lossless round-trips (tested on CJK and emoji). The regex stops merges from crossing word and punctuation boundaries. When two pairs have the same count, the smallest pair id wins, so training is deterministic. Encoding memoises each chunk, so encoding the 1.1 MB corpus takes seconds.
- **Own attention with an SDPA flag.** The explicit masked-softmax path is the reference implementation, and a test checks that it matches `F.scaled_dot_product_attention`. MPS's fused kernel has no dropout support, so in training on MPS the model falls back to the explicit path.
- **Weight tying.** The 124M config has exactly 124,439,808 parameters (tested on the `meta` device, so no memory is allocated). This matches GPT-2.
- **Checkpoints store `asdict(config)`, not a pickled dataclass.** This keeps them loadable with `torch.load`'s default `weights_only=True`.
- **The classifier reads the last *non-pad* token.** Causal attention means that position has seen the whole message and none of the padding. A test checks that extra padding does not change the logits.
- **DPO runs the policy with dropout off** (`eval()` mode, gradients still flow). Without this, dropout noise alone makes policy ≠ reference at step 0, and the first loss came out at 0.94 instead of log 2 in an earlier run.
- **No offline fake data.** Downloads raise an error on failure; they never fall back to fabricated samples. Default tests are fully offline. The HF parity test is marked `network` and `slow`.
- **CPU torch on Linux CI.** `[tool.uv.sources]` sends `torch` to the PyTorch CPU index on Linux only; macOS keeps the PyPI wheel, which includes MPS.
