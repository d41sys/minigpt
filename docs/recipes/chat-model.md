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

## 2. The chat model

One model, one seed (0). Every number in sections 2-5 comes from that single run; nothing was
repeated, so there is no seed range to put around them.

### Data and licences

| Stage | Dataset | Used | Source | Licence |
|---|---|---|---|---|
| Pretrain | TinyStories | first 400 MB of the train file | `roneneldan/TinyStories` | CDLA-Sharing-1.0 |
| Pretrain | FineWeb-edu, `sample-10BT` | first 120,000 documents (about 580 MB) | `HuggingFaceFW/fineweb-edu`; rows via the datasets-server API; derived from Common Crawl | ODC-By 1.0 (and Common Crawl terms of use) |
| Chat SFT | Dolly-15k | 15,011 rows, last 100 held out; 13,942 trained on (969 rows whose prompt fills the 512-token window leave no response and are dropped) | `databricks/databricks-dolly-15k` | CC BY-SA 3.0 |
| Benchmark (eval only) | ARC-Easy test (2,376), PIQA validation (1,838), OpenBookQA test (500), HellaSwag validation (10,042) | all rows | `allenai/ai2_arc`, `baber/piqa`, `allenai/openbookqa`, `Rowan/hellaswag` | CC BY-SA 4.0 (the ARC card); the other three cards do not state a licence on the Hub (PIQA, OpenBookQA, HellaSwag: not checked beyond that). Not redistributed here |

Licences are the ones on the Hugging Face dataset cards, read on 2026-11-28. All three training
sets allow redistributing derived weights for non-commercial research; the two share-alike
terms (CDLA-Sharing, CC BY-SA) attach to the data and its adaptations, and whether trained weights
count as an adaptation is a legal question this recipe does not settle. The Alpaca path (CC BY-NC)
is not used. The raw mix is a shuffled stream of documents (TinyStories and FineWeb-edu
interleaved, seed 0) so that the 90/10 cut holds both: 232.8M training, 12.9M validation, 12.9M
test tokens (258.7M in total, 3.78 bytes per token). Footprint on disk: 0.98 GB text plus 0.52 GB
tokens, 1.5 GB.

### Tokenizer

The repo's own byte-level BPE, vocabulary 6,400 (256 bytes, 2 specials, 6,142 merges), as in
minimind. The pure-Python trainer is O(merges x text), so it was trained on the **first 10M
characters** of the shuffled mix (not the whole 979 MB; the document separator replaced by a
blank line so "endoftext" does not become merges): 1,167 s on one CPU core, 3.82 bytes per token
on the first 100k characters. No change to `bpe.py`: encoding the full corpus takes 63 s with
`tokenize_cached` (6 processes) and is byte-identical to `encode` (tested).

### Model and how its size was chosen

RMSNorm, SwiGLU, RoPE and 8 query / 4 key-value heads, by the rule of section 1 (modern block
better than GPT-2 block on 4 of 4 seeds; GQA tied, so used). The ablation finished before the
chat model was started, so the architecture was not chosen after seeing chat results.

| | |
|---|---|
| d_model / layers / heads / kv heads | 512 / 16 / 8 / 4 |
| SwiGLU width, context, vocabulary | 1,408, 512, 6,400 |
| Parameters (tied embeddings, linear biases kept) | **50,557,440** |

Size: I measured training throughput of four sizes on the shared GPUs before committing
(bf16, micro-batch 16-32, context 512): 26.9M 156k tokens/s (GPU 3), 38.7M 149k (GPU 2), 50.6M
123k (GPU 2), 58.7M 83k (GPU 3, the busier card). I took the largest size under the 64M cap that gets about 20
tokens per parameter (Chinchilla's rule of thumb: 1.0B tokens for 50M) in about 2 GPU-hours, with
the corpus I could fetch and store (232.8M tokens, so about 4 epochs). It is a budget rule, not a
tuned optimum; no other size was trained.

Pretrain: `pretrain --corpus chatmix`, AdamW, peak lr 5e-4 with 750 warm-up steps and cosine decay
to 5e-5, weight decay 0.1, clip 1.0, bf16 autocast, dropout 0, batch 32 x 4 accumulation x 512 =
65,536 tokens per step, 15,000 steps = 983M tokens = 4.2 epochs. The learning rate and token
budget were fixed before the run and not tuned. Final-step model (the best-val checkpoint is the
last step): loss per token **train 2.245 (ppl 9.44), val 2.382 (10.83), test 2.365 (10.64)**; about
0.90 bits per byte on test (2.365 nats / 3.78 bytes per token / ln 2). The val curve was still
falling at the end (2.459 at step 10,000, 2.409 at 14,500, 2.407 at 15,000, on the 20-batch
validation subset the loop uses); val - train is 0.14, so 4 epochs overfit only mildly.

SFT: `sft --dataset dolly --epochs 3 --lr 1e-4 --batch-size 16`, AdamW, constant lr, prompt tokens
masked, the Alpaca instruction template, fp32. Chosen before the run; not tuned. Mean loss per
epoch 2.600, 2.075, 1.601. The drop from epoch 2 to 3 is large for a 14k-example set, so the
model is partly memorising Dolly answers [INFERENCE]; no held-out SFT loss was measured and no
other epoch count was tried.

## 3. Benchmarks

`minigpt benchmark`: zero-shot, each option scored by its summed log-probability given the
context (the lm-evaluation-harness prompts: `Question: ...\nAnswer:` for ARC-Easy and PIQA,
the question stem for OpenBookQA, the cleaned activity-plus-context for HellaSwag; the option is
scored with a leading space). Every row of each split (n below). `acc` is the summed score the
task asked for; `acc_norm` divides by the option's length in characters (the harness's
`acc_norm`) and is shown because the sum favours short options. No template, no few-shot.
`se` = sqrt(p (1 − p) / n). Two models differ only if their gap exceeds about 2 √2 se.

| Task | n | Chance | Pretrain acc ± se | SFT acc ± se | Pretrain acc_norm ± se | SFT acc_norm ± se | Gap needed (2√2 se) | SFT − pretrain (acc) |
|---|---|---|---|---|---|---|---|---|
| ARC-Easy | 2,376 | 0.250 | 0.376 ± 0.010 | 0.356 ± 0.010 | 0.362 ± 0.010 | 0.350 ± 0.010 | 0.028 | −0.021 |
| PIQA | 1,838 | 0.500 | 0.580 ± 0.012 | 0.577 ± 0.012 | 0.571 ± 0.012 | 0.562 ± 0.012 | 0.033 | −0.003 |
| OpenBookQA | 500 | 0.250 | 0.150 ± 0.016 | 0.168 ± 0.017 | 0.268 ± 0.020 | 0.262 ± 0.020 | 0.047 | +0.018 |
| HellaSwag | 10,042 | 0.250 | 0.271 ± 0.004 | 0.271 ± 0.004 | 0.278 ± 0.005 | 0.274 ± 0.004 | 0.013 | +0.000 |

Reading, with the rules above:

- **Pretrained and SFT do not separate on any task**: all four gaps are under the 2 √2 se
  threshold. SFT neither helped nor measurably hurt multiple-choice scores. (Post hoc: a paired
  comparison on the same ARC-Easy rows gives −0.021 with a paired se of 0.009, about 2.3 se,
  borderline; the stated rule does not call it.)
- **Above chance** by more than 4 se: ARC-Easy (0.36-0.38 vs 0.25), PIQA (0.58 vs 0.50),
  HellaSwag (0.271 vs 0.25, 4.7 se, a small margin: +2 points). These are small effects of a
  50M model on 233M unique tokens, not capabilities.
- **OpenBookQA `acc` is below chance** (0.150 and 0.168 vs 0.25, −6 and −5 se): the summed score
  picks the wrong option more often than random. [INFERENCE] the right answers are longer than
  the distractors and the sum penalises length. With length normalisation (0.268, 0.262) it
  does **not separate from chance** (0.9 and 0.6 se). So the model shows no OpenBookQA skill by
  either score.
- Length normalisation changes ARC-Easy, PIQA and HellaSwag by at most 0.01, inside one se.

Zero-shot multiple choice at this scale is near chance by construction; a 50M-parameter model
trained on 1B tokens is expected to be there. No external model was downloaded or compared.

## 4. Ten fixed prompts

Written before the model existed, run once on the SFT checkpoint through
`generate --chat --max-tokens 150 --top-k 40 --seed 0`, greedy (`--temperature 0`) and
`--temperature 0.7`. Unedited, bad ones included. A reply that ends mid-sentence was cut at 150
tokens, not by the model. The only correct answer is the capital of France; everything else is
wrong, repetitive or off-topic.

```
=== 1 [greedy] What is the capital of France?
The capital of France is Paris
=== 1 [t0.7] What is the capital of France?
The capital of France is Paris
=== 2 [greedy] Explain what photosynthesis is in two sentences.
A photosynthesis is a process that converts the energy from sunlight into water.  A photosynthesis is a process that converts the energy from sunlight into water.  A photosynthesis is a process that converts the energy from sunlight into water.  A photosynthesis is a process that converts the energy from sunlight into water.  A photosynthesis is a process that converts the energy from sunlight into water.
=== 2 [t0.7] Explain what photosynthesis is in two sentences.
Planet Earth is the second most common and most active planet in the solar system. Planet Earth is the third largest planet in the solar system. Planet Earth is the third largest planet in the solar system with 73.94% of the total number of planets. Planet Earth is the third largest planet in the solar system with 73.91% of the total number of planets. Planet Earth is the third largest planet in the solar system. Planet Earth is the third growing planet in the solar system. Planet Earth is the third growing planet in the solar system. Planet Earth is the third growing planet in the solar system. Planet Earth is
=== 3 [greedy] Write a short story about a cat who learns to swim.
It was a sunny afternoon when I saw a cat. The cat was so small and fluffy that I didn't realize it was a cat. I'm not sure if the cat was a good friend or a bad friend. 

I went to the cat and said, "Hi, cat! What are you doing?" The cat meared, "I'm swimming! I'm so good at swimming. I'm a good friend. Do you want to be my friend?" 

The cat meared again, "I'm so happy you like to swim. I'm so glad you like swimming. I'm so glad you like swimming. I'm so glad you like swimming. I'm going to go swimming every
=== 3 [t0.7] Write a short story about a cat who learns to swim.
Once upon a time, there was a cat in a small town called Melville. She was a nice and friendly cat, but she was very bossy. She would often find her way up to the top of the house to swim, but instead of giving up, she would clap and chirp. This made the cat very happy.

One summer, it started to snow a lot. The cat had to think about what to do to become the best friend. She would go by the door and look up at the sky. She would go to the far side of the city and look out over the snowy mountains. She would clap and chirp, enjoying the view from up there.
=== 4 [greedy] Give me three tips for studying effectively.
1. Be honest
2. Be honest
3. Be honest
4. Be honest
5. Be honest
6. Be honest
7. Be honest
8. Be honest
9. Be honest
10. Be honest
=== 4 [t0.7] Give me three tips for studying effectively.
1. Take a look at the world in a different way.  This will help you understand what is going to change when you are someone's school.
2. Take the time to study.  It will help you stay focused on what is going to change.  This will help you understand what is going to change.  It will help you think about what is going to change.  This will help you understand what is going to change.  Before should you look at the world in a different way.  This will help you understand what will be going to change in the world and what will be going to change.  This will help you understand the current state of the world.  Once you have a good understanding
=== 5 [greedy] What is 12 plus 15?
12 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus 15 plus
=== 5 [t0.7] What is 12 plus 15?
12 plus 15 is considered as one plus 15.
=== 6 [greedy] Translate 'good morning' into Spanish.
Great morning

Great morning

Great morning

Great morning

Great morning

Great morning

Great morning

Great morning

Great morning

Great morning

Great morning

Great morning
=== 6 [t0.7] Translate 'good morning' into Spanish.
Great morning
=== 7 [greedy] Who wrote Romeo and Juliet?
Romeo and Juliet was written by Juliet, a Romanian author who is credited with the founding of the Romans.
=== 7 [t0.7] Who wrote Romeo and Juliet?
Paul Albert Bendige in the year 1980
=== 8 [greedy] Write a haiku about the sea.
The sea is a deep, dark, deep, dark, deep, dark, deep, dark, deep, dark, deep, dark, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep, deep,
=== 8 [t0.7] Write a haiku about the sea.
This will be a wonderful night.
=== 9 [greedy] What is the difference between a virus and a bacterium?
Several different types of bacteria are known to be in the same genus as eukaryotic cells. For example, a bacteria from the same genus Eukaryotic are known to be in the same genus as eukaryotic cells. Several different types of bacteria are known to be in the same genus as eukaryotic cells. For example, a bacteria from the same genus Eukaryotic cells are known to be in the same genus Eukaryotic cells.
=== 9 [t0.7] What is the difference between a virus and a bacterium?
Viralani and his team are in the same category as well.
=== 10 [greedy] Tell me a joke.
I am a generative model. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a person or a history of being. I do not understand humor like a
=== 10 [t0.7] Tell me a joke.
I am a generative model. I can do many things differently than I am. For example, running a bike is a very demanding sport and I don't like it. But when I am a young person I might be able to think of a joke. 

I can do what I can do by doing the work of the model. I can make it yourself. So do a few things. 

I can help you with your tasks. I can help you do things your way through life. I can help you with the people around you. I can help you with the people around you. I can help you with the things that you can do. If you can, I can help you with the
```

The model follows the instruction format (it answers, in prose, in the right register) and
stops on its own for short questions; it does not know facts, cannot translate or do
arithmetic, and loops under greedy decoding. That is the expected quality of 50M parameters and
233M unique tokens, and it is why the benchmark scores above sit near chance.

## 5. Cost

The GPUs (4x RTX 4090) are shared with other users' jobs that kept them at 0-100% utilisation
during these runs, so wall times and tokens per second are lower bounds on what an idle card
does and vary with the neighbours. One process per card, my cards only.

| Stage | Hardware | Wall time | Steps | Throughput |
|---|---|---|---|---|
| Data download (`data chatmix`: 400 MB TinyStories, 120k FineWeb-edu rows) | network, 1 CPU core | 73 min (the datasets server rate-limits: HTTP 429, back-off 15-120 s) | 1,200 requests | |
| Tokenizer training | 1 CPU core | 1,167 s | 6,142 merges on 10M characters | |
| Tokenising the corpus | 6 CPU processes | 63 s | 258.7M tokens | about 15 MB/s of text |
| Block ablation (12 runs of 5M parameters, section 1) | GPU 2 and 3 | 123-177 s per run, 30 min of training time in all | 4,000 each | |
| Pretrain, 50.6M parameters | GPU 2 | 5,486 s of training (includes the validation passes) + 1,124 s of start-up, token-cache load and scoring the train, val and test halves = 6,610 s | 15,000 | 179,189 tokens/s |
| SFT on Dolly-15k | GPU 2 | 259 s | 3 epochs, 2,616 steps of 16 | 34,263 tokens/s (padding not counted) |
| Benchmarks, 4 tasks | pretrain: GPU 3; SFT: GPU 2 | 310 s; 238 s (HellaSwag 214 s and 164 s of that) | 14,756 rows each | |

About 2.6 GPU-hours in all, against the 6 allowed.

## 6. Use it

The final SFT checkpoint (weights only, no optimizer state, 202,279,555 bytes) and its tokenizer
are gitignored; on the Korea server they sit at `~/mg/main/runs/chat/` and on the Mac in the
canonical clone's `runs/chat/`.

```
minigpt --tokenizer runs/chat/tokenizer.json generate --chat --checkpoint runs/chat/ckpt.pt \
  --prompt "What is the capital of France?" --temperature 0
```

Reproduce, from scratch (about 2.5 GPU-hours plus the tokenizer's 20 CPU-minutes):

```
minigpt data chatmix && minigpt data dolly
minigpt train-tokenizer --corpus data/raw/chatmix.txt --vocab-size 6400 --sample-chars 10000000 \
  --out runs/chat/tokenizer.json
A="--norm rmsnorm --mlp-type swiglu --pos-encoding rope --num-kv-heads 4 --d-model 512 \
  --num-layers 16 --num-heads 8 --context-length 512"
minigpt --tokenizer runs/chat/tokenizer.json pretrain --corpus chatmix --name chat_pre $A \
  --dropout 0 --lr 5e-4 --steps 15000 --batch-size 32 --grad-accum 4 --bf16 --eval-interval 500 \
  --final --seed 0
minigpt --tokenizer runs/chat/tokenizer.json sft --base runs/chat_pre/ckpt.pt --dataset dolly \
  --epochs 3 --lr 1e-4 --batch-size 16 --out runs/chat_sft/ckpt.pt
minigpt --tokenizer runs/chat/tokenizer.json benchmark --checkpoint runs/chat_sft/ckpt.pt
```

## 7. Limits

- **One seed** for the chat model. Its loss and benchmark numbers have no seed range; the
  stated standard errors are sampling error over the benchmark rows only, not training noise.
  Pretrain-vs-SFT gaps of a few points could also come from the training seed.
- **Small data**: 233M unique training tokens (about 4 epochs), two sources (children's stories and
  educational web text), one SFT set of 14k human-written answers. The pretrain mix was cut
  at what could be fetched and stored (the download took 73 min because of rate limits).
- **Near-chance benchmarks are expected** and reported as such; the pretrained and SFT models are
  not separable on any of the four tasks, OpenBookQA shows no skill by either score, and the
  above-chance margins on ARC-Easy, PIQA and HellaSwag are small.
- Size, learning rates, epochs and token budget were not tuned; the size rule is a budget rule.
  SFT epoch 3 probably over-fits (loss 1.60 and falling fast); no held-out SFT loss was measured.
- The tokenizer saw 10M of 979M characters.
- The ablation is at 5M parameters, 1.75 epochs, one corpus, and changes three things at once;
  it chose the block, it did not show the block matters at 50M.
- No external model was downloaded or compared, and the benchmark prompts are the harness's, so
  the numbers can differ from other harnesses' by a point or two on choices of tokenisation.
- The 10 prompts test chat behaviour qualitatively; they are not a score.
