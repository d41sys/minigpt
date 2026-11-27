# Domain adaptation by continued pretraining: replay and LR re-warm

Enterprise recipes for adding a domain to a pretrained model mix some general data back in
("replay") and re-warm the learning rate before decaying it again. This page adds the flags and
measures what they buy at toy scale. Toy result: one 5.07M-parameter base, one general corpus,
one domain, 600 steps.

## What was added

`minigpt pretrain` gained four flags (code: `MixedLoader` in `src/minigpt/data.py`,
`cmd_pretrain` in `src/minigpt/cli.py`; tests in `tests/test_minigpt_continued_pretrain.py`):

| Flag | Effect |
|---|---|
| `--init CKPT` | load the weights and the config (the architecture flags are ignored), then train with a new optimizer. The schedule is the existing one: linear warm-up over `steps // 20` steps from 0, then cosine to 10% of `--lr`. That is the LR re-warm |
| `--replay-corpus {tinyshakespeare,tinystories}` | also score this corpus's val and test halves at the end (`replay_val_loss`, `replay_test_loss` in `--results-out`) |
| `--replay-frac F` | each batch takes `round(F * batch_size)` windows from the replay corpus and the rest from `--corpus`. One pass over `--corpus` is one epoch, so the replay corpus must be the larger one |
| `--constant-lr` | constant `--lr` from step 0, no warm-up and no decay (the "no re-warm" arm) |

Both corpora use the one tokenizer, `runs/tokenizer.json`, as every other command (a checkpoint
does not store its tokenizer; `--init` checks that the vocabulary size matches). That BPE was
trained on Shakespeare and is reused for TinyStories (see `deepseek-v4.md` section 1b).

## Setup

- General corpus: first 50 MB of TinyStories (`--corpus tinystories`). Domain: tiny-Shakespeare
  (414k training tokens). Held-out halves as everywhere: 90/10 cut, then val / test halves.
- Base: dense default size (5.07M), `pretrain --corpus tinystories --seed 0 --steps 4000
  --final`. One base for every arm (seed 0, so seeds below vary data order and dropout masks, not
  the starting weights). It reproduces the earlier `ts_dense_s0` run: general test 1.4182.
- Continued pretraining (CPT) on Shakespeare, 600 steps, batch 32, 256 tokens, AdamW as for the
  base, dropout 0.1 as in the base config. 600 steps are 4.9M tokens, about 12 passes over the
  Shakespeare training tokens. A seed-0 pilot with a val evaluation every 25 steps picked the
  budget: re-warm val loss 3.1242 at step 599 against 3.1240 at 575 (still creeping down), and the
  constant-LR arm still fell 0.015 per 25 steps. Longer budgets would overfit Shakespeare.
- Arms (4 seeds, 0-3 each, final-step model scored with `--final`):
  - **re-warm**: LR 0 → 1e-3 (the base's peak) over 30 steps, cosine to 1e-4;
  - **replay 0.1 / 0.25 / 0.5** with the same re-warm. `round(F * 32)` = 3, 8, 16 windows per
    batch, so the true fractions are 0.094, 0.25, 0.5, and the domain windows per step are 29,
    24 and 16;
  - **const**: constant 1e-4 (the base's final LR), no warm-up, no decay.
- Domain loss = Shakespeare val and test halves. General loss = TinyStories val and test halves
  (`--replay-corpus tinystories`, also on the arms without replay, with `--replay-frac 0`).
  Base domain loss: `pretrain-eval` on the base checkpoint.
- Verdict rule, fixed before the 4-seed runs: two arms differ measurably when their four-seed
  ranges do not overlap and the gap between the ranges exceeds the wider of the two ranges.
  Every replay and const arm is compared with re-warm.

```
minigpt pretrain --corpus tinystories --seed 0 --steps 4000 --eval-interval 250 --final \
  --name cpt_base
minigpt pretrain-eval --checkpoint runs/cpt_base/ckpt.pt
# per arm A and seed S:
minigpt pretrain --init runs/cpt_base/ckpt.pt --corpus tinyshakespeare \
  --replay-corpus tinystories --seed S --steps 600 --eval-interval 25 --final \
  --name cpt_A_sS --results-out runs/logs/cpt_A_sS.json \
  --lr 1e-3                                    # re-warm
  --lr 1e-3 --replay-frac 0.1                  # also 0.25, 0.5
  --lr 1e-4 --constant-lr                      # const
```

## Results

Base before any CPT: domain val 5.963 / test 6.142 (it has never seen Shakespeare); general val
1.5035 / test 1.4182.

Final-step loss, seeds 0 / 1 / 2 / 3 (mean in brackets). Lower is better.

| Arm | Domain val | Domain test | General val | General test | Wall s |
|---|---|---|---|---|---|
| base (no CPT) | 5.9634 | 6.1418 | 1.5035 | 1.4182 | 206 (pretrain) |
| re-warm 1e-3 | 3.124 / 3.126 / 3.114 / 3.122 (3.122) | 3.433 / 3.413 / 3.407 / 3.412 (3.416) | 4.109 / 4.098 / 4.132 / 4.091 (4.107) | 4.007 / 4.003 / 4.033 / 3.992 (4.009) | 35-36 |
| replay 0.1 + re-warm | 3.121 / 3.119 / 3.115 / 3.119 (3.118) | 3.392 / 3.393 / 3.386 / 3.409 (3.395) | 1.844 / 1.842 / 1.846 / 1.840 (1.843) | 1.747 / 1.748 / 1.747 / 1.744 (1.746) | 35-38 |
| replay 0.25 + re-warm | 3.143 / 3.116 / 3.116 / 3.132 (3.127) | 3.408 / 3.373 / 3.392 / 3.391 (3.391) | 1.686 / 1.685 / 1.685 / 1.686 (1.686) | 1.594 / 1.593 / 1.593 / 1.593 (1.593) | 38-39 |
| replay 0.5 + re-warm | 3.197 / 3.178 / 3.177 / 3.199 (3.188) | 3.418 / 3.413 / 3.406 / 3.416 (3.413) | 1.592 / 1.590 / 1.592 / 1.591 (1.591) | 1.503 / 1.504 / 1.504 / 1.503 (1.504) | 38 |
| const 1e-4 | 3.365 / 3.362 / 3.354 / 3.369 (3.362) | 3.507 / 3.504 / 3.488 / 3.503 (3.501) | 2.856 / 2.886 / 2.917 / 2.852 (2.878) | 2.754 / 2.788 / 2.819 / 2.759 (2.780) | 35-37 |

Verdicts against re-warm (mean difference in brackets):

| Arm | Domain val | Domain test | General val | General test |
|---|---|---|---|---|
| replay 0.1 | none (−0.003) | none (−0.021) | lower (−2.264) | lower (−2.262) |
| replay 0.25 | none (+0.005) | none (−0.025) | lower (−2.422) | lower (−2.415) |
| replay 0.5 | higher (+0.066) | none (−0.003) | lower (−2.516) | lower (−2.505) |
| const 1e-4 | higher (+0.241) | higher (+0.084) | lower (−1.230) | lower (−1.229) |

Reading (verdicts are against re-warm, rule above):

- **Re-warm without replay forgets almost everything**: general test 1.418 → 4.009 (+2.59), while
  domain test falls from 6.142 to 3.416. Every replay arm has a measurably lower general loss
  (val and test), by 2.26 / 2.42 / 2.51 on test.
- **Replay costs the domain nothing measurable on the test half** at any fraction (mean
  differences −0.003 to −0.025, inside the seed ranges of 0.011-0.036). On val, replay 0.1 and
  0.25 are also within the rule; replay 0.5 is
  measurably worse (+0.066), which is the one visible cost, and it also has the fewest domain
  windows per step [INFERENCE: dilution, not interference].
- **More replay keeps more general skill**: forgetting is +0.328 / +0.175 / +0.085 for
  F = 0.1 / 0.25 / 0.5, with seed ranges of 0.001-0.004 on general test, so these three are
  measurably different from one another. The decrease is roughly halving per doubling of F.
- **Low constant LR is dominated by replay**: against re-warm it is measurably worse on the domain
  (+0.241 val, +0.084 test) and measurably better on general (−1.23), but replay 0.1 beats it on
  both axes (domain val 3.118 vs 3.362, test 3.395 vs 3.501; general test 1.746 vs 2.780). It is
  also the arm with the widest seed range on general loss (0.065). Not re-warming is not a
  substitute for replay here.
- **Trade-off**: at this budget replay 0.25 looks like the working point (domain as good as
  re-warm, general loss +0.175 over the base); replay 0.5 buys another 0.09 of general loss for
  a worse domain val.
- Wall time, train loop only: 35-39 s per CPT run on a shared GPU (base 206 s), replay does not
  slow it measurably.

## Muon base

Same recipe with `--optimizer muon` for the base (`cpt_mbase`, 4000 steps, seed 0, general val
1.4583 / test 1.3689, domain val 5.964 / test 6.123) and for every CPT arm (peak 1e-3, const
1e-4). Not part of the pre-registered main comparison; same verdict rule.

| Arm (Muon) | Domain val | Domain test | General val | General test | Wall s |
|---|---|---|---|---|---|
| base (no CPT) | 5.9640 | 6.1225 | 1.4583 | 1.3689 | 307 (pretrain) |
| re-warm 1e-3 | 3.133 / 3.127 / 3.126 / 3.127 (3.128) | 3.413 / 3.423 / 3.441 / 3.429 (3.427) | 4.303 / 4.259 / 4.312 / 4.224 (4.274) | 4.242 / 4.204 / 4.239 / 4.165 (4.213) | 50-57 |
| replay 0.1 + re-warm | 3.115 / 3.105 / 3.116 / 3.112 (3.112) | 3.402 / 3.400 / 3.393 / 3.395 (3.397) | 1.809 / 1.806 / 1.811 / 1.808 (1.808) | 1.710 / 1.709 / 1.711 / 1.709 (1.709) | 60-61 |
| replay 0.25 + re-warm | 3.120 / 3.096 / 3.106 / 3.110 (3.108) | 3.399 / 3.394 / 3.393 / 3.406 (3.398) | 1.650 / 1.650 / 1.649 / 1.650 (1.650) | 1.556 / 1.556 / 1.555 / 1.555 (1.556) | 60-61 |
| replay 0.5 + re-warm | 3.148 / 3.140 / 3.139 / 3.158 (3.146) | 3.402 / 3.403 / 3.387 / 3.401 (3.398) | 1.554 / 1.552 / 1.554 / 1.554 (1.554) | 1.463 / 1.462 / 1.464 / 1.462 (1.463) | 60-61 |
| const 1e-4 | 3.364 / 3.367 / 3.358 / 3.366 (3.364) | 3.506 / 3.517 / 3.502 / 3.508 (3.508) | 2.760 / 2.777 / 2.788 / 2.723 (2.762) | 2.676 / 2.696 / 2.707 / 2.642 (2.680) | 55-57 |

Verdicts against re-warm (mean difference in brackets):

| Arm (Muon) | Domain val | Domain test | General val | General test |
|---|---|---|---|---|
| replay 0.1 | none (−0.016) | none (−0.029) | lower (−2.466) | lower (−2.503) |
| replay 0.25 | none (−0.020) | none (−0.029) | lower (−2.625) | lower (−2.657) |
| replay 0.5 | none (+0.018) | none (−0.028) | lower (−2.721) | lower (−2.750) |
| const 1e-4 | higher (+0.235) | higher (+0.082) | lower (−1.512) | lower (−1.532) |

- Same ordering as the dense base: replay cuts the general-loss rise from +2.84 to +0.34 / +0.19 /
  +0.09 (dense: +2.59 to +0.33 / +0.18 / +0.09), and no replay fraction has a measurably worse
  domain test or val loss, including 0.5 (dense 0.5: val worse by 0.066).
- Muon-base arms are not compared with dense-base arms: the bases differ in more than the
  optimizer's end state, and only one base seed exists for each.

## Limits

- One base, one domain pair, 5M parameters, 600 steps, one re-warm peak (1e-3) and one constant
  rate (1e-4). Seeds vary data order and dropout only. The ranking of arms could move with
  another peak LR or budget; none was tuned.
- The domain is tiny and very unlike the base corpus, so the forgetting is far larger than a
  language-model-scale domain shift would give [INFERENCE]. Shakespeare val and test halves
  differ by about 0.3 nats (different plays), so compare val with val and test with test.
- Replay arms see fewer domain windows per step, so at a fixed step budget replay 0.5 has seen
  half the domain tokens of re-warm; a fixed domain-token budget was not run.
- The const arm is not converged at 600 steps (val still falling), so its position partly
  reflects the budget, not only the schedule.
- The base's val-selected and final checkpoints coincide (best val at the last step), so `--init`
  starts from the final base.
- The pilot that chose the budget used seed 0, which is also one of the four evaluated seeds.
