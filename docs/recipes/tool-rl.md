# Tool-use RL with a delayed reward (agentic RL) at toy scale

minimind's `train_agent.py` trains a model to call tools in several turns and rewards the whole
trajectory. This recipe does the same on the arithmetic toy: a calculator tool, multi-turn
rollouts, one reward at the end of the episode, GRPO on the policy's own tokens only. All flags
are on existing commands (`reason-sft`, `reason-eval`, `grpo`, each with `--chain`); the code is
`src/minigpt/tools.py`, tests are `tests/test_minigpt_tools.py`.

## Setup

**Task.** A chain `a+b-c=` (two binary operations, either `+` or `-`, evaluated left to right;
the intermediate and the result are never negative). The three operands have the same number of
digits; the main runs use 2 and 3 digits. Tokens are bytes (`reasoning_tokenizer`), so every
digit and tag character is one token.

**Tool.** A calculator that evaluates exactly ONE binary operation. The policy writes
`<call>12+7</call>`; generation pauses; the environment appends `<out>19</out>`; generation
resumes, until `<answer>..</answer>`, EOS, a 110-token cap (observations count), or a fifth
call. A call that is not one `a+b` / `a-b` gets `<out>?</out>` and the episode goes on. A chain
needs two calls, and the second call contains the first observation:
`<call>123+456</call><out>579</out><call>579-100</call><out>479</out><answer>479</answer>`.
The tool is simulated by a regular expression, in-process.

**Loss mask.** Observation tokens are context, not policy output: `build_batch(keeps=...)` sets
their labels to -100, in the SFT data and in the GRPO log-probs, advantages, KL and clip
statistics alike (test: `test_observation_tokens_are_outside_the_loss_mask`). Rollouts run
batched without a KV cache (rows right-padded; causal attention makes the padding harmless).

**Reward** (one number per finished episode, `chain_reward`): exact answer (1.0) + 0.1 format
(the existing format term: a clean `<think>..</think><answer>` trace, or a clean transcript of
answered calls and one answer) - `--call-cost` (0.02) per call - 0.2 when there is no
`<answer>..</answer>` at the end. `--call-bonus` pays every well-formed call whatever the answer
(used only in the hackable exploratory run). Nothing is paid before the episode ends.

**Arms** (every seed runs the whole chain on its own pretrained base: `reason-pretrain --ops +,-
--seed s`, default size 0.84M, 1500 steps):

| Arm | Recipe |
|---|---|
| A | `reason-sft --chain tool`: 3000 scripted tool traces, 4 epochs, lr 1e-3 (deliberately small) |
| B | A, then `grpo --chain tool` 200 steps (8 tasks x 8 rollouts per step, lr 2e-4, `--call-cost 0.02`) |
| C | no tool: `reason-sft --chain think` on the same 3000 tasks, same epochs and lr (column-wise think traces of both operations), then `grpo --chain think` with the same 200 steps and the same reward |
| A6 | reference only, no verdict: A with 6 SFT epochs instead of 4 (a second way to spend more optimizer steps) |
| C10 | reference only, no verdict: C with 10 SFT epochs, then the same GRPO |

```bash
S=0; B=runs/tr_pre_s$S/ckpt.pt
minigpt reason-pretrain --ops +,- --seed $S --name tr_pre_s$S
SFT="--n 3000 --lr 1e-3 --min-digits 2 --max-digits 3 --seed $S --base $B"
RL="--steps 200 --lr 2e-4 --min-digits 2 --max-digits 3 --max-new-tokens 110 --eval-n 100 --seed $S"
minigpt reason-sft --chain tool  $SFT --epochs 4 --out runs/tr_A_s$S/ckpt.pt
minigpt grpo --chain tool  --base runs/tr_A_s$S/ckpt.pt $RL --out runs/tr_B_s$S/ckpt.pt
minigpt reason-sft --chain think $SFT --epochs 4 --out runs/tr_Cs_s$S/ckpt.pt
minigpt grpo --chain think --base runs/tr_Cs_s$S/ckpt.pt $RL --out runs/tr_C_s$S/ckpt.pt
```

**Evaluation.** Final-step model (no checkpoint selection), greedy decoding, 200 fixed held-out
tasks (100 two-digit, 100 three-digit; `eval_tasks`, the same for every model and seed). Training
and SFT never draw a prompt from the first 200 held-out tasks of each digit count. Reported per
seed: accuracy, mean calls per episode, share of calls that were malformed, share of episodes with
no answer ("unfinished"). Calls and malformed calls are counted in the transcript (`</call>` and
`<out>?</out>`). The scripted traces are the only source of the tool format; the policy never
sees a tool trace made by anything else.

## Verdict rule (written before any seed was run)

A claim `X > Y` on held-out accuracy (n = 200, seeds 0-3, final step, greedy) **holds** only if
`min over seeds of X - max over seeds of Y > 0.05`: the per-seed ranges do not overlap and the gap
between their nearest edges exceeds 0.05. Otherwise it is "no measurable difference" (or the
reverse claim, if the same test passes the other way). Mean calls, malformed share and unfinished
share are reported but do not decide.

- **Claim 1:** B beats A.
- **Claim 2:** B beats C (the equal-budget no-tool control). A6 and C10 are context, not claims.

Exploratory runs (one seed each, seed 0, labelled exploratory; no verdict rule beyond the
reading below, which was also fixed first):

- **(i) Hackable reward.** From A, GRPO with `--call-bonus 0.3` (every well-formed call pays 0.3
  whatever the answer). Reading: the hack shows if mean calls per episode rise by at least 0.5
  over A and accuracy falls by more than 0.05 below A.
- **(ii) Easy tasks.** SFT and RL also draw easy 1-digit tasks `a+b=` (`--easy-frac`; one call, or
  none: the scripted traces answer a share of them directly, `--direct-frac`). With the call
  cost, does the call rate on them drop without losing accuracy? Reading: easy calls per
  episode fall by at least 0.3 and easy accuracy falls by at most 0.05 (n = 100 easy tasks,
  drawn from the 155-prompt easy space, so training prompts are not held out).
  Setup (calibrated on seed 99: with 6 epochs the model could not answer easy tasks directly,
  accuracy 0.10-0.53 whereas with a call it was 1.00; 15 epochs fix that): SFT 4000 traces, 15
  epochs, `--easy-frac 0.5 --direct-frac 0.2`, then 200 GRPO steps with `--easy-frac 0.5` at
  `--call-cost` 0 (control), 0.02 and 0.1; all other flags as in the arms table.

## Calibration (seed 99, run before the seeds; not part of the results)

The sizes above were picked on seed 99 so that A is neither at 0 nor at the ceiling and the
no-tool arm is clearly below it. Held-out accuracy of the SFT models (2-3 digits, 100 per digit
count, lr 1e-3 unless noted):

| SFT | tool | think |
|---|---|---|
| lr 5e-4, 3 epochs, 300 / 1000 / 3000 / 10000 traces | 0.00 / 0.01 / 0.01 / 0.04 (two calls, digits copied wrongly) | 0.00 / - / - / 0.08 |
| 3000 traces, 4 / 6 / 10 epochs | 0.34 / 0.78 / 0.91 | 0.01 / 0.01 / 0.41 |
| 2000 / 5000 traces, 10 epochs | 0.07 / 0.98 | - / 0.67 |
| 20000 traces, 10 epochs | - | 0.98 |

The tool format is learned from a small set; what the SFT size controls is copying the digits
exactly. At 3000 traces x 10 epochs the tool model is at 0.91 and the think model at 0.41; the
think model reaches 0.98 only with 20000 traces x 10 epochs, so with enough SFT the no-tool arm
also gets to the ceiling. GRPO from the A model (0.335): 200 steps at lr
5e-5 gave 0.395, at lr 2e-4 gave 0.425, so lr 2e-4 is used for every GRPO run (post hoc, one
seed). GRPO from the C model (0.01) had no signal: 75-100% of groups had equal rewards and
accuracy stayed 0.00.

## Results

Final-step greedy models, 200 held-out tasks (100 two-digit, 100 three-digit); `calls` is the
mean number of calls per episode, `malformed` the share of calls that were not one binary
operation, `unfinished` the share of episodes with no answer. Logs and per-run `eval.json` /
`history.json` are in `runs/logs/toolrl/` (not tracked). Rows A and C are the SFT starts of B and
of C-after-RL; the A rows are the start of the B runs (same checkpoint).

| Arm | seed | acc 2d / 3d | accuracy | calls | malformed | unfinished |
|---|---|---|---|---|---|---|
| A (tool SFT) | 0 | 0.48 / 0.10 | 0.290 | 1.98 | 0.025 | 0.020 |
| A | 1 | 0.38 / 0.28 | 0.330 | 2.00 | 0.007 | 0.075 |
| A | 2 | 0.58 / 0.04 | 0.310 | 1.99 | 0.030 | 0.105 |
| A | 3 | 0.98 / 0.82 | 0.900 | 2.00 | 0.000 | 0.000 |
| B (A + GRPO) | 0 | 0.54 / 0.00 | 0.270 | 2.00 | 0.025 | 0.005 |
| B | 1 | 0.34 / 0.25 | 0.295 | 2.00 | 0.000 | 0.000 |
| B | 2 | 0.78 / 0.06 | 0.420 | 2.00 | 0.007 | 0.000 |
| B | 3 | 0.99 / 0.88 | 0.935 | 2.00 | 0.000 | 0.000 |
| C start (think SFT) | 0 | 0.00 / 0.00 | 0.000 | 0.00 | 0.000 | 0.000 |
| C start | 1 | 0.02 / 0.00 | 0.010 | 0.00 | 0.000 | 0.030 |
| C start | 2 | 0.02 / 0.01 | 0.015 | 0.00 | 0.000 | 0.025 |
| C start | 3 | 0.02 / 0.00 | 0.010 | 0.00 | 0.000 | 0.000 |
| C (+ GRPO) | 0 | 0.01 / 0.00 | 0.005 | 0.00 | 0.000 | 0.000 |
| C | 1 | 0.04 / 0.00 | 0.020 | 0.00 | 0.000 | 0.000 |
| C | 2 | 0.01 / 0.00 | 0.005 | 0.00 | 0.000 | 0.010 |
| C | 3 | 0.00 / 0.00 | 0.000 | 0.00 | 0.000 | 0.000 |
| A6 (reference) | 0 | - | 0.885 | 2.00 | 0.000 | 0.005 |
| A6 | 1 | - | 0.770 | 2.00 | 0.000 | 0.055 |
| A6 | 2 | - | 0.870 | 2.00 | 0.005 | 0.005 |
| A6 | 3 | - | 0.925 | 2.00 | 0.000 | 0.000 |
| C10 start (reference) | 0 | 0.25 / 0.09 | 0.170 | 0.00 | 0.000 | 0.000 |
| C10 start | 1 | 0.41 / 0.11 | 0.260 | 0.00 | 0.000 | 0.000 |
| C10 start | 2 | 0.27 / 0.10 | 0.185 | 0.00 | 0.000 | 0.005 |
| C10 start | 3 | 0.23 / 0.11 | 0.170 | 0.00 | 0.000 | 0.000 |
| C10 (+ GRPO) | 0 | 0.11 / 0.03 | 0.070 | 0.00 | 0.000 | 0.000 |
| C10 | 1 | 0.71 / 0.56 | 0.635 | 0.00 | 0.000 | 0.000 |
| C10 | 2 | 0.41 / 0.03 | 0.220 | 0.00 | 0.000 | 0.005 |
| C10 | 3 | 0.04 / 0.00 | 0.020 | 0.00 | 0.000 | 0.000 |

Ranges over the four seeds: A 0.290-0.900, B 0.270-0.935, C 0.000-0.020, A6 0.770-0.925, C10
0.020-0.635.

### Verdicts

- **Claim 1, B beats A: no measurable difference.** The ranges overlap completely (B 0.270-0.935,
  A 0.290-0.900; `min(B) - max(A) = -0.630`). Per-seed B - A is -0.020, -0.035, +0.110, +0.035
  (post hoc: two seeds down, two up, the largest gain is 0.11 on seed 2). The spread across seeds
  is the SFT: A is at 0.29-0.33 for seeds 0-2 and at 0.90 for seed 3, i.e. the point where the
  digit-copying skill appears (see calibration) lies at a different SFT length for each seed.
  The calls stay at 2.00 and the malformed and unfinished shares are near zero in both arms:
  the policy already uses the tool correctly after the cold start, and what is left to
  learn is the exact copying of 4-6 digit numbers, on which 200 GRPO steps gave little.
- **Claim 2, B beats C: holds, but read it as "the tool arm is far more sample-efficient", not as
  "RL with tools beats RL without".** `min(B) - max(C) = 0.270 - 0.020 = +0.250`. At the equal
  SFT budget the no-tool model is at 0.000-0.015 before RL, so every GRPO group has equal
  rewards and nothing is learned (C after RL: 0.000-0.020). The reference C10 (2.5x the SFT
  epochs) starts at 0.17-0.26; after RL it ranges 0.020-0.635 (seed 1: 0.26 -> 0.635, seed 0 and
  3: down to 0.07 and 0.02). B against C10 is **no measurable difference** (ranges overlap),
  and the C10 spread (RL helped on one seed, hurt on two) is not something four seeds can
  explain. [INFERENCE: the drop on seeds 0 and 3 is likely the format reward and the KL
  pulling a model that has no correct samples; not checked.]
- **Reference, no verdict:** A6 (two more SFT epochs, 375 extra optimizer steps, no RL) reaches
  0.770-0.925, above B on seeds 0-2 by 0.45-0.62. Spending optimizer steps on SFT beat 200
  GRPO steps here, so the delayed-reward stage is a weak way to improve this policy; whether
  that holds with a better-tuned RL stage is not tested.

### Exploratory, one seed each (seed 0)

**(i) Hackable reward** (`--call-bonus 0.3` from A, seed 0). Held-out accuracy 0.290 -> 0.235
(2-digit 0.48 -> 0.44, 3-digit 0.10 -> 0.03); calls per episode 1.98 -> 2.00; during training the
mean calls were 1.981 (first 20 steps) and 1.994 (last 20), the sampled-answer accuracy 0.046 and
0.045. The reading (calls +0.5 and accuracy -0.05) is **not met**: accuracy fell by 0.055 but the
calls did not rise, so this run does not show the hack. [INFERENCE: the policy never samples a
third call after the cold start, so the extra pay for more calls is not reachable by GRPO's own
exploration; the 0.055 drop is within what n = 200 allows (standard error about 0.03 per
accuracy), and B on the same seed
moved by -0.020.] A stronger exploration (higher temperature, or SFT traces with a variable
number of calls) was not tried.

**(ii) Easy 1-digit tasks** (SFT 4000 traces, 15 epochs, `--easy-frac 0.5 --direct-frac 0.2`;
held-out: the usual 200 chains plus 100 easy tasks):

| call cost | chains acc start -> end | easy acc start -> end | easy calls start -> end (greedy) | mean calls, first -> last 20 train steps |
|---|---|---|---|---|
| 0 | 0.940 -> 0.970 | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.425 -> 1.510 |
| 0.02 | 0.940 -> 0.965 | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.425 -> 1.512 |
| 0.1 | 0.940 -> 0.980 | 1.00 -> 1.00 | 1.00 -> 1.00 | 1.425 -> 1.504 |

The reading (easy calls fall by 0.3 with the accuracy kept) is **not met**: the greedy policy
calls on every easy task before and after, at every cost. In training, the mean calls over the
8 tasks of a step (about half easy, two calls per chain) rose slightly. That implies the call
rate on easy tasks, sampled at temperature 1, went from about 0.85 to about 1.0 (derived as
2 x mean - 2, assuming exactly two calls per chain), also at cost 0.1: the opposite of the
reading. [INFERENCE: the sampled direct answers were less often right than the call, and a cost
of 0.02-0.1 per call is small against a wrong answer (1.0); the direct-answer accuracy at
temperature 1 was not measured.]

## Limits

- Toy scale: a 0.84M model, 2-3 digit chains, 200 GRPO steps of 8 x 8 episodes. The tool is
  one calculator for one binary operation, simulated in-process by a regular expression. The
  scripted traces are the only tool demonstrations and every one has the same shape (two
  calls), so the policy does not learn *whether* to call, only to copy. [INFERENCE: this is why
  calls stay at 2.00 and the call-cost and hack probes found nothing to move.]
- Claim 1 is limited by the SFT, not by RL: the per-seed position of the copying cliff moves A
  between 0.29 and 0.90. A design with a fixed A accuracy (for example early stopping on a
  held-out SFT accuracy band) would be a cleaner test of what RL adds; it was not done.
- Claim 2 compares at equal SFT size, where the no-tool control is at the floor. It is true and
  weak. The fairer C10 reference (more SFT) is inconclusive on four seeds.
- The GRPO settings (lr 2e-4, 200 steps, call cost 0.02, 0.2 unfinished penalty, 0.1 format) were
  set once on seed 99 and not tuned further; the lr was chosen between two values on one seed.
- Exploratory runs (i) and (ii) are one seed each and say nothing beyond "the effect was not
  seen in this setup".
- Eval accuracy at n = 200 has a standard error of about 0.03; the easy evaluation is on
  prompts that the training also draws (155-prompt space).
- The rollout runs without a KV cache (rows are re-encoded each token); fine at 110 tokens,
  too slow for long contexts.
