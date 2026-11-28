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

TBD
