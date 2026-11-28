import random
import re
from typing import Any

import torch
import torch.nn as nn

from minigpt.model import GPTConfig, GPTModel
from minigpt.reason import build_batch, per_token_log_probs, reasoning_tokenizer
from minigpt.tools import (
    MAX_CALLS,
    Chain,
    calculator,
    call_counts,
    chain_reward,
    chain_task,
    chain_trace_dataset,
    draw_task,
    eval_tasks,
    held_out,
    rollout,
    train_chain_grpo,
)

TOK = reasoning_tokenizer()
TINY = GPTConfig(
    vocab_size=TOK.vocab_size, context_length=128, d_model=32, num_layers=2, num_heads=4
)


class Oracle(nn.Module):
    """A 'policy' that plays a fixed script per prompt: it predicts the next character of
    `prompt + script` while the text so far is a prefix of it, and EOS once the text deviates
    (for example when the environment's observation differs from the script)."""

    def __init__(self, scripts: dict[str, str]) -> None:
        super().__init__()
        self.scripts = scripts
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        pad = TOK.special_to_id["<|pad|>"]
        logits = torch.zeros(*x.shape, TOK.vocab_size)
        for b, row in enumerate(x.tolist()):
            for t, tid in enumerate(row):
                if tid == pad:
                    break
                text = TOK.decode(row[: t + 1])
                full = next((p + s for p, s in self.scripts.items() if text.startswith(p)), "")
                nxt = full[len(text)] if full.startswith(text) and len(text) < len(full) else None
                logits[b, t, TOK.eos_token_id if nxt is None else TOK.encode(nxt)[0]] = 10.0
        return logits, None


def run(
    scripts: dict[str, str], max_new_tokens: int = 100, tools: bool = True
) -> tuple[list[str], list[list[int]], list[list[bool]]]:
    model: Any = Oracle(scripts)
    comps, keeps = rollout(model, TOK, [TOK.encode(p) for p in scripts], max_new_tokens, 0.0, tools)
    return [TOK.decode([i for i in c if i != TOK.eos_token_id]) for c in comps], comps, keeps


def test_calculator_does_exactly_one_binary_operation() -> None:
    assert calculator("12+7") == "19"
    assert calculator("5-30") == "-25"
    assert calculator("1+2+3") is None and calculator("12") is None and calculator("a+1") is None


def test_chain_tasks_are_valid_and_the_second_call_uses_the_first_observation() -> None:
    rng = random.Random(0)
    for d in (1, 2, 3):
        for _ in range(100):
            t = chain_task(rng, d)
            assert t.mid >= 0 and int(t.answer) >= 0
            assert len(str(t.b)) == d or d == 1
    t = Chain(123, 456, 100, "+", "-")
    pieces = [text for text, _ in t.tool_trace()]
    assert pieces[0] == "<call>123+456</call>" and pieces[1] == "<out>579</out>"
    assert pieces[2] == "<call>579-100</call>" and pieces[-1] == "<answer>479</answer>"
    assert eval_tasks(5, (2, 3)) == eval_tasks(10, (2, 3))[:5] + eval_tasks(10, (2, 3))[10:15]
    assert not held_out((2, 3)) & {chain_task(rng, 3, held_out((2, 3))).prompt}


def test_rollout_pauses_at_a_call_and_resumes_after_the_observation() -> None:
    a, b = Chain(123, 456, 100, "+", "-"), Chain(7, 5, 2, "-", "+")  # prompts of two lengths
    scripts = {t.prompt: "".join(s for s, _ in t.tool_trace()) for t in (a, b)}
    texts, _, keeps = run(scripts)
    assert texts == list(scripts.values())
    assert texts[0] == "<call>123+456</call><out>579</out><call>579-100</call><out>479</out>" + (
        "<answer>479</answer>"
    )
    # keep is False on exactly the observation characters
    for text, keep in zip(texts, keeps, strict=True):
        spans = [m.span() for m in re.finditer(r"<out>[^<]*</out>", text)]
        assert [i for i, k in enumerate(keep) if not k] == [
            i for s, e in spans for i in range(s, e)
        ]


def test_environment_output_replaces_what_the_model_would_have_written() -> None:
    p = "12+7-3="
    texts, comps, keeps = run({p: "<call>12+7</call><out>99</out><answer>96</answer>"})
    # the script's wrong observation is never read: the true one is appended, the script breaks
    assert texts == ["<call>12+7</call><out>19</out>"]
    assert comps[0][-1] == TOK.eos_token_id and keeps[0][-1] is True


def test_malformed_call_gets_a_marked_observation_and_the_episode_goes_on() -> None:
    p = "1+2-1="
    texts, _, keeps = run({p: "<call>1+2+3</call><out>?</out><call>1+2</call><out>3</out>"})
    assert call_counts(texts[0]) == (2, 1) and texts[0].count("<out>?</out>") == 1
    assert keeps[0].count(False) == len("<out>?</out>") + len("<out>3</out>")


def test_call_cap_and_length_cap_end_the_episode_unfinished() -> None:
    p = "1+1-1="
    texts, *_ = run({p: "<call>1+1</call><out>2</out>" * 9}, max_new_tokens=200)
    assert texts[0].count("</call>") == MAX_CALLS + 1 and texts[0].count("<out>") == MAX_CALLS
    short, *_ = run({p: "<call>1+1</call><out>2</out><answer>2</answer>"}, max_new_tokens=20)
    assert "<answer>" not in short[0]


def test_without_tools_a_call_tag_is_just_text() -> None:
    texts, *_ = run({"1+1-1=": "<call>1+1</call><answer>1</answer>"}, tools=False)
    assert texts == ["<call>1+1</call><answer>1</answer>"]


def test_observation_tokens_are_outside_the_loss_mask() -> None:
    t = Chain(12, 34, 5, "+", "-")
    script = "".join(s for s, _ in t.tool_trace())
    _, comps, keeps = run({t.prompt: script})
    prompt = TOK.encode(t.prompt)
    x, y = build_batch([prompt], comps, 0, keeps)
    n_obs = keeps[0].count(False)
    assert n_obs == len("<out>46</out>") + len("<out>41</out>")
    assert (y[0] != -100).sum().item() == len(comps[0]) - n_obs
    torch.manual_seed(0)
    logps, mask = per_token_log_probs(GPTModel(TINY), x, y)
    obs_at = [len(prompt) - 1 + j for j, k in enumerate(keeps[0]) if not k]
    assert not mask[0, obs_at].any() and (logps[0, obs_at] == 0).all()
    assert mask.sum().item() == len(comps[0]) - n_obs


def test_sft_traces_mask_observations_and_direct_traces_skip_the_call() -> None:
    tasks = [Chain(3, 4, 0, "+", ""), Chain(30, 4, 1, "+", "-")]  # an easy task and a chain
    assert tasks[0].prompt == "3+4=" and tasks[0].answer == "7"
    easy = [s for s, _ in tasks[0].tool_trace()]
    assert easy == ["<call>3+4</call>", "<out>7</out>", "<answer>7</answer>"]
    tool = chain_trace_dataset(tasks, TOK, "tool", direct_frac=1.0)
    direct, call = tool.examples
    assert (direct[1] != -100).sum().item() == len("<answer>7</answer>") + 1  # + EOS
    pieces = tasks[1].tool_trace()  # direct_frac leaves a chain alone
    assert (call[1] != -100).sum().item() == sum(len(s) for s, k in pieces if k) + 1
    think = chain_trace_dataset(tasks[1:], TOK, "think")
    assert (think.examples[0][1] != -100).sum().item() == len(tasks[1].solution()) + 1
    rng = random.Random(0)
    assert all(not draw_task(rng, (2, 3), 1.0, frozenset()).op2 for _ in range(20))
    assert all(draw_task(rng, (2, 3), 0.0, frozenset()).op2 for _ in range(20))


def test_reward_arrives_at_the_end_of_the_episode() -> None:
    t = Chain(12, 34, 5, "+", "-")
    ok = "<call>12+34</call><out>46</out><call>46-5</call><out>41</out><answer>41</answer>"
    reward = chain_reward(call_cost=0.02)
    assert abs(reward(t, ok) - (1.0 + 0.1 - 0.04)) < 1e-9
    assert abs(reward(t, ok.replace("<answer>41", "<answer>40")) - (0.1 - 0.04)) < 1e-9
    partial = "<call>12+34</call><out>46</out>"  # no answer: nothing is paid before the end
    assert abs(reward(t, partial) - (-0.02 - 0.2)) < 1e-9
    assert abs(reward(t, "<think>x</think><answer>41</answer>") - 1.1) < 1e-9  # no-tool arm
    hack = chain_reward(call_cost=0.02, call_bonus=0.3)
    assert abs(hack(t, partial) - (0.3 - 0.02 - 0.2)) < 1e-9
    bad = "<call>1+2+3</call><out>?</out>"
    assert abs(hack(t, bad) - (-0.02 - 0.2)) < 1e-9  # a malformed call earns no bonus


def test_chain_grpo_runs_on_a_tiny_model() -> None:
    torch.manual_seed(0)
    policy = GPTModel(TINY)
    ref = GPTModel(TINY)
    ref.load_state_dict(policy.state_dict())
    res = train_chain_grpo(
        policy,
        ref,
        TOK,
        chain_reward(0.02),
        steps=2,
        tasks_per_step=2,
        group_size=4,
        digits=(1, 1),
        max_new_tokens=24,
        device=torch.device("cpu"),
    )
    assert [r["step"] for r in res["history"]] == [0.0, 1.0]
    assert all(0.0 <= r["unfinished"] <= 1.0 for r in res["history"])
