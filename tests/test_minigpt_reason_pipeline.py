import torch
import torch.nn as nn

from minigpt.reason import Task, evaluate_reasoning, reasoning_tokenizer, sft_trace_dataset


class Oracle(nn.Module):
    """Stand-in model whose greedy decode is the reference solution (or a wrong answer)."""

    def __init__(self, wrong: bool) -> None:
        super().__init__()
        self.tok = reasoning_tokenizer()
        self.wrong = wrong
        self.anchor = nn.Parameter(torch.zeros(1))  # gives `parameters()` a device

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        eos = self.tok.eos_token_id
        rows = []
        for row in x.tolist():
            a, b = self.tok.decode(row[:-1]).split("+")
            sol = [*self.tok.encode(Task(int(a), int(b) + int(self.wrong)).solution()), eos]
            rows.append(row + (sol + [eos] * max_new_tokens)[:max_new_tokens])
        return torch.tensor(rows)


def test_evaluate_reasoning_scores_a_perfect_and_a_wrong_solver() -> None:
    tok = reasoning_tokenizer()
    good = evaluate_reasoning(Oracle(wrong=False), tok, n_per_digit=4, seed=1)
    assert good["accuracy"] == 1.0
    assert good["format"] == 1.0
    lens = [good["per_digit"][d]["mean_tokens"] for d in range(1, 6)]
    assert lens == sorted(lens)
    assert lens[0] < 50 and lens[-1] < 100  # cut after the first EOS, not padded to the limit
    bad = evaluate_reasoning(Oracle(wrong=True), tok, n_per_digit=4, seed=1)
    assert bad["accuracy"] == 0.0
    assert bad["format"] == 1.0
    assert evaluate_reasoning(Oracle(wrong=False), tok, n_per_digit=4, seed=1) == good


def test_sft_trace_dataset_scores_only_the_trace_and_eos() -> None:
    tok = reasoning_tokenizer()
    t = Task(47, 85)
    ds = sft_trace_dataset([t], tok)
    x, y = ds[0]
    assert len(ds) == 1
    assert x[:6].tolist() == tok.encode(t.prompt)
    assert (y[:5] == -100).all()  # positions that would predict prompt tokens
    assert y[y != -100].tolist() == [*tok.encode(t.solution()), tok.eos_token_id]
