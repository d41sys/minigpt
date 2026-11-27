import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minigpt import cli
from minigpt.data import MixedLoader, create_dataloader
from minigpt.model import GPTConfig, GPTModel
from minigpt.train import save_checkpoint, train

TINY = GPTConfig(vocab_size=50, context_length=8, d_model=16, num_layers=1, num_heads=2)
DOMAIN = [i % 25 for i in range(2000)]  # ids 0-24
REPLAY = [25 + i % 25 for i in range(4000)]  # ids 25-49


def splits(ids: list[int]) -> Callable[[object], tuple[list[int], list[int]]]:
    return lambda _tokenizer: (ids, ids[-400:])


def n_replay_rows(x: torch.Tensor) -> int:
    return int((x[:, 0] >= 25).sum())


@pytest.mark.parametrize(("frac", "expected"), [(0.0, 0), (0.25, 2), (0.5, 4), (1.0, 8)])
def test_mixed_loader_draws_the_requested_fraction_per_batch(frac: float, expected: int) -> None:
    loader = MixedLoader(DOMAIN, REPLAY, TINY.context_length, 8, frac)
    batches = [b for _, b in zip(range(5), loader, strict=False)]
    assert len(batches) == 5
    for x, y in batches:
        assert x.shape == y.shape == (8, TINY.context_length)
        assert n_replay_rows(x) == expected
        assert n_replay_rows(y) == expected  # targets come from the same source as inputs


def test_mixed_loader_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction"):
        MixedLoader(DOMAIN, REPLAY, 8, 8, 1.5)


def test_constant_lr_has_no_warmup_and_no_decay(tmp_path: Path) -> None:
    ids = [i % 50 for i in range(200)]
    loader = create_dataloader(ids, 8, batch_size=4)
    res = train(
        GPTModel(TINY),
        loader,
        loader,
        max_steps=4,
        warmup_steps=0,
        learning_rate=1e-3,
        eval_interval=1,
        checkpoint_dir=tmp_path,
        device=torch.device("cpu"),
        min_lr_ratio=1.0,
    )
    assert [row["lr"] for row in res["history"]] == [1e-3] * 4


@pytest.mark.parametrize("frac", [0.0, 0.5])
def test_init_loads_identical_weights_and_scores_the_replay_corpus(
    frac: float, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_tokenizer", lambda: SimpleNamespace(vocab_size=50))
    monkeypatch.setattr(cli, "shakespeare_splits", splits(DOMAIN))
    monkeypatch.setattr(cli, "tinystories_splits", splits(REPLAY))
    torch.manual_seed(0)
    init = tmp_path / "base.pt"
    save_checkpoint(init, GPTModel(TINY), None, 0, 0.0)
    out = tmp_path / "res.json"
    args = cli.build_parser().parse_args(
        [
            *["pretrain", "--init", str(init), "--replay-corpus", "tinystories", "--final"],
            *["--replay-frac", str(frac), "--steps", "3", "--eval-interval", "1"],
            *["--batch-size", "4", "--name", "cpt", "--results-out", str(out)],
            *["--lr", "0"],  # lr 0: the weights must not move
        ]
    )
    cli.cmd_pretrain(args)
    saved = torch.load(init, weights_only=True)["model_state_dict"]
    trained = torch.load(tmp_path / "runs/cpt/ckpt.pt", map_location="cpu", weights_only=True)
    for name, w in saved.items():
        assert torch.equal(trained["model_state_dict"][name], w.cpu()), name
    rec = json.loads(out.read_text())
    assert rec["init"] == str(init) and rec["replay_frac"] == frac
    assert rec["replay_val_loss"] > 0 and rec["replay_test_loss"] > 0
