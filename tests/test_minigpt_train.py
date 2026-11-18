from pathlib import Path

import torch

from minigpt.bpe import BPETokenizer
from minigpt.data import create_dataloader
from minigpt.dpo import PreferenceDataset, PreferenceSample
from minigpt.model import GPTConfig, GPTModel
from minigpt.sft import SFTDataset, format_prompt
from minigpt.train import load_checkpoint, train

CPU = torch.device("cpu")
TINY = GPTConfig(vocab_size=50, context_length=8, d_model=16, num_layers=1, num_heads=2)


def _ids() -> list[int]:
    torch.manual_seed(0)
    return torch.randint(0, 50, (8 * 8 + 1,)).tolist()  # 8 windows


def test_checkpoint_exists_even_when_val_never_improves(tmp_path: Path) -> None:
    ids = _ids()
    loader = create_dataloader(ids, 8, batch_size=2, shuffle=False)
    torch.manual_seed(0)
    model = GPTModel(TINY)
    # lr=0 keeps every val loss equal to the initial one, so nothing "improves"
    res = train(
        model, loader, loader, max_steps=2, eval_interval=1, learning_rate=0.0,
        checkpoint_dir=tmp_path, device=CPU,
    )  # fmt: skip
    assert Path(res["checkpoint_path"]).exists()
    load_checkpoint(res["checkpoint_path"], GPTModel(TINY))


def test_without_val_loader_the_final_model_is_saved(tmp_path: Path) -> None:
    loader = create_dataloader(_ids(), 8, batch_size=2, shuffle=False)
    torch.manual_seed(0)
    model = GPTModel(TINY)
    res = train(model, loader, None, max_steps=3, eval_interval=1, checkpoint_dir=tmp_path,
                device=CPU)  # fmt: skip
    loaded = GPTModel(TINY)
    payload = load_checkpoint(res["checkpoint_path"], loaded)
    assert payload["step"] == 2
    for a, b in zip(model.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(a, b)


def test_grad_accum_matches_the_larger_batch(tmp_path: Path) -> None:
    ids = _ids()
    models = []
    for name, batch, accum in (("accum", 2, 2), ("big", 4, 1)):
        torch.manual_seed(0)
        model = GPTModel(TINY)
        loader = create_dataloader(ids, 8, batch_size=batch, shuffle=False)
        train(model, loader, None, max_steps=1, warmup_steps=1, grad_accum_steps=accum,
              checkpoint_dir=tmp_path / name, device=CPU)  # fmt: skip
        models.append(model)
    for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-5)


def test_fully_masked_examples_are_skipped() -> None:
    tok = BPETokenizer()
    tok.train("Below is an instruction. Write a response. hello world good bad", 280)
    max_length = len(tok.encode(format_prompt("hi"), allow_special=True)) + 8
    long_prompt = "hi " * 100  # the prompt alone overflows max_length: no target survives
    records = [
        {"instruction": "hi", "output": "hello"},
        {"instruction": long_prompt, "output": "hello"},
    ]
    assert len(SFTDataset(records, tok, max_length=max_length)) == 1
    pairs = [
        PreferenceSample("hi", "good", "bad"),
        PreferenceSample(long_prompt, "good", "bad"),
    ]
    assert len(PreferenceDataset(pairs, tok, max_length=max_length)) == 1
