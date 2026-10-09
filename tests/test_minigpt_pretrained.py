import pytest
import torch
from minigpt.pretrained import from_pretrained


@pytest.mark.network
@pytest.mark.slow
def test_gpt2_greedy_matches_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    hf_tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    hf_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = hf_tok("Hello, I'm a language model,", return_tensors="pt").input_ids
    with torch.no_grad():
        expected = hf_model.generate(ids, max_new_tokens=10, do_sample=False)
    ours = from_pretrained("gpt2").eval()
    got = ours.generate(ids, max_new_tokens=10, temperature=0.0)
    assert got.tolist() == expected.tolist()
