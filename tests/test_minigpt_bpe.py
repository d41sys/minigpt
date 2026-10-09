from pathlib import Path

from minigpt.bpe import BPETokenizer


def test_unicode_round_trip() -> None:
    tok = BPETokenizer()
    tok.train("hello world hello there, héllo wörld 你好 你好 😊 😊", target_vocab_size=300)
    for text in ["héllo 你好世界 😊🚀 naïve café\n\tend", "unseen ✓ ∑ Ωmega", ""]:
        assert tok.decode(tok.encode(text)) == text


def test_merges_on_toy_corpus() -> None:
    # "low" x5, "lower" x2, "lowest" x3: pre-tokenized as "low", " lower", " lowest", ...
    corpus = "low low low low low lower lower lowest lowest lowest"
    tok = BPETokenizer()
    tok.train(corpus, target_vocab_size=258 + 3, specials=("<|endoftext|>", "<|pad|>"))
    merged = [tok.vocab[i] for i in tok.merges.values()]
    # (l,o) and (o,w) tie at 10 occurrences; canonical tie-break picks the smaller pair (l,o).
    assert merged[0] == b"lo"
    assert merged[1] == b"low"
    assert merged[2] == b" low"
    assert tok.encode(" low") == [tok.inv_vocab[b" low"]]


def test_specials_and_save_load(tmp_path: Path) -> None:
    tok = BPETokenizer()
    tok.train("abc abc abd abd", target_vocab_size=262)
    text = "abc<|endoftext|>abd"
    ids = tok.encode(text, allow_special=True)
    assert tok.special_to_id["<|endoftext|>"] in ids
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.encode(text, allow_special=True) == ids
    assert loaded.decode(ids) == text
