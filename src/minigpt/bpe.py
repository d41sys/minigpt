"""Byte-level Byte-Pair Encoding (BPE) tokenizer from scratch.

Features:
- Full byte alphabet (0..255) ensures arbitrary UTF-8 coverage without out-of-vocab tokens.
- GPT-2 style regex pre-tokenization prevents merges across punctuation/whitespace boundaries.
- Deterministic merge training with canonical tie-breaking.
- JSON serialization and deserialization.
- Support for special tokens (<|endoftext|>, <|pad|>).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

BYTE_ALPHABET_SIZE = 256
DEFAULT_SPECIALS: tuple[str, ...] = ("<|endoftext|>", "<|pad|>")

# GPT-2 style pre-tokenization regex adapted for Python's standard `re` module
# Captures contractions, letters, digits, other symbols, and whitespace runs.
GPT2_SPLIT_PATTERN: re.Pattern[str] = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.IGNORECASE,
)


@dataclass
class BPETokenizer:
    """Byte-level BPE tokenizer."""

    vocab: dict[int, bytes] = field(default_factory=dict)
    inv_vocab: dict[bytes, int] = field(default_factory=dict)
    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    special_to_id: dict[str, int] = field(default_factory=dict)
    id_to_special: dict[int, str] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in vocabulary."""
        return len(self.vocab)

    @property
    def eos_token_id(self) -> int:
        """Token ID for <|endoftext|> if present, otherwise default to 256 or 0."""
        if "<|endoftext|>" in self.special_to_id:
            return self.special_to_id["<|endoftext|>"]
        return 0

    def _add_token(self, token_bytes: bytes) -> int:
        if token_bytes in self.inv_vocab:
            return self.inv_vocab[token_bytes]
        token_id = len(self.vocab)
        self.vocab[token_id] = token_bytes
        self.inv_vocab[token_bytes] = token_id
        return token_id

    def initialize(self, specials: Iterable[str] = DEFAULT_SPECIALS) -> None:
        """Initialize the byte alphabet (0..255) and reserve special token IDs."""
        self.vocab.clear()
        self.inv_vocab.clear()
        self.merges.clear()
        self.special_to_id.clear()
        self.id_to_special.clear()

        # 0..255 raw bytes
        for i in range(BYTE_ALPHABET_SIZE):
            self._add_token(bytes([i]))

        # Special tokens directly follow byte alphabet
        for s in specials:
            token_id = len(self.vocab)
            token_bytes = s.encode("utf-8")
            self.vocab[token_id] = token_bytes
            self.inv_vocab[token_bytes] = token_id
            self.special_to_id[s] = token_id
            self.id_to_special[token_id] = s

    def train(
        self,
        corpus: str,
        target_vocab_size: int,
        specials: Iterable[str] = DEFAULT_SPECIALS,
        min_frequency: int = 2,
    ) -> None:
        """Train BPE merges on the given text corpus until target_vocab_size is reached."""
        self.initialize(specials)
        units = self._build_initial_units(corpus)

        while self.vocab_size < target_vocab_size:
            pairs = self._count_pairs(units)
            if not pairs:
                break
            max_count = max(pairs.values())
            if max_count < min_frequency:
                break

            # Deterministic tie-breaking by sorting candidates
            candidates = sorted(p for p, c in pairs.items() if c == max_count)
            best_pair = candidates[0]

            new_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            new_id = self._add_token(new_bytes)
            self.merges[best_pair] = new_id
            units = self._apply_merge_to_corpus(units, best_pair, new_id)

    @staticmethod
    def _pretokenize(text: str) -> list[str]:
        """Split text into chunks using GPT-2 regex."""
        matches = GPT2_SPLIT_PATTERN.findall(text)
        return matches if matches else ([text] if text else [])

    @classmethod
    def _build_initial_units(cls, corpus: str) -> dict[tuple[int, ...], int]:
        chunks = cls._pretokenize(corpus)
        units: dict[tuple[int, ...], int] = {}
        for chunk in chunks:
            symbols = tuple(chunk.encode("utf-8"))
            units[symbols] = units.get(symbols, 0) + 1
        return units

    @staticmethod
    def _count_pairs(units: dict[tuple[int, ...], int]) -> Counter[tuple[int, int]]:
        pairs: Counter[tuple[int, int]] = Counter()
        for symbols, count in units.items():
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += count
        return pairs

    @staticmethod
    def _apply_merge(
        symbols: tuple[int, ...], pair: tuple[int, int], new_id: int
    ) -> tuple[int, ...]:
        if len(symbols) < 2:
            return symbols
        out: list[int] = []
        i = 0
        a, b = pair
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    @classmethod
    def _apply_merge_to_corpus(
        cls,
        units: dict[tuple[int, ...], int],
        pair: tuple[int, int],
        new_id: int,
    ) -> dict[tuple[int, ...], int]:
        new_units: dict[tuple[int, ...], int] = {}
        for symbols, count in units.items():
            merged = cls._apply_merge(symbols, pair, new_id)
            new_units[merged] = new_units.get(merged, 0) + count
        return new_units

    def _encode_chunk(self, chunk: str, ranked_merges: dict[tuple[int, int], int]) -> list[int]:
        symbols: list[int] = list(chunk.encode("utf-8"))
        if len(symbols) < 2:
            return symbols

        while True:
            best_rank: int | None = None
            best_index = -1
            best_pair: tuple[int, int] | None = None

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = ranked_merges.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_index = i
                    best_pair = pair

            if best_pair is None:
                break

            new_id = self.merges[best_pair]
            symbols = [*symbols[:best_index], new_id, *symbols[best_index + 2 :]]
        return symbols

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text into a list of token IDs."""
        ranked_merges = {pair: rank for rank, pair in enumerate(self.merges.keys())}
        cache: dict[str, list[int]] = {}

        def enc(chunk: str) -> list[int]:
            ids = cache.get(chunk)
            if ids is None:
                ids = self._encode_chunk(chunk, ranked_merges)
                cache[chunk] = ids
            return ids

        if not allow_special or not self.special_to_id:
            out: list[int] = []
            for chunk in self._pretokenize(text):
                out.extend(enc(chunk))
            return out

        specials_sorted = sorted(self.special_to_id.keys(), key=len, reverse=True)
        pattern = "(" + "|".join(re.escape(s) for s in specials_sorted) + ")"
        parts = re.split(pattern, text)

        out = []
        for part in parts:
            if not part:
                continue
            if part in self.special_to_id:
                out.append(self.special_to_id[part])
            else:
                for chunk in self._pretokenize(part):
                    out.extend(enc(chunk))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        """Decode a sequence of token IDs back into a string."""
        pieces: list[bytes] = []
        for token_id in ids:
            if token_id in self.id_to_special:
                pieces.append(self.id_to_special[token_id].encode("utf-8"))
            elif token_id in self.vocab:
                pieces.append(self.vocab[token_id])
            else:
                raise KeyError(f"Unknown token id: {token_id}")
        return b"".join(pieces).decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        """Save vocabulary and merges to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": {
                str(token_id): list(token_bytes) for token_id, token_bytes in self.vocab.items()
            },
            "merges": [[list(pair), new_id] for pair, new_id in self.merges.items()],
            "specials": self.special_to_id,
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load vocabulary and merges from a JSON file."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)

        tokenizer = cls()
        tokenizer.vocab = {int(k): bytes(v) for k, v in payload["vocab"].items()}
        tokenizer.inv_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = {
            (int(pair[0]), int(pair[1])): int(new_id) for pair, new_id in payload["merges"]
        }
        tokenizer.special_to_id = {str(k): int(v) for k, v in payload["specials"].items()}
        tokenizer.id_to_special = {int(v): str(k) for k, v in payload["specials"].items()}
        return tokenizer
