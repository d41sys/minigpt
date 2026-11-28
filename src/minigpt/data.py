"""Dataset utilities and sliding-window token chunking for language modeling."""

from __future__ import annotations

import json
import multiprocessing
import time
import urllib.parse
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from minigpt.bpe import BPETokenizer


class SlidingWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset for next-token prediction.

    Extracts overlapping or non-overlapping windows of length `context_length`
    from a 1D sequence of token IDs.
    Each item is a tuple `(x, y)` where:
    - `x` is the input token sequence of length `context_length`
    - `y` is the target token sequence of length `context_length`, shifted by 1.
    """

    def __init__(
        self,
        ids: Sequence[int] | torch.Tensor,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        if isinstance(ids, torch.Tensor):
            self.ids = ids.to(dtype=torch.long).cpu()
        else:
            self.ids = torch.tensor(ids, dtype=torch.long)

        if context_length < 1:
            raise ValueError(f"context_length must be positive, got {context_length}")
        self.context_length = context_length
        self.stride = stride if stride is not None and stride > 0 else context_length

        if len(self.ids) <= self.context_length:
            self.num_samples = 0
        else:
            # We need context_length + 1 tokens to construct input x and target y
            total_tokens = len(self.ids)
            self.num_samples = max(0, (total_tokens - self.context_length - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        start = idx * self.stride
        x = self.ids[start : start + self.context_length]
        y = self.ids[start + 1 : start + self.context_length + 1]
        return x, y


def split_val_test(ids: Sequence[int]) -> tuple[list[int], list[int]]:
    """Deterministically cut validation tokens into contiguous val (first half) and test halves."""
    half = len(ids) // 2
    return list(ids[:half]), list(ids[half:])


def create_dataloader(
    ids: Sequence[int] | torch.Tensor,
    context_length: int,
    batch_size: int = 4,
    stride: int | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Create a DataLoader over a SlidingWindowDataset."""
    dataset = SlidingWindowDataset(ids, context_length=context_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last and len(dataset) >= batch_size,
        num_workers=num_workers,
    )


class MixedLoader:
    """Training batches with `round(frac * batch_size)` windows from `replay_ids`, the rest from
    `ids`. One pass over `ids` is one epoch.

    shortcut: an epoch ends when the shorter source runs out, so `replay_ids` should be the larger
    corpus; upgrade if replay is ever smaller than the domain data.
    """

    def __init__(
        self,
        ids: Sequence[int],
        replay_ids: Sequence[int],
        context_length: int,
        batch_size: int,
        frac: float,
        stride: int | None = None,
    ) -> None:
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"replay fraction must be in [0, 1], got {frac}")
        n_replay = round(frac * batch_size)
        self.loaders = [
            create_dataloader(src, context_length, n, stride=stride)
            for src, n in ((ids, batch_size - n_replay), (replay_ids, n_replay))
            if n > 0
        ]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for parts in zip(*self.loaders, strict=False):
            yield torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


def hf_rows(dataset: str, config: str, split: str, n: int) -> Iterator[dict[str, Any]]:
    """The first `n` rows of a Hugging Face dataset split (fewer if it ends), 100 per request,
    from the datasets-server JSON API. Raises after 5 failed attempts at one request."""
    for offset in range(0, n, 100):
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": min(100, n - offset),
            }
        )
        for attempt in range(5):
            try:
                req = urllib.request.Request(
                    f"https://datasets-server.huggingface.co/rows?{query}",
                    headers={"User-Agent": "minigpt"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    rows = json.loads(resp.read())["rows"]
                break
            except OSError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        yield from (r["row"] for r in rows)
        if len(rows) < min(100, n - offset):
            return


_worker_tokenizer: BPETokenizer | None = None


def _init_worker(tokenizer: BPETokenizer) -> None:
    global _worker_tokenizer
    _worker_tokenizer = tokenizer


def _encode_piece(piece: str) -> npt.NDArray[np.uint16]:
    assert _worker_tokenizer is not None
    return np.asarray(_worker_tokenizer.encode(piece, allow_special=True), dtype=np.uint16)


def tokenize_cached(
    tokenizer: BPETokenizer,
    text_path: Path,
    cache_path: Path,
    workers: int = 4,
    piece_chars: int = 4_000_000,
) -> npt.NDArray[np.uint16]:
    """Token IDs (uint16) of a text file of `<|endoftext|>`-ended documents, cached as .npy.

    The pure-Python encoder is slow, so ~`piece_chars` pieces cut at document ends are encoded
    in `workers` processes; the result equals `tokenizer.encode(text, allow_special=True)`.
    The cache is keyed by file name only: delete it when the tokenizer changes.
    """
    if cache_path.exists():
        cached: npt.NDArray[np.uint16] = np.load(cache_path)
        return cached
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("uint16 token cache needs vocab_size <= 65536")
    eot = "<|endoftext|>"
    docs = text_path.read_text(encoding="utf-8").split(eot)
    if docs[-1] == "":
        docs.pop()
    pieces: list[str] = []
    group: list[str] = []
    size = 0
    for doc in docs:
        group.append(doc)
        size += len(doc)
        if size >= piece_chars:
            pieces.append(eot.join(group) + eot)
            group, size = [], 0
    if group:
        pieces.append(eot.join(group) + eot)
    with multiprocessing.Pool(workers, _init_worker, (tokenizer,)) as pool:
        ids: npt.NDArray[np.uint16] = np.concatenate(pool.map(_encode_piece, pieces, chunksize=1))
    tmp = cache_path.with_suffix(".tmp.npy")
    np.save(tmp, ids)
    tmp.replace(cache_path)
    return ids
