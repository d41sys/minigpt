"""Dataset utilities and sliding-window token chunking for language modeling."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset


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
