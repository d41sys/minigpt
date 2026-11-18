"""GPT model from scratch: embeddings, causal MHA, pre-LN blocks, weight-tied LM head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-layer key/value cache: {"k": (B, heads, T, head_dim), "v": same}; empty dict = nothing cached.
LayerCache = dict[str, torch.Tensor]


@dataclass
class GPTConfig:
    """Hyperparameters for GPT model architecture."""

    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int | None = None
    dropout: float = 0.0
    bias: bool = True
    use_sdpa: bool = False
    pos_encoding: str = "learned"  # "learned" (GPT-2 wpe table) or "rope" (rotary, no wpe)

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.pos_encoding not in ("learned", "rope"):
            raise ValueError(f"pos_encoding must be 'learned' or 'rope', got {self.pos_encoding!r}")
        if self.pos_encoding == "rope" and (self.d_model // self.num_heads) % 2 != 0:
            raise ValueError("rope needs an even head_dim")


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (B, heads, T, head_dim) by per-position angles (T, head_dim/2), half-split layout."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.use_sdpa = config.use_sdpa
        self.dropout_p = config.dropout

        # Combined Q, K, V projection in one matrix for efficiency
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: 1 where attention is allowed (lower triangular), 0 elsewhere
        mask = torch.tril(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool)
        )
        self.register_buffer(
            "bias",
            mask.view(1, 1, config.context_length, config.context_length),
            persistent=False,
        )

        self.rope = config.pos_encoding == "rope"
        if self.rope:
            # Angle table for absolute positions 0..context_length-1 (theta = 10000).
            inv_freq = 10000.0 ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
            angles = torch.outer(torch.arange(config.context_length).float(), inv_freq)
            self.register_buffer("rope_cos", angles.cos(), persistent=False)
            self.register_buffer("rope_sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        b, t, c = x.size()

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # The new tokens sit at absolute positions start .. start+t-1.
        start = cache["k"].size(2) if cache else 0
        if self.rope:
            rope_cos: torch.Tensor = self.rope_cos  # type: ignore[assignment]
            rope_sin: torch.Tensor = self.rope_sin  # type: ignore[assignment]
            cos, sin = rope_cos[start : start + t], rope_sin[start : start + t]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)  # k is cached rotated
        if cache is not None:
            if start:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v
        # Allowed keys for each new query: row i of the full mask, columns up to start+i.
        full_mask: torch.Tensor = self.bias  # type: ignore[assignment]
        mask = full_mask[:, :, start : start + t, : start + t]

        # MPS's fused kernel has no dropout support; use the explicit path there.
        mps_dropout = self.training and self.dropout_p > 0 and x.device.type == "mps"
        if self.use_sdpa and not mps_dropout:
            # Scaled Dot-Product Attention via PyTorch's native kernel
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None if start == 0 else mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=start == 0,
            )
        else:
            # Explicit manual causal attention
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, v)

        # Re-assemble head dimensions: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        result: torch.Tensor = self.resid_dropout(self.c_proj(out))
        return result


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        d_ff = config.d_ff if config.d_ff is not None else 4 * config.d_model
        self.c_fc = nn.Linear(config.d_model, d_ff, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c_fc(x)
        h = self.gelu(h)
        h = self.c_proj(h)
        result: torch.Tensor = self.dropout(h)
        return result


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """Full GPT Language Model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        if config.pos_encoding == "learned":
            self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)

        # Output LM Head (without bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output projection
        self.lm_head.weight = self.wte.weight

        # Parameter initialization
        self.apply(self._init_weights)

        # Special scaled init for residual projections (GPT-2 paper section 2.3)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden_states(
        self, idx: torch.Tensor, cache: list[LayerCache] | None = None
    ) -> torch.Tensor:
        """Embeddings, blocks and final LayerNorm: (B, T) token IDs -> (B, T, d_model)."""
        device = idx.device
        _b, t = idx.size()
        start = cache[0]["k"].size(2) if cache and cache[0] else 0
        if start + t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {start + t}, "
                f"block size is {self.config.context_length}"
            )

        x = self.wte(idx)
        if self.config.pos_encoding == "learned":
            x = x + self.wpe(torch.arange(start, start + t, dtype=torch.long, device=device))
        x = self.drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x, cache[i] if cache is not None else None)

        out: torch.Tensor = self.ln_f(x)
        return out

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[LayerCache] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: Tensor of token IDs with shape (B, T)
            targets: Optional ground-truth token IDs with shape (B, T)
            cache: Optional per-layer KV cache (see `generate`). `idx` then holds only the
                new tokens, positioned right after the cached ones; the cache is extended in place.

        Returns:
            Tuple of (logits, loss). If targets is None, loss is None.
        """
        x = self.hidden_states(idx, cache)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively generate next tokens given a conditioning prompt.

        With `use_cache` (default) each layer keeps the keys and values of every token seen so
        far. The prompt is processed once; each later step feeds only the newest token and its
        single query attends over the cached keys, so a step costs O(T) instead of the O(T^2)
        of re-running the whole prefix. Once the sequence outgrows `context_length` the window
        slides, every token's position changes, and the cache is stale: those steps crop to the
        last `context_length` tokens and recompute, exactly like `use_cache=False`.
        """
        self.eval()
        cache: list[LayerCache] = [{} for _ in self.blocks]
        for _ in range(max_new_tokens):
            if idx.size(1) > self.config.context_length:
                # Crop to context length if sequence length exceeds capacity
                logits, _ = self(idx[:, -self.config.context_length :])
            elif use_cache:
                # Empty cache: whole prompt; otherwise just the newest token.
                logits, _ = self(idx[:, -1:] if cache[0] else idx, cache=cache)
            else:
                logits, _ = self(idx)
            # Pick logits at the last position
            logits = logits[:, -1, :]

            if temperature <= 0.0:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_token), dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx


def count_parameters(model: nn.Module, tied: bool = True) -> int:
    """Count the total number of trainable parameters.

    If tied=True, parameters that share memory storage (e.g. wte and lm_head)
    are counted only once.
    """
    if tied:
        unique_params = set(model.parameters())
        return sum(p.numel() for p in unique_params)
    return sum(p.numel() for p in model.parameters())
