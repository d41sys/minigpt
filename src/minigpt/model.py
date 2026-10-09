"""GPT model from scratch: embeddings, causal MHA, pre-LN blocks, weight-tied LM head."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        # MPS's fused kernel has no dropout support; use the explicit path there.
        mps_dropout = self.training and self.dropout_p > 0 and x.device.type == "mps"
        if self.use_sdpa and not mps_dropout:
            # Scaled Dot-Product Attention via PyTorch's native kernel
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Explicit manual causal attention
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask: torch.Tensor = self.bias  # type: ignore[assignment]
            scores = scores.masked_fill(~mask[:, :, :t, :t], float("-inf"))
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
        self.ln_1 = nn.LayerNorm(config.d_model, eps=1e-5, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=1e-5, elementwise_affine=config.bias)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPTModel(nn.Module):
    """Full GPT Language Model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5, elementwise_affine=config.bias)

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

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            idx: Tensor of token IDs with shape (B, T)
            targets: Optional ground-truth token IDs with shape (B, T)

        Returns:
            Tuple of (logits, loss). If targets is None, loss is None.
        """
        device = idx.device
        _b, t = idx.size()
        if t > self.config.context_length:
            raise ValueError(
                f"Cannot forward sequence of length {t}, block size is {self.config.context_length}"
            )

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
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
    ) -> torch.Tensor:
        """Autoregressively generate next tokens given a conditioning prompt."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to context length if sequence length exceeds capacity
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length :]
            )
            logits, _ = self(idx_cond)
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
