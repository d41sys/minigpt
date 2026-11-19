"""GPT model from scratch: embeddings, causal MHA, pre-LN blocks, weight-tied LM head."""

from __future__ import annotations

import math
from collections.abc import Callable
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

    # DeepSeek-V4 options (toy versions), all off by default so old checkpoints load unchanged.
    n_experts: int = 0  # routed experts per MoE FFN; 0 = dense FFN (report section 2.1)
    n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_d_ff: int | None = None  # default d_ff // (moe_top_k + n_shared_experts)
    moe_dense_first: bool = False  # keep layer 0 dense
    moe_bias_rate: float = 0.01  # gamma of the aux-loss-free bias update
    moe_balance_alpha: float = 1e-3  # weight of the sequence-wise balance loss
    mtp_depth: int = 0  # extra multi-token-prediction modules (report section 2.1)
    mtp_weight: float = 0.3
    hc_streams: int = 1  # n residual streams of mHC; 1 = plain residual (report section 2.2)
    hc_sinkhorn_iters: int = 20
    attn_type: str = "dense"  # "dense" or "csa" (report section 2.3.1)
    csa_block: int = 4  # m: tokens per compressed entry
    csa_topk: int = 4  # compressed entries kept per query
    csa_window: int = 8  # raw-token sliding window
    csa_index_heads: int = 2
    csa_index_dim: int = 16

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
        if self.attn_type not in ("dense", "csa"):
            raise ValueError(f"attn_type must be 'dense' or 'csa', got {self.attn_type!r}")
        if self.hc_streams < 1 or self.mtp_depth < 0 or self.csa_block < 1:
            raise ValueError("need hc_streams >= 1, mtp_depth >= 0, csa_block >= 1")
        if self.n_experts:
            if not 1 <= self.moe_top_k <= self.n_experts:
                raise ValueError(f"moe_top_k must be in [1, n_experts={self.n_experts}]")
            if self.moe_d_ff is None:
                self.moe_d_ff = max(1, self.d_ff // (self.moe_top_k + self.n_shared_experts))


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


class Compressor(nn.Module):
    """Pool every `m` tokens into one entry with learned softmax weights plus a positional bias.

    DeepSeek-V4 report eqs. 9-12, simplified to one series (no overlapping a/b halves); a
    trailing partial block is dropped.
    """

    def __init__(self, d_model: int, dim: int, m: int) -> None:
        super().__init__()
        self.kv = nn.Linear(d_model, dim, bias=False)
        self.z = nn.Linear(d_model, dim, bias=False)
        self.pos = nn.Parameter(torch.zeros(m * dim))  # (m, dim) bias, kept 1-D: not a matrix
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        blocks = x[:, : t // self.m * self.m].reshape(b, -1, self.m, d)
        bias = self.pos.view(self.m, -1)
        weights = torch.softmax(self.z(blocks) + bias, dim=2)
        return (weights * self.kv(blocks)).sum(dim=2)  # (B, T // m, dim)


class CompressedSparseAttention(CausalSelfAttention):
    """Compressed sparse attention, simplified (DeepSeek-V4 report section 2.3.1).

    Every `m` tokens are pooled into one shared key/value entry (MQA style). A lightning indexer
    (report eq. 16: sum_h w_h * ReLU(q_h . k)) scores the compressed entries of earlier blocks
    and keeps the top-k per query; the query attends to those plus a sliding window of raw
    tokens. Entry s is visible to query t only if s < t // m, so every entry is built from past
    tokens and attention stays causal. Differences from the report: no overlapping blocks, no
    HCA, no FP4, a direct (not low-rank) indexer query, no partial RoPE on compressed entries,
    and the index score is added to the selected entries' logits so the LM loss trains the
    indexer (the report trains it with a separate distillation loss).

    At our ~100-256 token contexts this is about quality parity, not speed: the mask is dense.
    It keeps no KV cache; `GPTModel.generate` falls back to full recompute for it.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        d, m = config.d_model, config.csa_block
        self.m, self.topk, self.window = m, config.csa_topk, config.csa_window
        self.n_index_heads, self.index_dim = config.csa_index_heads, config.csa_index_dim
        self.comp = Compressor(d, self.head_dim, m)
        self.index_comp = Compressor(d, self.index_dim, m)
        self.index_q = nn.Linear(d, self.n_index_heads * self.index_dim, bias=False)
        self.index_w = nn.Linear(d, self.n_index_heads, bias=False)

    def index_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Indexer score of every compressed entry for every query: (B, T, T // m)."""
        b, t, _ = x.size()
        q = self.index_q(x).view(b, t, self.n_index_heads, self.index_dim)
        hits = F.relu(torch.einsum("bthc,bnc->bthn", q, self.index_comp(x)))
        return torch.einsum("bth,bthn->btn", self.index_w(x), hits)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        if cache is not None:
            raise ValueError("CompressedSparseAttention has no KV cache; recompute instead")
        b, t, c = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, self.num_heads, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q_comp = q  # compressed entries carry no position, so score them with the unrotated q
        if self.rope:
            rope_cos: torch.Tensor = self.rope_cos  # type: ignore[assignment]
            rope_sin: torch.Tensor = self.rope_sin  # type: ignore[assignment]
            cos, sin = rope_cos[:t], rope_sin[:t]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)

        pos = torch.arange(t, device=x.device)
        near = (pos[:, None] >= pos[None, :]) & (pos[:, None] - pos[None, :] < self.window)
        scores = (q @ k.transpose(-2, -1) * scale).masked_fill(~near, float("-inf"))

        n_comp = t // self.m
        if n_comp:
            comp = self.comp(x)  # (B, n_comp, head_dim), used as both key and value
            allowed = torch.arange(n_comp, device=x.device)[None, :] < (pos // self.m)[:, None]
            index = self.index_scores(x).masked_fill(~allowed, float("-inf"))
            top = index.topk(min(self.topk, n_comp), dim=-1).indices
            chosen = torch.zeros_like(index, dtype=torch.bool).scatter(-1, top, True) & allowed
            comp_scores = q_comp @ comp.transpose(-2, -1).unsqueeze(1) * scale
            comp_scores = comp_scores + index.masked_fill(~chosen, 0.0).unsqueeze(1)
            comp_scores = comp_scores.masked_fill(~chosen.unsqueeze(1), float("-inf"))
            scores = torch.cat((scores, comp_scores), dim=-1)

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights[..., :t] @ v
        if n_comp:
            out = out + weights[..., t:] @ comp.unsqueeze(1)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        result: torch.Tensor = self.resid_dropout(self.c_proj(out))
        return result


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""

    def __init__(self, config: GPTConfig, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or config.d_ff or 4 * config.d_model
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


class MoEFeedForward(nn.Module):
    """DeepSeekMoE FFN (DeepSeek-V4 report section 2.1): shared experts plus routed experts.

    Affinity is sqrt(softplus(logit)); each token takes the top-k routed experts by
    affinity + `route_bias` and mixes them with the normalized affinities. Aux-loss-free
    balancing: in training, `route_bias` (a buffer, used only for selection, never in the
    gradient) moves by gamma * sign(mean load - load) per expert. A small sequence-wise
    balance loss is left in `balance_loss` for `GPTModel` to add. Experts are looped over,
    which is fine for a handful of toy experts.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.top_k, self.gamma = config.moe_top_k, config.moe_bias_rate
        self.alpha = config.moe_balance_alpha
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(config, config.moe_d_ff) for _ in range(config.n_experts)
        )
        self.shared = nn.ModuleList(
            FeedForward(config, config.moe_d_ff) for _ in range(config.n_shared_experts)
        )
        self.register_buffer("route_bias", torch.zeros(config.n_experts))
        self.balance_loss = torch.zeros(())
        self.last_load = torch.zeros(config.n_experts)  # tokens per expert in the last forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.size()
        flat = x.reshape(-1, d)
        n_experts = len(self.experts)
        affinity = F.softplus(self.router(flat).float()).sqrt()  # (N, E)
        top = (affinity + self.route_bias).topk(self.top_k, dim=-1).indices  # (N, k)
        gate = affinity.gather(1, top)
        gate = gate / gate.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat)
        for expert in self.shared:
            out = out + expert(flat)
        for i, expert in enumerate(self.experts):
            rows, slot = (top == i).nonzero(as_tuple=True)
            if rows.numel():
                y = expert(flat[rows]) * gate[rows, slot, None]
                out = out.index_add(0, rows, y.to(out.dtype))

        if self.training:
            chosen = F.one_hot(top, n_experts).sum(dim=1)  # (N, E) 1 where selected
            load = chosen.sum(dim=0).float()
            self.last_load = load
            with torch.no_grad():
                self.route_bias.add_(self.gamma * torch.sign(load.mean() - load))
            frac = chosen.view(b, t, -1).float().mean(dim=1) * n_experts / self.top_k  # (B, E)
            prob = (affinity / affinity.sum(dim=-1, keepdim=True)).view(b, t, -1).mean(dim=1)
            self.balance_loss = self.alpha * (frac * prob).sum(dim=-1).mean()
        return out.view(b, t, d)


def sinkhorn(logits: torch.Tensor, iters: int) -> torch.Tensor:
    """Project (..., n, n) logits to doubly stochastic matrices: exp, then alternately
    normalize columns and rows (DeepSeek-V4 report eq. 8). Rows sum to 1 exactly; columns
    converge as `iters` grows."""
    logits = logits.float()
    m = (logits - logits.amax(dim=(-2, -1), keepdim=True)).exp()  # shift is a no-op after norm
    for _ in range(iters):
        m = m / m.sum(dim=-2, keepdim=True)
        m = m / m.sum(dim=-1, keepdim=True)
    return m


class HyperConnection(nn.Module):
    """Manifold-constrained hyper-connection (DeepSeek-V4 report section 2.2).

    The residual is n streams X (B, T, n, d). One layer F updates it as
    X' = B X + C F(A X): A = sigmoid and C = 2 sigmoid are non-negative and bounded; B is
    doubly stochastic (Sinkhorn), so mixing the streams cannot amplify them. A, B, C are
    a static bias plus a small, gated, input-dependent term (eqs. 3-7). Initialized so that
    equal streams stay equal and F's output is added to each stream, like a plain residual.
    """

    def __init__(self, d_model: int, n: int, iters: int) -> None:
        super().__init__()
        self.n, self.iters = n, iters
        self.dyn = nn.Linear(n * d_model, n * n + 2 * n, bias=False)  # W_res, W_pre, W_post
        self.alpha = nn.Parameter(torch.full((3,), 0.01))  # gates of res, pre, post
        self.s_res = nn.Parameter(2.0 * torch.eye(n).flatten())  # near-identity mixing
        self.s_pre = nn.Parameter(torch.full((n,), math.log(1.0 / max(n - 1, 1e-9))))  # A = 1/n
        self.s_post = nn.Parameter(torch.zeros(n))  # C = 1

    def maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A (B,T,n), B (B,T,n,n) doubly stochastic, C (B,T,n) for X (B,T,n,d)."""
        flat = x.flatten(2)
        flat = flat * torch.rsqrt(flat.pow(2).mean(dim=-1, keepdim=True) + 1e-6)  # RMSNorm
        res, pre, post = self.dyn(flat).split((self.n * self.n, self.n, self.n), dim=-1)
        a = torch.sigmoid(self.alpha[1] * pre + self.s_pre)
        c = 2.0 * torch.sigmoid(self.alpha[2] * post + self.s_post)
        b = sinkhorn((self.alpha[0] * res + self.s_res).unflatten(-1, (self.n, self.n)), self.iters)
        return a, b, c

    def forward(
        self, x: torch.Tensor, layer: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        a, b, c = self.maps(x)
        y = layer((a.unsqueeze(-1) * x).sum(dim=2))
        return b.to(x.dtype) @ x + c.unsqueeze(-1) * y.unsqueeze(2)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block; `moe` swaps the FFN, `hc` the residual for mHC."""

    def __init__(self, config: GPTConfig, moe: bool = False, hc: bool = False) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)
        self.attn = (
            CompressedSparseAttention(config)
            if config.attn_type == "csa"
            else CausalSelfAttention(config)
        )
        self.ln_2 = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)
        self.mlp = MoEFeedForward(config) if moe else FeedForward(config)
        self.hc_attn: HyperConnection | None = None
        self.hc_mlp: HyperConnection | None = None
        if hc:  # one hyper-connection per sublayer (attention, FFN)
            n, iters = config.hc_streams, config.hc_sinkhorn_iters
            self.hc_attn = HyperConnection(config.d_model, n, iters)
            self.hc_mlp = HyperConnection(config.d_model, n, iters)

    def forward(self, x: torch.Tensor, cache: LayerCache | None = None) -> torch.Tensor:
        if self.hc_attn is not None and self.hc_mlp is not None:
            x = self.hc_attn(x, lambda h: self.attn(self.ln_1(h), cache))
            return self.hc_mlp(x, lambda h: self.mlp(self.ln_2(h)))
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class MTPModule(nn.Module):
    """One multi-token-prediction step (DeepSeek-V4 report section 2.1, as in DeepSeek-V3).

    Merges the previous depth's hidden state at position t with the embedding of the token at
    t+d, runs one dense block, and returns the hidden state used (with the shared final norm
    and LM head) to predict token t+d+1.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_h = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)
        self.ln_e = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)
        self.proj = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.block = TransformerBlock(config)

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.block(self.proj(torch.cat((self.ln_h(h), self.ln_e(emb)), -1)))
        return out


class GPTModel(nn.Module):
    """Full GPT Language Model."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        if config.pos_encoding == "learned":
            self.wpe = nn.Embedding(config.context_length, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(
                config,
                moe=config.n_experts > 0 and not (config.moe_dense_first and i == 0),
                hc=config.hc_streams > 1,
            )
            for i in range(config.num_layers)
        )
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-5, bias=config.bias)

        # Output LM Head (without bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output projection
        self.lm_head.weight = self.wte.weight

        # Multi-token-prediction modules share wte / ln_f / lm_head; training-only.
        self.mtp = nn.ModuleList(MTPModule(config) for _ in range(config.mtp_depth))
        self.aux_losses: dict[str, torch.Tensor] = {}  # weighted extra losses of the last step

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
        if self.config.hc_streams > 1:
            x = x.unsqueeze(2).expand(-1, -1, self.config.hc_streams, -1)

        for i, block in enumerate(self.blocks):
            x = block(x, cache[i] if cache is not None else None)

        if self.config.hc_streams > 1:
            x = x.mean(dim=2)
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
            if self.training and (self.mtp or self.config.n_experts):
                aux = self._aux_losses(idx, x, targets)
                self.aux_losses = {name: v.detach() for name, v in aux.items()}
                for v in aux.values():
                    loss = loss + v

        return logits, loss

    def _aux_losses(
        self, idx: torch.Tensor, h: torch.Tensor, targets: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Training-only terms already weighted: MTP cross-entropy and MoE balance loss.

        MTP module d (1-based) sees the hidden state at t and the token at t+d, so its targets
        are `targets[:, d:]`, the token at t+d+1; positions without a target are dropped.
        """
        aux: dict[str, torch.Tensor] = {}
        ces = []
        for d, module in enumerate(self.mtp, start=1):
            if idx.size(1) <= d:
                break
            h = module(h[:, :-1], self.wte(idx[:, d:]))
            logits = self.lm_head(self.ln_f(h))
            target = targets[:, d:].reshape(-1)
            ces.append(
                F.cross_entropy(logits.reshape(-1, logits.size(-1)), target, ignore_index=-100)
            )
        if ces:
            aux["mtp"] = self.config.mtp_weight * torch.stack(ces).mean()
        balance = [b.mlp.balance_loss for b in self.blocks if isinstance(b.mlp, MoEFeedForward)]
        if balance:
            aux["balance"] = torch.stack(balance).sum()
        return aux

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
        last context_length tokens and recompute, exactly like `use_cache=False`. Compressed
        sparse attention (`attn_type="csa"`) keeps no cache, so it always recomputes. MoE and
        mHC act per token and cache fine; MTP is training-only and unused here.
        """
        self.eval()
        use_cache = use_cache and self.config.attn_type == "dense"
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


def count_parameters(model: nn.Module, tied: bool = True, active: bool = False) -> int:
    """Count the total number of trainable parameters.

    If tied=True, parameters that share memory storage (e.g. wte and lm_head)
    are counted only once. With active=True, count what one token uses: every routed
    expert of a MoE layer beyond its top-k is left out (MTP modules still count).
    """
    total = (
        sum(p.numel() for p in set(model.parameters()))
        if tied
        else sum(p.numel() for p in model.parameters())
    )
    if not active:
        return total
    idle = 0
    for m in model.modules():
        if isinstance(m, MoEFeedForward):
            routed = sum(p.numel() for p in m.experts.parameters())
            idle += routed * (len(m.experts) - m.top_k) // len(m.experts)
    return total - idle
