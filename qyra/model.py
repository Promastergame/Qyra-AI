"""
model.py — Modern Qyra with RoPE, GQA, SwiGLU, RMSNorm.

Architecture improvements (v2):
  • RoPE (Rotary Position Embeddings) — no learned positional embeddings
  • GQA (Grouped Query Attention) — shared KV heads for memory efficiency
  • SwiGLU — gated feed-forward network
  • RMSNorm — faster normalization without mean/bias
  • QK-Norm — normalize Q and K before attention for training stability
  • SDPA — uses PyTorch's scaled_dot_product_attention (Flash/Memory-efficient)
  • KV Cache — efficient autoregressive generation
  • Gradient Checkpointing support — reduced activation memory
  • Embedding scaling — scale embeddings by sqrt(d_model) like modern LLMs
  • Improved weight initialization — muP-inspired scaling
"""

import math
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from typing import Optional, Union, Tuple, Set, Any
from config import ModelConfig  # type: ignore


# ═══════════════════════════════════════════════════════════════════════
#  RMSNorm — Root Mean Square Layer Normalization
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    RMSNorm normalizes by root mean square instead of mean+std.
    Simpler, faster, and used by LLaMA, Mistral, Gemma.
    Uses rsqrt for speed.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use rsqrt for speed (1/sqrt instead of sqrt + divide)
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x.float() * norm).to(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════════
#  RoPE — Rotary Position Embeddings
# ═══════════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding encodes position via rotation.
    No learned parameters, generalizes to unseen sequence lengths.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Frequency bands: θ_i = 1 / (theta^(2i/d))
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos/sin for positions 0..seq_len-1."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim//2)
        freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Apply RoPE to input tensor.
        x: (batch, n_heads, seq_len, head_dim)
        """
        seq_len = x.shape[2]

        # Extend cache if needed
        if start_pos + seq_len > self.max_seq_len:
            self._build_cache(start_pos + seq_len + 64)  # +64 to avoid frequent rebuilds

        cos = self.cos_cached[start_pos : start_pos + seq_len]
        sin = self.sin_cached[start_pos : start_pos + seq_len]
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_rotated = self._rotate_half(x)
        return (x * cos) + (x_rotated * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate second half to first half, negated."""
        d = x.shape[-1]
        x1 = x[..., : d // 2]
        x2 = x[..., d // 2 :]
        return torch.cat([-x2, x1], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
#  SwiGLU — Gated Feed-Forward Network
# ═══════════════════════════════════════════════════════════════════════

class SwiGLU(nn.Module):
    """
    SwiGLU uses gating for better expressivity.
    Three matrices at 2/3 width ≈ two matrices at full width.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        # Match parameter count: d_ff_swiglu = 2/3 * d_ff
        d_ff_swiglu = int(2 * cfg.d_ff / 3)
        d_ff_swiglu = ((d_ff_swiglu + 63) // 64) * 64  # align for efficiency

        self.w_gate = nn.Linear(cfg.d_model, d_ff_swiglu, bias=False)
        self.w_up   = nn.Linear(cfg.d_model, d_ff_swiglu, bias=False)
        self.w_down = nn.Linear(d_ff_swiglu, cfg.d_model, bias=False)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.drop(self.w_down(gate * up))


# ═══════════════════════════════════════════════════════════════════════
#  GQA — Grouped Query Attention with QK-Norm and SDPA
# ═══════════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    GQA shares KV heads across query head groups.
    Reduces KV cache size and parameters.
    
    Improvements:
      - QK-Norm: normalize Q and K before attention for stability (Gemma, Gemini)
      - SDPA: uses PyTorch's scaled_dot_product_attention for speed
      - Supports sliding window attention (TODO)
    """
    def __init__(self, cfg: ModelConfig, rotary_emb: RotaryEmbedding = None):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = getattr(cfg, 'n_kv_heads', cfg.n_heads)
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_groups = self.n_heads // self.n_kv_heads
        self.d_model = cfg.d_model

        assert cfg.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({cfg.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.rotary_emb = rotary_emb
        self.dropout_p = cfg.dropout

        # QK-Norm for training stability (used by Gemma, Gemini)
        self.use_qk_norm = getattr(cfg, 'use_qk_norm', True)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=getattr(cfg, 'norm_eps', 1e-6))
            self.k_norm = RMSNorm(self.head_dim, eps=getattr(cfg, 'norm_eps', 1e-6))

    def forward(self, x: torch.Tensor, kv_cache: Optional[tuple] = None, start_pos: int = 0):
        B, T, C = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE
        if self.rotary_emb is not None:
            q = self.rotary_emb(q, start_pos=start_pos)
            k = self.rotary_emb(k, start_pos=start_pos)

        # KV cache for generation
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match Q heads (for GQA)
        if self.n_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            k = k.reshape(B, self.n_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            v = v.reshape(B, self.n_heads, -1, self.head_dim)

        # Use PyTorch's SDPA for speed (Flash Attention on GPU, efficient on CPU)
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=(T > 1 and kv_cache is None),  # causal only for training/prefill
        )

        # For cached generation with T>1, we need manual causal mask
        if T > 1 and kv_cache is not None:
            kv_len = k.shape[2]
            causal_mask = torch.tril(
                torch.ones(T, kv_len, device=x.device, dtype=torch.bool),
                diagonal=kv_len - T
            )
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=dropout_p,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.out_proj(out))

        return out, new_kv_cache


# ═══════════════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCK
# ═══════════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Pre-norm block with GQA + SwiGLU + RMSNorm."""

    def __init__(self, cfg: ModelConfig, layer_idx: int, rotary_emb: RotaryEmbedding = None):
        super().__init__()
        self.layer_idx = layer_idx

        # RMSNorm (or fallback to LayerNorm)
        use_rms = getattr(cfg, 'use_rmsnorm', True)
        norm_eps = getattr(cfg, 'norm_eps', 1e-6)
        Norm = RMSNorm if use_rms else nn.LayerNorm
        norm_kwargs = {"eps": norm_eps}

        self.ln1 = Norm(cfg.d_model, **norm_kwargs)
        self.ln2 = Norm(cfg.d_model, **norm_kwargs)

        # Attention with GQA + RoPE + QK-Norm
        self.attn = GroupedQueryAttention(cfg, rotary_emb=rotary_emb)

        # Feed-forward (SwiGLU or standard MLP)
        use_swiglu = getattr(cfg, 'use_swiglu', True)
        self.mlp = SwiGLU(cfg) if use_swiglu else MLP(cfg)

    def forward(self, x: torch.Tensor, kv_cache: Optional[tuple] = None, start_pos: int = 0):
        # Pre-norm + residual
        attn_out, new_kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_kv_cache


# ═══════════════════════════════════════════════════════════════════════
#  LEGACY MLP (for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """Standard two-layer MLP with GELU."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x


# ═══════════════════════════════════════════════════════════════════════
#  QYRA MODEL
# ═══════════════════════════════════════════════════════════════════════

class Qyra(nn.Module):
    """
    Modern GPT with RoPE, GQA, SwiGLU, RMSNorm, QK-Norm.
    Supports gradient checkpointing and KV cache.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Token embedding (no positional embedding if using RoPE)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_scale = math.sqrt(cfg.d_model)  # Scale embeddings like modern LLMs
        self.drop = nn.Dropout(cfg.dropout)

        # Learned positional embedding (only if NOT using RoPE)
        use_rope = getattr(cfg, 'use_rope', True)
        if not use_rope:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        else:
            self.pos_emb = None

        # Shared rotary embedding
        head_dim = cfg.d_model // cfg.n_heads
        max_len = cfg.max_seq_len * 2 if use_rope else cfg.max_seq_len
        rotary = RotaryEmbedding(head_dim, max_len, getattr(cfg, 'rope_theta', 10000.0)) if use_rope else None

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, layer_idx=i, rotary_emb=rotary)
            for i in range(cfg.n_layers)
        ])

        # Final norm
        use_rms = getattr(cfg, 'use_rmsnorm', True)
        norm_eps = getattr(cfg, 'norm_eps', 1e-6)
        Norm = RMSNorm if use_rms else nn.LayerNorm
        self.ln_f = Norm(cfg.d_model, eps=norm_eps)

        # LM head
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        self._init_residual_scaling()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_scaling(self):
        """Scale residual projections for stable training."""
        scale = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scale)
            if hasattr(block.mlp, 'w_down'):
                nn.init.normal_(block.mlp.w_down.weight, mean=0.0, std=scale)
            elif hasattr(block.mlp, 'fc2'):
                nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=scale)

    def count_parameters(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            pid = p.data_ptr()
            if pid not in seen:
                seen.add(pid)
                total += int(p.numel())  # type: ignore
        return total

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None,
                use_checkpoint: bool = False, label_smoothing: float = 0.0):
        B, T = input_ids.shape

        # Embedding with scaling (Gemma style: embed * sqrt(d_model))
        x = self.tok_emb(input_ids) * self.emb_scale

        # Add positional embedding only if not using RoPE
        if self.pos_emb is not None:
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            if use_checkpoint and self.training:
                # Gradient checkpointing: recompute forward during backward
                x, _ = torch.utils.checkpoint.checkpoint(
                    block, x, None, 0, use_reentrant=False
                )
            else:
                x, _ = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=label_smoothing,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        stop_token_ids: Optional[set] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with KV cache and multiple stop conditions.
        """
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device

        # Build combined stop set
        stop_ids = set()
        if eos_token_id is not None:
            stop_ids.add(eos_token_id)
        if stop_token_ids is not None:
            stop_ids.update(stop_token_ids)

        # ── Prefill: process entire prompt ──
        kv_caches = [None] * self.cfg.n_layers

        x = self.tok_emb(input_ids) * self.emb_scale
        if self.pos_emb is not None:
            pos = torch.arange(T, device=device).unsqueeze(0)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        for i, block in enumerate(self.blocks):
            x, kv_caches[i] = block(x, kv_cache=None, start_pos=0)

        x = self.ln_f(x)  # type: ignore
        logits = self.lm_head(x[:, -1:, :])

        start_pos = T

        # ── Decode: one token at a time with KV cache ──
        for step in range(max_new_tokens):
            next_logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(input_ids[0].tolist()):
                    if next_logits[0, token_id] > 0:
                        next_logits[0, token_id] /= repetition_penalty
                    else:
                        next_logits[0, token_id] *= repetition_penalty

            # Temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Sampling
            if temperature == 0:
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)

                if top_p is not None and float(top_p) < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs > top_p
                    sorted_probs[mask] = 0
                    sorted_probs = sorted_probs / sorted_probs.sum()
                    next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
                elif top_k is not None and int(top_k) > 0:
                    v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                    probs[probs < v[:, [-1]]] = 0
                    probs = probs / probs.sum()
                    next_id = torch.multinomial(probs, 1)
                else:
                    next_id = torch.multinomial(probs, 1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

            # Check stop condition
            if next_id.item() in stop_ids:
                break

            # Process new token through model with KV cache
            x = self.tok_emb(next_id) * self.emb_scale
            if self.pos_emb is not None:
                p = torch.tensor([[start_pos + step]], device=device)
                x = x + self.pos_emb(p)
            for j, block in enumerate(self.blocks):
                x, kv_caches[j] = block(
                    x, kv_cache=kv_caches[j], start_pos=start_pos + step
                )
            x = self.ln_f(x)  # type: ignore
            logits = self.lm_head(x)

        return input_ids
