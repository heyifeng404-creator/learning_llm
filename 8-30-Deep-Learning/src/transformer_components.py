"""可复用的因果自注意力与 Transformer Decoder Block。"""

from __future__ import annotations

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_heads:
            raise ValueError("n_embd 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        if length > self.causal_mask.size(0):
            raise ValueError("序列长度超过 block_size")
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))  # [B, H, T, D]
        scores = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        scores = scores.masked_fill(~self.causal_mask[:length, :length], float("-inf"))
        weights = self.attn_dropout(scores.softmax(dim=-1))
        mixed = weights @ v
        mixed = mixed.transpose(1, 2).contiguous().view(batch, length, channels)
        output = self.resid_dropout(self.proj(mixed))
        return (output, weights) if return_weights else output


class TransformerBlock(nn.Module):
    """GPT 风格的 pre-norm decoder block。"""

    def __init__(
        self, n_embd: int, n_heads: int, block_size: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = CausalSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

