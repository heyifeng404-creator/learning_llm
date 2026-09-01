"""教学用迷你 GPT：模型定义与采样函数。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from transformer_components import TransformerBlock


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 24
    n_embd: int = 48
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """从最后一维的 logits 采样，返回 `[..., 1]` token id。"""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    if temperature < 0:
        raise ValueError("temperature 不能为负")
    logits = logits / temperature
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 内")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(-1, sorted_indices, sorted_logits)
    probabilities = logits.softmax(dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    config.n_embd,
                    config.n_heads,
                    config.block_size,
                    config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 权重绑定：输入和输出共享 token 表示参数。
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = token_ids.shape
        if length > self.config.block_size:
            raise ValueError("输入超过模型的 block_size")
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = token_ids[:, -self.config.block_size :]
            logits, _ = self(context)
            next_id = sample_from_logits(logits[:, -1], temperature, top_k, top_p)
            token_ids = torch.cat((token_ids, next_id), dim=1)
        return token_ids

