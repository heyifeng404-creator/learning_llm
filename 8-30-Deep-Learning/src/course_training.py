"""迷你 GPT 的共享训练函数，供第 10、12 章使用。"""

from __future__ import annotations

import torch

from common import get_device, read_corpus, set_seed
from lm_utils import CharTokenizer, random_batch
from mini_gpt_model import GPTConfig, MiniGPT


def prepare_data() -> tuple[CharTokenizer, torch.Tensor, torch.Tensor]:
    text = read_corpus()
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(0.8 * len(ids))
    # 训练段重复只是为了让玩具模型获得足够更新；验证段保持独立位置。
    train_ids = ids[:split].repeat(8)
    valid_ids = ids[split:]
    return tokenizer, train_ids, valid_ids


def train_minigpt(
    steps: int = 80,
    seed: int = 10,
) -> tuple[MiniGPT, CharTokenizer, torch.Tensor, list[float]]:
    set_seed(seed)
    device = get_device()
    tokenizer, train_ids, valid_ids = prepare_data()
    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        x, y = random_batch(train_ids, config.block_size, batch_size=12, device=device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return model, tokenizer, valid_ids, losses


@torch.no_grad()
def token_loss(model: MiniGPT, ids: torch.Tensor, block_size: int | None = None) -> float:
    """按不重叠窗口估计 token 交叉熵；教学版，不代替标准评测框架。"""
    model.eval()
    device = next(model.parameters()).device
    size = block_size or model.config.block_size
    losses: list[tuple[float, int]] = []
    for start in range(0, len(ids) - 1, size):
        chunk = ids[start : start + size + 1]
        if len(chunk) < 2:
            continue
        x, y = chunk[:-1].unsqueeze(0).to(device), chunk[1:].unsqueeze(0).to(device)
        _, loss = model(x, y)
        assert loss is not None
        losses.append((loss.item(), y.numel()))
    total_tokens = sum(count for _, count in losses)
    return sum(loss * count for loss, count in losses) / total_tokens

