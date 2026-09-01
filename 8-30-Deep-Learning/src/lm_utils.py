"""字符级语言模型的数据工具；刻意保持简单，便于观察因果标签移位。"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.tokens = sorted(set(text))
        self.stoi = {token: i for i, token in enumerate(self.tokens)}
        self.itos = {i: token for token, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - set(self.stoi))
        if missing:
            raise ValueError(f"词表中没有这些字符: {missing}")
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class CausalTextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """对长度为 block_size+1 的窗口做一位错开。"""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if len(token_ids) <= block_size:
            raise ValueError("token 数必须大于 block_size")
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def random_batch(
    token_ids: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    high = len(token_ids) - block_size
    if high <= 0:
        raise ValueError("token 数必须大于 block_size")
    starts = torch.randint(high, (batch_size,))
    x = torch.stack([token_ids[i : i + block_size] for i in starts])
    y = torch.stack([token_ids[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)

