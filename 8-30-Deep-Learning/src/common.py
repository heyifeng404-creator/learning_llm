"""课程案例共享工具。"""

from __future__ import annotations

import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int = 42) -> None:
    """固定常用随机源，使教学实验尽量可复现。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def grad_norm(model: torch.nn.Module) -> float:
    squared = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt())


def read_corpus() -> str:
    return (ROOT / "data" / "tiny_corpus.txt").read_text(encoding="utf-8")

