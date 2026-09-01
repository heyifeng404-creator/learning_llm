"""教程示例共用工具。"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """尽量固定常见随机源；研究中仍应报告多种子结果。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def train_standardize(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只使用训练集统计量标准化，演示如何避免数据泄漏。"""
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def binary_accuracy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    predictions = (probabilities.reshape(-1) >= 0.5).astype(int)
    return float(np.mean(predictions == targets.reshape(-1).astype(int)))

