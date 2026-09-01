"""知识点 11：冻结基础权重，只训练低秩 LoRA 增量。"""

import argparse
import math

import torch
from torch import nn

from common import count_parameters, set_seed


class LoRALinear(nn.Module):
    """y = x W^T + scale * x A^T B^T，W 冻结，A/B 可训练。"""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank 必须为正")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = (x @ self.lora_a.T) @ self.lora_b.T
        return self.base(x) + self.scale * update


def train(steps: int = 120) -> tuple[LoRALinear, list[float]]:
    set_seed(11)
    in_features, out_features, rank = 24, 12, 3
    base = nn.Linear(in_features, out_features, bias=False)
    frozen_weight = base.weight.detach().clone()

    # 构造一个确实可由低秩增量表达的“新任务”。
    teacher_a = torch.randn(rank, in_features) * 0.25
    teacher_b = torch.randn(out_features, rank) * 0.25
    x = torch.randn(256, in_features)
    with torch.no_grad():
        target = base(x) + (x @ teacher_a.T) @ teacher_b.T

    model = LoRALinear(base, rank=rank, alpha=rank)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.04)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert torch.equal(model.base.weight, frozen_weight)
    return model, losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    model, losses = train(args.steps)
    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)
    print(f"all parameters: {total:,}")
    print(f"trainable LoRA parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"adaptation MSE: {losses[0]:.6f} -> {losses[-1]:.6f}")
    print("基础矩阵保持逐位不变，任务差异由两个低秩矩阵学习。")
    print("\n思考题：rank 增大会如何影响表达能力、参数量和过拟合风险？")


if __name__ == "__main__":
    main()

