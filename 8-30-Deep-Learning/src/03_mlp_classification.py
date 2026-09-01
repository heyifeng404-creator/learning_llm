"""知识点 03：MLP、非线性激活、交叉熵与分类决策边界。"""

import argparse

import torch
from torch import nn

from common import set_seed


def make_xor(n: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """四个象限的 XOR；线性模型无法用一条直线正确分开。"""
    x = 2 * torch.rand(n, 2) - 1
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long()
    x += 0.08 * torch.randn_like(x)
    return x, y


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 输出 logits；CrossEntropyLoss 内部做 log-softmax。
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(steps: int = 120) -> tuple[nn.Module, float]:
    set_seed(3)
    x, y = make_xor()
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=-1) == y).float().mean().item()
    return model, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    model, accuracy = train(args.steps)
    probes = torch.tensor([[0.7, 0.7], [0.7, -0.7], [-0.7, 0.7], [-0.7, -0.7]])
    with torch.no_grad():
        probs = model(probes).softmax(dim=-1)
    print("training accuracy:", round(accuracy, 4))
    print("probe P(class=1):", [round(v, 3) for v in probs[:, 1].tolist()])
    assert accuracy > 0.90
    print("\n思考题：移除 ReLU 后，为什么增加线性层仍不能解决 XOR？")


if __name__ == "__main__":
    main()

