"""知识点 04：过拟合、Dropout、AdamW 权重衰减和训练/验证差距。"""

import argparse
from dataclasses import dataclass

import torch
from torch import nn

from common import set_seed


@dataclass
class Result:
    train_acc: float
    valid_acc: float
    weight_norm: float


def make_dataset(n: int, noisy_labels: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 20)
    score = 1.7 * x[:, 0] - 1.2 * x[:, 1] + 0.7 * x[:, 2]
    y = (score > 0).long()
    if noisy_labels:
        # 小训练集含标签噪声，宽网络很容易记住这些错误。
        flip = torch.rand(n) < 0.20
        y[flip] = 1 - y[flip]
    return x, y


class Classifier(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run(dropout: float, weight_decay: float, steps: int) -> Result:
    set_seed(4)
    train_x, train_y = make_dataset(96, noisy_labels=True)
    valid_x, valid_y = make_dataset(512, noisy_labels=False)
    model = Classifier(dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=weight_decay)
    for _ in range(steps):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()  # 评估时必须关闭 Dropout。
    with torch.no_grad():
        train_acc = (model(train_x).argmax(-1) == train_y).float().mean().item()
        valid_acc = (model(valid_x).argmax(-1) == valid_y).float().mean().item()
        weight_norm = sum(p.pow(2).sum() for p in model.parameters()).sqrt().item()
    return Result(train_acc, valid_acc, weight_norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    plain = run(dropout=0.0, weight_decay=0.0, steps=args.steps)
    regularized = run(dropout=0.25, weight_decay=0.03, steps=args.steps)
    print("configuration     train_acc valid_acc weight_norm")
    print(f"plain             {plain.train_acc:9.3f} {plain.valid_acc:9.3f} {plain.weight_norm:11.3f}")
    print(f"dropout + AdamW   {regularized.train_acc:9.3f} {regularized.valid_acc:9.3f} {regularized.weight_norm:11.3f}")
    print("注意：一次小实验不保证正则化必然提高准确率，应比较多个随机种子。")
    print("\n思考题：为何只报告训练准确率会掩盖记忆噪声的问题？")


if __name__ == "__main__":
    main()

