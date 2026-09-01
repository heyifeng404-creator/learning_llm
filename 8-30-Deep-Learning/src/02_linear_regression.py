"""知识点 02：线性层、MSE、mini-batch SGD 和参数估计。"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import set_seed


def make_data(n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(n, 2)
    true_w = torch.tensor([[3.0], [-2.0]])
    y = x @ true_w + 0.5 + 0.15 * torch.randn(n, 1)
    return x, y


def train(epochs: int = 40) -> tuple[nn.Linear, list[float]]:
    set_seed(2)
    x, y = make_data()
    loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
    criterion = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_x)
        history.append(total / len(x))
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    model, history = train(args.epochs)

    print(f"initial/final MSE: {history[0]:.4f} -> {history[-1]:.4f}")
    print("estimated weight:", model.weight.detach().flatten().tolist())
    print("estimated bias:  ", model.bias.detach().tolist())
    assert history[-1] < history[0]
    print("target: weight=[3, -2], bias=0.5")
    print("\n思考题：增大噪声后，参数估计和训练损失分别怎样变化？")


if __name__ == "__main__":
    main()

