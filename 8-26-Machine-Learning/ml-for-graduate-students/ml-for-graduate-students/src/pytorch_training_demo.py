"""一个包含训练、验证、早停和最终测试的 PyTorch 示例。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import set_seed


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)  # 返回 logits，不提前 sigmoid


def make_loaders() -> tuple[DataLoader, DataLoader, DataLoader]:
    x, y = make_moons(n_samples=1800, noise=0.25, random_state=42)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=42
    )
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    def loader(features: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        return DataLoader(dataset, batch_size=64, shuffle=shuffle)

    return (
        loader(x_train, y_train, True),
        loader(x_valid, y_valid, False),
        loader(x_test, y_test, False),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += float(criterion(logits, targets)) * len(inputs)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(targets.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(probabilities),
        np.concatenate(labels),
    )


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = make_loaders()
    model = MLP().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)

    best_state = deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    stale_epochs = 0
    patience = 20

    for epoch in range(1, 301):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        valid_loss, _, _ = evaluate(model, valid_loader, criterion, device)
        if valid_loss < best_valid_loss - 1e-4:
            best_valid_loss = valid_loss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"epoch={epoch:03d} valid_loss={valid_loss:.4f}")
        if stale_epochs >= patience:
            print(f"早停于第 {epoch} 轮，恢复验证集最佳参数。")
            break

    model.load_state_dict(best_state)
    test_loss, probability, target = evaluate(model, test_loader, criterion, device)
    prediction = (probability >= 0.5).astype(int)
    print(f"设备：{device}")
    print(f"Test loss={test_loss:.4f}")
    print(f"Test Accuracy={accuracy_score(target, prediction):.3f}")
    print(f"Test ROC-AUC={roc_auc_score(target, probability):.3f}")


if __name__ == "__main__":
    main()

