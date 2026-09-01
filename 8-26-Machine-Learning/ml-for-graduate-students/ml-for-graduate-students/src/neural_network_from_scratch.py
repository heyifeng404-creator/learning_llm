"""仅用 NumPy 实现二分类两层神经网络和梯度检查。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from common import binary_accuracy, set_seed, train_standardize


@dataclass
class TwoLayerNet:
    input_dim: int
    hidden_dim: int = 16
    learning_rate: float = 0.05
    l2: float = 1e-4
    seed: int = 42

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # He 初始化适合 ReLU。
        self.w1 = rng.normal(0.0, np.sqrt(2.0 / self.input_dim), (self.input_dim, self.hidden_dim))
        self.b1 = np.zeros((1, self.hidden_dim))
        self.w2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ self.w2 + self.b2
        probability = self._sigmoid(logits)
        return probability, {"x": x, "z1": z1, "h1": h1, "probability": probability}

    def loss_and_gradients(
        self, x: np.ndarray, y: np.ndarray
    ) -> tuple[float, dict[str, np.ndarray]]:
        target = y.reshape(-1, 1).astype(float)
        probability, cache = self.forward(x)
        eps = 1e-12
        data_loss = -np.mean(
            target * np.log(probability + eps)
            + (1.0 - target) * np.log(1.0 - probability + eps)
        )
        penalty = 0.5 * self.l2 * (np.sum(self.w1**2) + np.sum(self.w2**2))
        loss = float(data_loss + penalty)

        n = len(x)
        d_logits = (probability - target) / n
        grad_w2 = cache["h1"].T @ d_logits + self.l2 * self.w2
        grad_b2 = d_logits.sum(axis=0, keepdims=True)
        d_hidden = d_logits @ self.w2.T
        d_z1 = d_hidden * (cache["z1"] > 0.0)
        grad_w1 = x.T @ d_z1 + self.l2 * self.w1
        grad_b1 = d_z1.sum(axis=0, keepdims=True)
        gradients = {"w1": grad_w1, "b1": grad_b1, "w2": grad_w2, "b2": grad_b2}
        return loss, gradients

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 2000) -> list[float]:
        history: list[float] = []
        for _ in range(epochs):
            loss, gradients = self.loss_and_gradients(x, y)
            history.append(loss)
            for name, gradient in gradients.items():
                setattr(self, name, getattr(self, name) - self.learning_rate * gradient)
        return history

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0].reshape(-1)

    def gradient_check(self, x: np.ndarray, y: np.ndarray, epsilon: float = 1e-5) -> float:
        """抽查 w1[0, 0] 的解析梯度与中心差分。"""
        _, gradients = self.loss_and_gradients(x, y)
        analytic = float(gradients["w1"][0, 0])
        original = float(self.w1[0, 0])
        self.w1[0, 0] = original + epsilon
        loss_plus, _ = self.loss_and_gradients(x, y)
        self.w1[0, 0] = original - epsilon
        loss_minus, _ = self.loss_and_gradients(x, y)
        self.w1[0, 0] = original
        numeric = (loss_plus - loss_minus) / (2.0 * epsilon)
        return abs(analytic - numeric) / max(1e-12, abs(analytic) + abs(numeric))


def main() -> None:
    set_seed(42)
    x, y = make_moons(n_samples=800, noise=0.22, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    x_train, x_test, _, _ = train_standardize(x_train, x_test)
    network = TwoLayerNet(input_dim=2, hidden_dim=24, learning_rate=0.08)
    relative_error = network.gradient_check(x_train[:8], y_train[:8])
    history = network.fit(x_train, y_train, epochs=2500)
    accuracy = binary_accuracy(network.predict_proba(x_test), y_test)
    print(f"梯度检查相对误差：{relative_error:.2e}")
    print(f"训练损失：{history[0]:.4f} → {history[-1]:.4f}")
    print(f"测试 Accuracy：{accuracy:.3f}")


if __name__ == "__main__":
    main()
