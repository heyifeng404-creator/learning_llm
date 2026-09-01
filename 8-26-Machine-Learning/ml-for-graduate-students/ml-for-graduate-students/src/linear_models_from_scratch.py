"""仅用 NumPy 手写线性回归与逻辑回归。

运行：python src/linear_models_from_scratch.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from common import binary_accuracy, set_seed, train_standardize


@dataclass
class LinearRegressionGD:
    learning_rate: float = 0.05
    epochs: int = 1500
    l2: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        n_samples, n_features = x.shape
        target = y.reshape(-1, 1).astype(float)
        self.weights_ = np.zeros((n_features, 1))
        self.bias_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.epochs):
            prediction = x @ self.weights_ + self.bias_
            error = prediction - target
            data_loss = float(np.mean(error**2))
            penalty = self.l2 * float(np.sum(self.weights_**2))
            self.loss_history_.append(data_loss + penalty)

            # 对 MSE 求导；偏置不做正则化。
            grad_w = (2.0 / n_samples) * x.T @ error + 2.0 * self.l2 * self.weights_
            grad_b = float(2.0 * error.mean())
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (x @ self.weights_ + self.bias_).reshape(-1)


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.1
    epochs: int = 2000
    l2: float = 1e-3

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # clip 防止 exp 溢出；工业代码可使用 scipy.special.expit。
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        n_samples, n_features = x.shape
        target = y.reshape(-1, 1).astype(float)
        self.weights_ = np.zeros((n_features, 1))
        self.bias_ = 0.0
        self.loss_history_: list[float] = []

        for _ in range(self.epochs):
            logits = x @ self.weights_ + self.bias_
            probabilities = self._sigmoid(logits)
            eps = 1e-12
            bce = -np.mean(
                target * np.log(probabilities + eps)
                + (1.0 - target) * np.log(1.0 - probabilities + eps)
            )
            penalty = 0.5 * self.l2 * float(np.sum(self.weights_**2))
            self.loss_history_.append(float(bce + penalty))

            # sigmoid + BCE 的梯度会简化为 p - y。
            error = probabilities - target
            grad_w = x.T @ error / n_samples + self.l2 * self.weights_
            grad_b = float(error.mean())
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self.weights_ + self.bias_).reshape(-1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def regression_demo() -> None:
    x, y = make_regression(
        n_samples=600, n_features=8, n_informative=6, noise=15.0, random_state=42
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    x_train, x_test, _, _ = train_standardize(x_train, x_test)
    model = LinearRegressionGD(learning_rate=0.04, epochs=2000, l2=1e-4).fit(
        x_train, y_train
    )
    rmse = mean_squared_error(y_test, model.predict(x_test)) ** 0.5
    print(f"[线性回归] 初始损失={model.loss_history_[0]:.2f}, "
          f"最终损失={model.loss_history_[-1]:.2f}, 测试 RMSE={rmse:.2f}")


def classification_demo() -> None:
    x, y = make_classification(
        n_samples=800,
        n_features=12,
        n_informative=7,
        n_redundant=2,
        class_sep=1.3,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )
    x_train, x_test, _, _ = train_standardize(x_train, x_test)
    model = LogisticRegressionGD().fit(x_train, y_train)
    probability = model.predict_proba(x_test)
    print(
        f"[逻辑回归] 初始损失={model.loss_history_[0]:.4f}, "
        f"最终损失={model.loss_history_[-1]:.4f}, "
        f"Accuracy={binary_accuracy(probability, y_test):.3f}, "
        f"ROC-AUC={roc_auc_score(y_test, probability):.3f}"
    )


if __name__ == "__main__":
    set_seed(42)
    regression_demo()
    classification_demo()

