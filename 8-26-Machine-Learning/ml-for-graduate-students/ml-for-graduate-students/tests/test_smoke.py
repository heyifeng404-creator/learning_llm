"""核心手写实现的快速测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from linear_models_from_scratch import LinearRegressionGD, LogisticRegressionGD
from neural_network_from_scratch import TwoLayerNet


def test_linear_regression_learns_simple_line() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 0.5
    model = LinearRegressionGD(learning_rate=0.08, epochs=800).fit(x, y)
    mse = np.mean((model.predict(x) - y) ** 2)
    assert mse < 1e-5


def test_logistic_regression_separates_easy_data() -> None:
    rng = np.random.default_rng(1)
    negative = rng.normal(loc=-2.0, scale=0.4, size=(100, 2))
    positive = rng.normal(loc=2.0, scale=0.4, size=(100, 2))
    x = np.vstack([negative, positive])
    y = np.r_[np.zeros(100), np.ones(100)]
    model = LogisticRegressionGD(epochs=800).fit(x, y)
    assert np.mean(model.predict(x) == y) > 0.99


def test_neural_network_gradient() -> None:
    x = np.array([[-1.0, 0.2], [0.4, 1.3], [1.2, -0.7], [-0.3, -1.0]])
    y = np.array([0, 1, 1, 0])
    model = TwoLayerNet(input_dim=2, hidden_dim=5, seed=7)
    assert model.gradient_check(x, y) < 1e-6
