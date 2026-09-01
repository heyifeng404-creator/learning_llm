"""混合类型表格数据的端到端、防泄漏实验模板。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import set_seed


def make_mixed_data(seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    features, target = make_classification(
        n_samples=2400,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.78, 0.22],
        class_sep=1.0,
        random_state=seed,
    )
    frame = pd.DataFrame(features, columns=[f"numeric_{i}" for i in range(8)])
    # 构造两个演示用类别变量。真实项目应根据业务含义构造特征。
    frame["region"] = pd.cut(
        frame["numeric_0"], bins=[-np.inf, -0.7, 0.7, np.inf], labels=["west", "central", "east"]
    ).astype(object)
    frame["channel"] = np.where(frame["numeric_1"] > 0, "online", "offline")

    rng = np.random.default_rng(seed)
    for column in ["numeric_2", "numeric_5", "region"]:
        missing_rows = rng.choice(len(frame), size=int(0.06 * len(frame)), replace=False)
        frame.loc[missing_rows, column] = np.nan
    return frame, target


def bootstrap_auc(
    y_true: np.ndarray, probability: np.ndarray, repeats: int = 500, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(repeats):
        index = rng.integers(0, len(y_true), size=len(y_true))
        # 某些极端重采样可能只有一个类别，无法计算 AUC。
        if np.unique(y_true[index]).size == 2:
            values.append(roc_auc_score(y_true[index], probability[index]))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    set_seed(42)
    frame, target = make_mixed_data()
    x_train, x_test, y_train, y_test = train_test_split(
        frame, target, test_size=0.25, stratify=target, random_state=42
    )
    numeric_columns = x_train.select_dtypes(include="number").columns.tolist()
    category_columns = x_train.select_dtypes(exclude="number").columns.tolist()

    numeric_transform = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    category_transform = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric_transform, numeric_columns),
         ("category", category_transform, category_columns)]
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
        ]
    )

    search = GridSearchCV(
        pipeline,
        param_grid={"model__C": np.logspace(-2, 2, 7)},
        scoring="average_precision",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    probability = search.predict_proba(x_test)[:, 1]

    # 阈值应在验证数据上按成本选择。这里只为展示固定阈值下的指标。
    threshold = 0.5
    prediction = (probability >= threshold).astype(int)
    auc_low, auc_high = bootstrap_auc(y_test, probability)
    prob_true, prob_pred = calibration_curve(y_test, probability, n_bins=8)

    print(f"训练/测试样本：{len(x_train)}/{len(x_test)}")
    print(f"最佳超参数：{search.best_params_}")
    print(f"ROC-AUC：{roc_auc_score(y_test, probability):.3f} "
          f"(bootstrap 95% CI {auc_low:.3f}–{auc_high:.3f})")
    print(f"PR-AUC：{average_precision_score(y_test, probability):.3f}")
    print(f"Accuracy：{accuracy_score(y_test, prediction):.3f}")
    print(f"F1：{f1_score(y_test, prediction):.3f}")
    print(f"Brier score：{brier_score_loss(y_test, probability):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, prediction))
    print("校准曲线点 (预测均值 → 实际正例率)：")
    print(list(zip(np.round(prob_pred, 3), np.round(prob_true, 3))))


if __name__ == "__main__":
    main()

