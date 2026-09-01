"""比较经典监督学习模型，并演示 Pipeline 与交叉验证调参。"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import set_seed


def main() -> None:
    set_seed(42)
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )

    models = {
        "Logistic": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000))]
        ),
        "RBF-SVM": Pipeline(
            [("scale", StandardScaler()), ("model", SVC(probability=True, random_state=42))]
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=200, random_state=42
        ),
    }

    print("模型                    Accuracy  ROC-AUC")
    print("-" * 46)
    for name, model in models.items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        print(
            f"{name:<24}{accuracy_score(y_test, prediction):>8.3f}"
            f"{roc_auc_score(y_test, probability):>9.3f}"
        )

    # 预处理放在 Pipeline 内部，因此每个交叉验证折不会看到其他折的均值和方差。
    search = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("model", SVC(probability=True))]),
        param_grid={"model__C": [0.1, 1.0, 10.0], "model__gamma": ["scale", 0.01]},
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    test_probability = search.predict_proba(x_test)[:, 1]
    print(f"\nSVM 最佳参数：{search.best_params_}")
    print(f"SVM 测试 ROC-AUC：{roc_auc_score(y_test, test_probability):.3f}")
    print("\n阈值 0.5 下的分类报告：")
    print(classification_report(y_test, np.asarray(test_probability >= 0.5, dtype=int)))


if __name__ == "__main__":
    main()
