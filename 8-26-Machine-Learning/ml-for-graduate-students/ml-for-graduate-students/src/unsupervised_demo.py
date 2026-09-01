"""PCA、K-Means 与 Isolation Forest 的小型实验。"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common import set_seed


def clustering_and_pca() -> None:
    x, true_label = load_iris(return_X_y=True)
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(x_scaled)
    kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
    cluster = kmeans.fit_predict(x_scaled)

    print("[PCA + K-Means]")
    print(f"二维累计解释方差：{pca.explained_variance_ratio_.sum():.3f}")
    print(f"轮廓系数：{silhouette_score(x_scaled, cluster):.3f}")
    print(f"与真实类别的 ARI：{adjusted_rand_score(true_label, cluster):.3f}")
    print(f"二维表示形状：{embedding.shape}")
    print("说明：聚类编号可任意置换，因此不能直接用 accuracy 比较。")


def anomaly_detection() -> None:
    normal, _ = make_blobs(
        n_samples=400, centers=[(-2, -2), (2, 2)], cluster_std=0.8, random_state=42
    )
    rng = np.random.default_rng(42)
    anomalies = rng.uniform(low=-8.0, high=8.0, size=(24, 2))
    x = np.vstack([normal, anomalies])
    y_true = np.r_[np.zeros(len(normal), dtype=int), np.ones(len(anomalies), dtype=int)]

    detector = IsolationForest(contamination=len(anomalies) / len(x), random_state=42)
    raw_prediction = detector.fit_predict(x)  # -1 表示异常
    y_prediction = (raw_prediction == -1).astype(int)
    recall = np.sum((y_prediction == 1) & (y_true == 1)) / np.sum(y_true == 1)
    false_positive_rate = np.sum((y_prediction == 1) & (y_true == 0)) / np.sum(y_true == 0)
    print("\n[Isolation Forest]")
    print(f"异常召回率：{recall:.3f}，正常样本误报率：{false_positive_rate:.3f}")


if __name__ == "__main__":
    set_seed(42)
    clustering_and_pca()
    anomaly_detection()

