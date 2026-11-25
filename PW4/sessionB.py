import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # B1. Load and hide the labels
    wine = load_wine(as_frame=True)
    X = wine.data.copy()
    cultivar = wine.target.copy()

    print("=== B1: WINE DESCR ===")
    print(wine.DESCR)
    print("\n=== B1: X.describe() ===")
    desc = X.describe()
    print(desc.round(3).to_string())

    # B2. PCA on raw data
    pca_raw = PCA(n_components=None)
    pca_raw.fit(X)
    pc1_loadings_raw = pd.Series(pca_raw.components_[0], index=X.columns).sort_values(
        key=abs, ascending=False
    )
    print("\n=== B2: RAW PCA ===")
    print(f"PC1 explained variance ratio (raw): {pca_raw.explained_variance_ratio_[0]:.4f}")
    print("PC1 loadings (raw, sorted by |loading|):\n", pc1_loadings_raw.round(4))

    # B3. Scale, then PCA again
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X)
    print("\n=== B3: SCALED DATA CHECK ===")
    print("Z mean max abs:", float(np.max(np.abs(Z.mean(axis=0)))))
    print("Z std min/max:", float(Z.std(axis=0).min()), float(Z.std(axis=0).max()))

    pca_scaled = PCA(n_components=None)
    Z_pca = pca_scaled.fit_transform(Z)
    evr = pca_scaled.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    scree_df = pd.DataFrame({
        "PC": [f"PC{i}" for i in range(1, len(evr) + 1)],
        "explained_variance_ratio": evr,
        "cumulative_variance": cum_evr,
    })
    scree_df.to_csv("results/B3_pca_variance.csv", index=False)

    loadings_12 = pd.DataFrame({
        "PC1": pca_scaled.components_[0],
        "PC2": pca_scaled.components_[1],
    }, index=X.columns).sort_values("PC1", ascending=False)
    loadings_12.to_csv("results/B3_loadings_pc1_pc2.csv")

    n_90 = int(np.argmax(cum_evr >= 0.90) + 1)
    print(f"Components needed for >=90% variance: {n_90}, PC1+PC2 variance: {cum_evr[1]:.4f}")
    print("\nPC1 & PC2 loadings (sorted by PC1):\n", loadings_12.round(4).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    idx = np.arange(1, len(evr) + 1)
    axes[0].bar(idx, evr, alpha=0.75, label="Individual EVR")
    axes[0].plot(idx, cum_evr, marker="o", color="red", label="Cumulative EVR")
    axes[0].axhline(0.90, color="green", linestyle="--", label="0.90 threshold")
    axes[0].set_xticks(idx)
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].set_title("Scree Plot (Scaled Wine Data)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.75, color="steelblue")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Wines on PC1-PC2 Plane (Unlabeled)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/B3_pca_scree_and_scatter.png", dpi=180)
    plt.close()

    # B4. K-Means: random_state sweep & elbow curve
    inertias_rs = []
    for rs in range(8):
        km_rs = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=rs)
        km_rs.fit(Z)
        inertias_rs.append({"random_state": rs, "inertia": float(km_rs.inertia_)})
    rs_df = pd.DataFrame(inertias_rs)
    rs_df.to_csv("results/B4_random_state_inertia.csv", index=False)
    print("\n=== B4(a): k=3, n_init=1 across random_state 0..7 ===")
    print(rs_df.round(2).to_string(index=False))

    elbow_rows = []
    for k in range(1, 11):
        km_k = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
        km_k.fit(Z)
        elbow_rows.append({"k": k, "inertia": float(km_k.inertia_)})
    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv("results/B4_elbow_k1_k10.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(elbow_df["k"], elbow_df["inertia"], marker="o")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means Elbow Curve (Scaled Wine Data)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/B4_elbow_curve.png", dpi=180)
    plt.close()

    # B5. What are the clusters? (k=3)
    chosen_k = 3
    km = KMeans(
        n_clusters=chosen_k, init="k-means++", n_init=10, max_iter=300, random_state=0
    )
    km.fit(Z)
    cluster_counts = np.bincount(km.labels_)
    centers_real = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=X.columns,
    )
    centers_real.insert(0, "cluster", range(chosen_k))
    centers_real.insert(1, "size", cluster_counts)
    centers_real.to_csv("results/B5_cluster_centers_real_units.csv", index=False)

    print("\n=== B5: CLUSTER SIZES & CENTERS (REAL UNITS) ===")
    print("Cluster sizes:", cluster_counts.tolist())
    print(centers_real.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km.labels_, cmap="viridis", alpha=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PC1-PC2 Colored by K-Means Cluster (k=3)")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/B5_clusters_pc1_pc2.png", dpi=180)
    plt.close()

    # B6. The reveal (compare against cultivar)
    def agreement_score(y_true: pd.Series, labels: np.ndarray) -> tuple[pd.DataFrame, float]:
        ct = pd.crosstab(y_true, labels)
        score = float(ct.max(axis=0).sum() / len(y_true))
        return ct, score

    ct_scaled, agr_scaled = agreement_score(cultivar, km.labels_)

    km_raw = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km_raw.fit(X)
    ct_raw, agr_raw = agreement_score(cultivar, km_raw.labels_)

    km_pca2 = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
    km_pca2.fit(Z_pca[:, :2])
    ct_pca2, agr_pca2 = agreement_score(cultivar, km_pca2.labels_)

    print("\n=== B6: CROSSTABS ===")
    print("Scaled K-Means crosstab:\n", ct_scaled)
    print("Raw K-Means crosstab:\n", ct_raw)
    print("PCA(2D) K-Means crosstab:\n", ct_pca2)

    agree_df = pd.DataFrame([
        {"setting": "scaled_kmeans_k3", "agreement": agr_scaled},
        {"setting": "raw_kmeans_k3", "agreement": agr_raw},
        {"setting": "pca2_kmeans_k3", "agreement": agr_pca2},
    ])
    agree_df.to_csv("results/B_agreement.csv", index=False)
    print("\n=== B6: AGREEMENT TABLE (results/B_agreement.csv) ===")
    print(agree_df.round(4).to_string(index=False))

    # Bonus: Supervised comparison on wine split
    Xw_train, Xw_test, yw_train, yw_test = train_test_split(
        X, cultivar, test_size=0.25, random_state=42, stratify=cultivar
    )
    pipe_knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    pipe_knn.fit(Xw_train, yw_train)
    acc_knn = float(pipe_knn.score(Xw_test, yw_test))

    pipe_pca_knn = make_pipeline(
        StandardScaler(), PCA(n_components=2), KNeighborsClassifier(n_neighbors=5)
    )
    pipe_pca_knn.fit(Xw_train, yw_train)
    acc_pca_knn = float(pipe_pca_knn.score(Xw_test, yw_test))

    bonus_w_df = pd.DataFrame([
        {"model": "StandardScaler + KNN(5)", "test_accuracy": acc_knn},
        {"model": "StandardScaler + PCA(2) + KNN(5)", "test_accuracy": acc_pca_knn},
    ])
    bonus_w_df.to_csv("results/B_bonus_supervised.csv", index=False)
    print("\n=== BONUS (WINE SUPERVISED VS UNSUPERVISED) ===")
    print(bonus_w_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()