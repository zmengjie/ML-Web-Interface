import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    Birch, SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from matplotlib.patches import Ellipse

def clustering_ui():
    st.header("🔹 Clustering Playground")

    # --- Dataset selection ---
    dataset_choice = st.selectbox("Select a dataset", ["Blobs", "Moons", "Iris"])

    dataset_explanations = {
        "Blobs": "🔵 **Blobs**: Artificial Gaussian clusters, good for testing spherical cluster algorithms.",
        "Moons": "🌙 **Moons**: Two interleaving half-circles. Ideal for non-convex clustering tests.",
        "Iris": "🌸 **Iris**: Real-world flower dataset. We use the first two features for 2D clustering."
    }
    st.info(dataset_explanations[dataset_choice])

    if dataset_choice == "Blobs":
        n_samples = st.slider("Number of Samples", 100, 1000, 300, 50)
        X, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)

    elif dataset_choice == "Moons":
        n_samples = st.slider("Number of Samples", 100, 1000, 300, 50)
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

    elif dataset_choice == "Iris":
        iris = load_iris()
        X = iris.data[:, :2]

    # --- Clustering method ---
    method = st.radio("Choose clustering method", [
        "K-Means", "DBSCAN", "Agglomerative", "Birch", "GMM", "Spectral"
    ])

    algo_explanations = {
        "K-Means": "**K-Means**: Partitions data into K clusters by minimizing intra-cluster variance. Good for spherical clusters.",
        "DBSCAN": "**DBSCAN**: Density-based clustering. Detects arbitrary shaped clusters and filters out noise.",
        "Agglomerative": "**Agglomerative Clustering**: Hierarchical approach that merges closest clusters iteratively.",
        "Birch": "**Birch**: Builds a compact tree and clusters incrementally. Scalable to large datasets.",
        "GMM": "**GMM (Gaussian Mixture)**: Probabilistic soft clustering based on Gaussian distributions.",
        "Spectral": "**Spectral Clustering**: Uses graph Laplacian and eigen decomposition. Effective for non-convex clusters."
    }
    st.markdown(algo_explanations[method])
    st.markdown("---")

    labels, centers, covariances = None, None, None

    if method == "K-Means":
        k = st.slider("Number of Clusters (K)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_

    elif method == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("min_samples", 2, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)

    elif method == "Agglomerative":
        k = st.slider("Number of Clusters", 2, 10, 3)
        linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
        labels = model.fit_predict(X)

    elif method == "Birch":
        threshold = st.slider("Threshold", 0.01, 2.0, 0.5, 0.01)
        k = st.slider("Number of Clusters", 2, 10, 3)
        model = Birch(threshold=threshold, n_clusters=k)
        labels = model.fit_predict(X)

    elif method == "GMM":
        k = st.slider("Number of Components", 2, 10, 3)
        model = GaussianMixture(n_components=k, random_state=0)
        labels = model.fit_predict(X)
        centers = model.means_
        covariances = model.covariances_

    elif method == "Spectral":
        k = st.slider("Number of Clusters", 2, 10, 3)
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = model.fit_predict(X)

    # --- Evaluation Metrics ---
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if unique_labels > 1:
        sil = silhouette_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        st.success(f"Silhouette: {sil:.3f} | Calinski-Harabasz: {cal:.1f} | Davies-Bouldin: {db:.3f}")
    else:
        st.warning("Clustering metrics not available (only one cluster or all noise)")

    # --- Plotting ---
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=40, label="Data")

    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='x', label='Centers')

    if method == "GMM" and covariances is not None:
        for i in range(len(centers)):
            cov = covariances[i]
            if cov.shape == (2, 2):  # Full covariance
                vals, vecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals)
                ellipse = Ellipse(centers[i], width, height, angle, edgecolor='gray', facecolor='none', lw=1.5, ls='--')
                ax.add_patch(ellipse)

    ax.set_title(f"{method} Clustering Result")
    ax.legend()
    st.pyplot(fig)
