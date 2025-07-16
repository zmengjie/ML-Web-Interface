import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def clustering_ui():
    st.header("🔹 Clustering Playground")

    # --- Dataset selection ---
    dataset_choice = st.selectbox("Select a dataset", ["Blobs", "Moons", "Iris"])

    if dataset_choice == "Blobs":
        n_samples = st.slider("Number of Samples", 100, 1000, 300, 50)
        X, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)

    elif dataset_choice == "Moons":
        n_samples = st.slider("Number of Samples", 100, 1000, 300, 50)
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

    elif dataset_choice == "Iris":
        iris = load_iris()
        X = iris.data[:, :2]  # use only 2D for plotting

    # --- Clustering method ---
    method = st.radio("Choose clustering method", [
        "K-Means", "DBSCAN", "Agglomerative", "Birch", "GMM", "Spectral"
    ])

    centers = None  # for plotting

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
        # Birch doesn’t expose centers directly

    elif method == "GMM":
        k = st.slider("Number of Components", 2, 10, 3)
        model = GaussianMixture(n_components=k, random_state=0)
        labels = model.fit_predict(X)
        centers = model.means_

    elif method == "Spectral":
        k = st.slider("Number of Clusters", 2, 10, 3)
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = model.fit_predict(X)

    # --- Silhouette Score ---
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if unique_labels > 1:
        score = silhouette_score(X, labels)
        st.success(f"Silhouette Score: {score:.3f}")
    else:
        st.warning("Silhouette Score not available (only one cluster or noise)")

    # --- Plot ---
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=40, label="Data")

    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='x', label='Centers')

    ax.set_title(f"{method} Clustering Result")
    ax.legend()
    st.pyplot(fig)
