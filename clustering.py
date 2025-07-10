import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def clustering_ui():
    st.header("🔹 Clustering Playground")

    # --- Dataset selection ---
    dataset_choice = st.selectbox("Select a dataset", ["Blobs", "Moons", "Iris"])

    if dataset_choice == "Blobs":
        n_samples = st.slider("Number of Samples", 100, 1000, 300, 50)
        X, _ = make_blobs(n_samples=n_samples, centers=4, random_state=42)

    elif dataset_choice == "Moons":
        n_samples = st.slider("Number of Samples", 100, 1000, 300, 50)
        from sklearn.datasets import make_moons
        X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)

    elif dataset_choice == "Iris":
        iris = load_iris()
        X = iris.data[:, :2]  # use only 2D for plotting

    # --- Clustering method ---
    method = st.radio("Choose clustering method", ["K-Means", "DBSCAN", "Agglomerative"])

    if method == "K-Means":
        k = st.slider("Number of Clusters (K)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)

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

    # --- Silhouette score ---
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if unique_labels > 1:
        score = silhouette_score(X, labels)
        st.success(f"Silhouette Score: {score:.3f}")
    else:
        st.warning("Silhouette Score not available (only one cluster or noise)")

    # --- Plot ---
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=40)
    ax.set_title(f"{method} Result")
    st.pyplot(fig)

