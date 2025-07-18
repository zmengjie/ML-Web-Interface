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
import pandas as pd
from io import BytesIO

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
        "K-Means": "**K-Means**: Fast, scalable method that minimizes intra-cluster distance. Best for spherical clusters.",
        "DBSCAN": "**DBSCAN**: Density-based clustering algorithm that can find arbitrarily shaped clusters and identify noise.",
        "Agglomerative": "**Agglomerative Clustering**: Hierarchical bottom-up clustering that merges clusters based on a linkage strategy.",
        "Birch": "**Birch**: Efficient for large datasets. Builds a tree structure and clusters incrementally.",
        "GMM": "**GMM (Gaussian Mixture Model)**: Soft clustering based on probabilistic assignment using Gaussian distributions.",
        "Spectral": "**Spectral Clustering**: Converts data into a graph structure and performs dimensionality reduction before clustering."
    }
    st.markdown(algo_explanations[method])
    st.markdown("---")

    labels, centers, covariances = None, None, None

    if method == "K-Means":
        k = st.slider("Number of Clusters (K)", 2, 10, 3)
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_

        st.subheader("📈 Elbow & Silhouette Plot")
        with st.expander("❓ How to read this plot?"):
            st.markdown("""
            - The **blue line** shows **WCSS (Within-Cluster Sum of Squares)**: lower values mean more compact clusters.
            - Look for the **'elbow point'**: where the curve flattens. It's a good candidate for the optimal number of clusters.
            - The **orange line** shows **Silhouette Score** (range −1 to 1): higher values indicate better-separated clusters.
            - Look for the **peak**: it suggests the best separation.
            
            Use both together: pick a k where WCSS flattens and Silhouette Score is high.
            """)
        wcss, sils = [], []
        for i in range(2, 11):
            km = KMeans(n_clusters=i, random_state=0).fit(X)
            wcss.append(km.inertia_)
            sils.append(silhouette_score(X, km.labels_))
        fig_k, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(range(2, 11), wcss, label="WCSS", color="tab:blue")
        ax2.plot(range(2, 11), sils, label="Silhouette", color="tab:orange")
        ax1.set_ylabel("WCSS", color="tab:blue")
        ax2.set_ylabel("Silhouette Score", color="tab:orange")
        fig_k.legend(loc="upper right")
        ax1.set_title("K Selection Metrics")
        st.pyplot(fig_k)


    elif method == "DBSCAN":
        eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("min_samples", 2, 20, 5)
        metric = st.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"], key="db_metric")
        with st.expander("❓ What do these parameters mean?"):
            st.markdown("""
            - **Epsilon (eps)**: Radius to search for neighboring points. Smaller values mean tighter clusters.
            - **min_samples**: Minimum points required (within eps radius) to form a dense region (core point).
            - **Distance Metric**:
                - `euclidean`: Straight-line distance.
                - `manhattan`: Block-wise distance (L1 norm).
                - `cosine`: Measures angle similarity — useful for text/high-dimensional data.
            
            **Evaluation Metrics:**
            - **Silhouette Score**: (−1 to 1) — higher is better. Measures how well-separated the clusters are.
            - **Calinski-Harabasz**: Higher is better. Ratio of between-cluster to within-cluster dispersion.
            - **Davies-Bouldin**: Lower is better. Measures cluster overlap — penalizes similar clusters.
            """)
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
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

        if hasattr(model, 'covariances_'):
            if model.covariance_type == 'full':
                covariances = model.covariances_
            elif model.covariance_type == 'tied':
                covariances = [model.covariances_] * model.n_components
            elif model.covariance_type == 'diag':
                covariances = [np.diag(cov) for cov in model.covariances_]
            elif model.covariance_type == 'spherical':
                covariances = [np.eye(X.shape[1]) * cov for cov in model.covariances_]

    elif method == "Spectral":
        k = st.slider("Number of Clusters", 2, 10, 3)
        model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans')
        labels = model.fit_predict(X)

    # --- Evaluation ---
    unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
    if unique_labels > 1:
        sil = silhouette_score(X, labels)
        cal = calinski_harabasz_score(X, labels)
        db = davies_bouldin_score(X, labels)
        st.success(f"**Silhouette**: {sil:.3f} | **Calinski-Harabasz**: {cal:.1f} | **Davies-Bouldin**: {db:.3f}")
    else:
        st.warning("Clustering metrics not available (only one cluster or all noise)")

    # --- Cluster Summary ---
    st.subheader("📋 Cluster Summary")
    for label in np.unique(labels):
        count = np.sum(labels == label)
        centroid = np.mean(X[labels == label], axis=0)
        st.write(f"Cluster {label}: {count} points, centroid at {np.round(centroid, 2)}")

    # --- Plot ---
    st.markdown("---")
    st.subheader("📊 Clustering Result")
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=40, label="Data")
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='x', label='Centers')
    if method == "GMM" and covariances is not None and X.shape[1] == 2:
        for i in range(len(centers)):
            try:
                cov = covariances[i]
                if cov.ndim == 1 and cov.shape[0] == 2:
                    cov = np.diag(cov)
                elif cov.ndim == 0 or cov.shape == (): cov = np.eye(2) * cov
                elif cov.ndim != 2 or cov.shape != (2, 2): continue
                vals, vecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * np.sqrt(vals)
                ellipse = Ellipse(xy=centers[i], width=width, height=height, angle=angle,
                                  edgecolor='gray', facecolor='none', lw=1.5, ls='--')
                ax.add_patch(ellipse)
            except Exception as e:
                print(f"Skipping ellipse for component {i}: {e}")
    ax.set_title(f"{method} Clustering Result")
    ax.legend()
    st.pyplot(fig)

    # --- Downloadable Outputs ---
    st.subheader("📥 Download Results")
    df_out = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
    df_out["Cluster"] = labels
    csv = df_out.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "clustered_data.csv", "text/csv")
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.download_button("Download Plot", buf.getvalue(), "cluster_plot.png", "image/png")
