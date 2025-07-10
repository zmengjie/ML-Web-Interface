import streamlit as st
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def clustering_ui():
    st.header("🔹 Clustering Playground")

    st.markdown("Try adjusting the number of clusters and sample points to see how K-Means behaves.")

    n_samples = st.slider("Number of Samples", min_value=100, max_value=1000, value=300, step=50)
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)

    # Generate synthetic data
    X, _ = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=42)

    # Fit KMeans
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X)

    # Plot result
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=40)
    ax.set_title("K-Means Clustering Result")
    st.pyplot(fig)
