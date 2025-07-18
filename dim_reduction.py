import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dim_reduction_ui():
    st.header("🔻 Dimensionality Reduction Playground")

    # --- Dataset Selection ---
    dataset = st.selectbox("Select Dataset", ["Blobs", "Iris"])
    if dataset == "Blobs":
        X, _ = make_blobs(n_samples=300, n_features=5, centers=4, random_state=42)
    elif dataset == "Iris":
        data = load_iris()
        X = data.data
        st.caption("Using all 4 numerical features from Iris dataset")

    st.write(f"Original Data Shape: {X.shape}")

    # --- Technique Selection ---
    method = st.selectbox("Choose a Reduction Technique", ["PCA", "t-SNE"])

    if method == "PCA":
        n_components = st.slider("Number of Components", 2, min(5, X.shape[1]), 2)
        X_reduced = PCA(n_components=n_components).fit_transform(X)
        st.success(f"PCA reduced to shape: {X_reduced.shape}")

    elif method == "t-SNE":
        st.warning("t-SNE is slower. Recommended for small datasets.")
        perplexity = st.slider("Perplexity", 5, 50, 30)
        learning_rate = st.slider("Learning Rate", 10, 500, 200)
        X_reduced = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42).fit_transform(X)
        st.success("t-SNE reduced to 2D")

    # --- Scatter Plot ---
    st.subheader("🔍 2D Projection")
    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=40, alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(f"{method} Projection")
    st.pyplot(fig)
