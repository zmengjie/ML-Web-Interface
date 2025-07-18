import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.datasets import make_blobs, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score


def dim_reduction_ui():
    st.header("🔻 Dimensionality Reduction Playground")

    # --- Dataset Selection ---
    dataset = st.selectbox("Select Dataset", ["Blobs", "Iris"])
    if dataset == "Blobs":
        X, y = make_blobs(n_samples=300, n_features=5, centers=4, random_state=42)
    elif dataset == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        st.caption("Using all 4 numerical features from Iris dataset")

    st.write(f"Original Data Shape: {X.shape}")

    # --- Preprocessing ---
    X = StandardScaler().fit_transform(X)

    # --- Technique Selection ---
    method = st.selectbox("Choose a Reduction Technique", ["PCA", "t-SNE", "LDA"])

    if method == "PCA":
        n_components = st.slider("Number of Components", 2, min(5, X.shape[1]), 2)
        model = PCA(n_components=n_components)
        X_reduced = model.fit_transform(X)
        st.success(f"PCA reduced to shape: {X_reduced.shape}")
        explained = model.explained_variance_ratio_[:n_components]
        st.info(f"Explained Variance Ratio: {np.round(explained.sum(), 4)}")

    elif method == "t-SNE":
        st.warning("t-SNE is slower. Recommended for small datasets.")
        perplexity = st.slider("Perplexity", 5, 50, 30)
        learning_rate = st.slider("Learning Rate", 10, 500, 200)
        X_reduced = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42).fit_transform(X)
        st.success("t-SNE reduced to 2D")

    elif method == "LDA":
        n_components = st.slider("Number of Components", 1, min(len(np.unique(y)) - 1, X.shape[1]), 1)
        model = LDA(n_components=n_components)
        X_reduced = model.fit_transform(X, y)
        st.success(f"LDA reduced to shape: {X_reduced.shape}")

    # --- Projection Plot ---
    st.subheader("🔍 2D or 3D Projection")
    if X_reduced.shape[1] >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='tab10', alpha=0.8)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        ax.set_title(f"{method} Projection (3D)")
    else:
        fig, ax = plt.subplots()
        sc = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='tab10', s=40, alpha=0.8)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(f"{method} Projection (2D)")
    st.pyplot(fig)

    # --- Download ---
    st.subheader("📥 Download")
    csv = np.concatenate([X_reduced, y.reshape(-1, 1)], axis=1)
    csv_bytes = BytesIO()
    np.savetxt(csv_bytes, csv, delimiter=",", header=",".join([f"Component {i+1}" for i in range(X_reduced.shape[1])] + ["Label"]), comments='')
    st.download_button("Download Reduced Data CSV", data=csv_bytes.getvalue(), file_name="reduced_data.csv", mime="text/csv")

    img_buf = BytesIO()
    fig.savefig(img_buf, format="png")
    st.download_button("Download Projection Image", data=img_buf.getvalue(), file_name="projection.png", mime="image/png")
