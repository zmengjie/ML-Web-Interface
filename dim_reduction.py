import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.datasets import make_blobs, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from umap import UMAP  # Temporarily disabled due to install issue
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.metrics import explained_variance_score
import plotly.express as px
import pandas as pd

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
    st.markdown("""
    **💡 Tip:**
    - **PCA** is good for linear variance and visualization.
    - **t-SNE** is nonlinear and preserves local structure.
    - **LDA** is supervised and focuses on class separability.
    - **KernelPCA** supports nonlinear mappings using different kernels.
    """)
    method = st.selectbox("Choose a Reduction Technique", ["PCA", "t-SNE", "LDA", "KernelPCA"])  # UMAP removed

    if method == "PCA":
        n_components = st.slider("Number of Components", 2, min(5, X.shape[1]), 2)
        model = PCA(n_components=n_components)
        X_reduced = model.fit_transform(X)
        st.success(f"PCA reduced to shape: {X_reduced.shape}")
        explained = model.explained_variance_ratio_[:n_components]
        st.info(f"Explained Variance Ratio: {np.round(explained.sum(), 4)}")
        fig_var, ax_var = plt.subplots()
        ax_var.bar(range(1, n_components+1), explained, color='skyblue')
        ax_var.set_xlabel("Component")
        ax_var.set_ylabel("Variance Ratio")
        ax_var.set_title("Explained Variance per Component")
        st.pyplot(fig_var)

    elif method == "t-SNE":
        st.warning("t-SNE is slower. Recommended for small datasets.")
        perplexity = st.slider("Perplexity", 5, 50, 30)
        learning_rate = st.slider("Learning Rate", 10, 500, 200)
        X_reduced = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42).fit_transform(X)
        st.success("t-SNE reduced to 2D")

    elif method == "KernelPCA":
        n_components = st.slider("Number of Components", 2, min(5, X.shape[1]), 2)
        kernel = st.selectbox("Kernel Function", ["linear", "poly", "rbf", "sigmoid", "cosine"])
        kpca = KernelPCA(n_components=n_components, kernel=kernel, fit_inverse_transform=True)
        X_reduced = kpca.fit_transform(X)
        st.success(f"KernelPCA reduced to shape: {X_reduced.shape}")

    elif method == "LDA":
        n_components = st.slider("Number of Components", 1, min(len(np.unique(y)) - 1, X.shape[1]), 1)
        model = LDA(n_components=n_components)
        X_reduced = model.fit_transform(X, y)
        st.success(f"LDA reduced to shape: {X_reduced.shape}")

    # --- Interactive Projection Plot ---
    st.subheader("🔍 2D or 3D Interactive Projection")
    try:
        df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(X_reduced.shape[1])])
        df['Label'] = y

        if X_reduced.shape[1] >= 3:
            fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", color=df['Label'].astype(str),
                                title=f"{method} Projection (3D)", opacity=0.7)
        else:
            fig = px.scatter(df, x="Component 1", y="Component 2", color=df['Label'].astype(str),
                             title=f"{method} Projection (2D)", opacity=0.7)

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Plotting error: {str(e)}")

    # --- Download ---
    st.subheader("📥 Download")
    try:
        csv = np.concatenate([X_reduced, y.reshape(-1, 1)], axis=1)
        csv_bytes = BytesIO()
        np.savetxt(csv_bytes, csv, delimiter=",", header=",".join([f"Component {i+1}" for i in range(X_reduced.shape[1])] + ["Label"]), comments='')
        st.download_button("Download Reduced Data CSV", data=csv_bytes.getvalue(), file_name="reduced_data.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to generate CSV: {e}")
