import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.datasets import make_blobs, load_iris, load_wine
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
    dataset = st.selectbox("Select Dataset", ["Blobs", "Iris", "Wine"])
    if dataset == "Blobs":
        X, y = make_blobs(n_samples=300, n_features=5, centers=4, random_state=42)
    elif dataset == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        st.caption("Using all 4 numerical features from Iris dataset")
    elif dataset == "Wine":
        data = load_wine()
        X, y = data.data, data.target
        st.caption("Using all 13 numerical features from Wine dataset")

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
        n_components = st.slider("Number of Components", 2, X.shape[1], min(2, X.shape[1]))
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
        n_components = st.slider("Number of Components", 2, X.shape[1], min(2, X.shape[1]))
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
    st.subheader("🔍 2D and/or 3D Interactive Projection")

    # Plot mode selection
    plot_mode = st.radio("Choose projection display:", ["Auto (based on n_components)", "2D only", "3D only", "Both 2D & 3D"], horizontal=True)

    try:
        df = pd.DataFrame(X_reduced, columns=[f"Component {i+1}" for i in range(X_reduced.shape[1])])
        df['Label'] = y.astype(str)

        fig2d, fig3d = None, None

        if X_reduced.shape[1] >= 2:
            fig2d = px.scatter(df, x="Component 1", y="Component 2", color="Label",
                            title=f"{method} Projection (2D)", opacity=0.7)
            fig2d.update_traces(marker=dict(size=5, line=dict(width=0.5, color='DarkSlateGrey')))

        if X_reduced.shape[1] >= 3:
            fig3d = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", color="Label",
                                title=f"{method} Projection (3D)", opacity=0.7)
            fig3d.update_traces(marker=dict(size=4))

        if plot_mode == "Auto (based on n_components)":
            if fig3d:
                st.plotly_chart(fig3d, use_container_width=True)
            elif fig2d:
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.warning("Not enough components to plot.")
        elif plot_mode == "2D only":
            if fig2d:
                st.plotly_chart(fig2d, use_container_width=True)
            else:
                st.warning("Need at least 2 components for 2D plot.")
        elif plot_mode == "3D only":
            if fig3d:
                st.plotly_chart(fig3d, use_container_width=True)
            else:
                st.warning("Need at least 3 components for 3D plot.")
        elif plot_mode == "Both 2D & 3D":
            col1, col2 = st.columns(2)
            with col1:
                if fig2d:
                    st.plotly_chart(fig2d, use_container_width=True)
                else:
                    st.warning("Need at least 2 components.")
            with col2:
                if fig3d:
                    st.plotly_chart(fig3d, use_container_width=True)
                else:
                    st.warning("Need at least 3 components.")

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
