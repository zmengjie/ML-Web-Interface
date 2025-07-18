import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import plotly.express as px


def anomaly_detection_ui():
    st.header("🔍 Anomaly Detection")

    # Sample datasets
    dataset_name = st.selectbox("Select Dataset", ["Synthetic", "Wine", "Iris"])

    if dataset_name == "Synthetic":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)
        outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
        X = np.vstack([X, outliers])
        y_true = np.array([1]*300 + [-1]*20)
        data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        data['Label'] = y_true

    elif dataset_name == "Wine":
        from sklearn.datasets import load_wine
        raw = load_wine()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['Label'] = raw.target

    elif dataset_name == "Iris":
        from sklearn.datasets import load_iris
        raw = load_iris()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['Label'] = raw.target

    st.write(f"Original Data Shape: {data.shape}")
    features = st.multiselect("Select features for detection", data.columns[:-1], default=list(data.columns[:2]))

    X = data[features].values
    X = StandardScaler().fit_transform(X)

    method = st.selectbox("Choose Detection Method", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])

    if method == "Isolation Forest":
        model = IsolationForest(contamination=0.1, random_state=42)
        preds = model.fit_predict(X)
    elif method == "One-Class SVM":
        model = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
        preds = model.fit_predict(X)
    elif method == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        preds = model.fit_predict(X)

    data['Anomaly'] = np.where(preds == -1, "Outlier", "Inlier")

    st.subheader("📊 Visualization")
    fig = px.scatter(data, x=features[0], y=features[1], color='Anomaly', symbol='Anomaly',
                     title=f"Anomaly Detection using {method}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Anomaly Counts")
    st.write(data['Anomaly'].value_counts())
