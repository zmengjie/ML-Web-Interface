import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from scipy.stats import zscore
from sklearn.datasets import load_iris, load_wine, fetch_openml
from sklearn.datasets import make_blobs



def load_datasets(dataset_name):
    from sklearn.datasets import fetch_openml, make_classification
    import pandas as pd
    import numpy as np

    if dataset_name == "Synthetic":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)
        outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
        X = np.vstack([X, outliers])
        y_true = np.array([1]*300 + [-1]*20)
        data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        data["Label"] = y_true
        dataset_type = "Tabular"

    elif dataset_name == "Wine":
        from sklearn.datasets import load_wine
        raw = load_wine()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data["Label"] = raw.target
        dataset_type = "Tabular"

    elif dataset_name == "Iris":
        from sklearn.datasets import load_iris
        raw = load_iris()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data["Label"] = raw.target
        dataset_type = "Tabular"

    elif dataset_name == "Time Series":
        time = np.arange(0, 100)
        signal = np.sin(time) + 0.1 * np.random.randn(100)
        signal[20:25] = 3
        signal[60:65] = -3
        data = pd.DataFrame({"Time": time, "Signal": signal})
        dataset_type = "Time Series"

    elif dataset_name == "MNIST":
        from sklearn.datasets import load_digits
        raw = load_digits()
        data = pd.DataFrame(raw.data)
        data["Label"] = raw.target
        dataset_type = "Tabular"

    elif dataset_name == "KDDCup":
        X, _ = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        data = pd.DataFrame(X, columns=[f"Feature {i}" for i in range(1, 21)])
        data["Label"] = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
        dataset_type = "Tabular"

    elif dataset_name == "UCI Adult":
        raw = fetch_openml(name="adult", version=2, as_frame=True)
        df = raw.frame.copy()
        df = df.dropna()
        df["Label"] = df["class"]
        df.drop(columns=["class"], inplace=True)
        df_numeric = df.select_dtypes(include=["int64", "float64"])
        data = df_numeric.reset_index(drop=True)
        dataset_type = "Tabular"

    elif dataset_name == "Titanic":
        raw = fetch_openml(name="titanic", version=1, as_frame=True)
        df = raw.frame.copy()

        # 选择我们关心的数值列
        keep_cols = ["pclass", "age", "fare", "sibsp", "parch", "survived"]
        df = df[keep_cols]

        # 只对 age/fare 等数值列填补缺失值
        df["age"] = df["age"].fillna(df["age"].median())
        df["fare"] = df["fare"].fillna(df["fare"].median())

        df["Label"] = df["survived"]
        df.drop(columns=["survived"], inplace=True)

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        data = df[numeric_cols + ["Label"]].reset_index(drop=True)
        dataset_type = "Tabular"


    elif dataset_name == "Fashion MNIST":
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, y_train), (_, _) = fashion_mnist.load_data()

        # Flatten the 28x28 images to 784-dim vectors
        X_flat = X_train.reshape((X_train.shape[0], -1)) / 255.0
        data = pd.DataFrame(X_flat)
        data["Label"] = y_train
        dataset_type = "Image"

        # Optional: set features to all pixel columns
        features = data.columns[:-1].tolist()


    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return data, dataset_type


# def apply_pca_for_plotting(data, features):
#     st.warning("High-dimensional data detected. Reducing to 2D using PCA for visualization.")
#     X_proj = PCA(n_components=2).fit_transform(data.loc[:, features])
#     return X_proj[:, 0], X_proj[:, 1]



def apply_pca_for_plotting(data, features=None):
    from sklearn.decomposition import PCA

    if isinstance(data, np.ndarray):
        X = data
    elif isinstance(data, pd.DataFrame):
        if features is None or len(features) == 0:
            features = data.columns.tolist()
        X = data[features].values
    else:
        raise ValueError("Unsupported data type for PCA input")

    X_proj = PCA(n_components=2).fit_transform(X)
    return X_proj[:, 0], X_proj[:, 1]


def anomaly_detection_ui():
    st.header("🔍 Anomaly Detection")

    # Dataset selection
    dataset_groups = {
        "📊 Tabular / Image Datasets": ["Synthetic", "Wine", "Iris", "MNIST", "KDDCup", "UCI Adult", "Titanic", "Fashion MNIST"],
        "📈 Time Series Datasets": ["Time Series"]
    }

    category = st.selectbox("Choose Dataset Type", list(dataset_groups.keys()))
    dataset_name = st.selectbox("Select Dataset", dataset_groups[category])
    data, dataset_type = load_datasets(dataset_name)

    st.write(f"Original Data Shape: {data.shape}")
    st.write(f"Dataset Type: {dataset_type}")

    # if dataset_type == "Time Series":
    #     available_features = data.columns
    #     default_features = ["Signal"] if "Signal" in data.columns else list(available_features[:1])
    # else:
    #     available_features = data.columns[:-1]
    #     default_features = list(available_features[:2]) if len(available_features) >= 2 else list(available_features)
    if dataset_type == "Time Series":
        available_features = data.columns
        default_features = ["Signal"] if "Signal" in data.columns else list(available_features[:1])
    else:
        available_features = data.columns[:-1]
        if len(available_features) > 50:
            default_features = list(available_features)
        else:
            default_features = list(available_features[:2]) if len(available_features) >= 2 else list(available_features)


    features = st.multiselect("Select features for detection", available_features, default=default_features)
    
    use_pca = False
    if len(features) > 2:
        use_pca = st.checkbox("Apply PCA for visualization", value=True)


    if dataset_type == "Time Series" and "Time" in data.columns and len(features) == 1:
        st.subheader("📈 Raw Time Series Preview")
        st.line_chart(data.set_index("Time")[features[0]])

    X = data[features].values
    X = StandardScaler().fit_transform(X)

    # Method selection for anomaly detection
    method = st.selectbox("Choose Detection Method", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Point Anomaly", "Contextual Anomaly", "Duration Anomaly"])
#     method = st.selectbox("Choose Detection Method", [
#     "🔍 Isolation Forest", 
#     "🔍 One-Class SVM", 
#     "🔍 Local Outlier Factor", 
#     "📏 Point Anomaly (Z-Score)",
#     "🕒 Contextual Anomaly (Time-Series)", 
#     "⏱️ Duration Anomaly (Time-Series)"
# ])

    # Method explanations
    if method == "Isolation Forest":
        st.markdown("""
        **Isolation Forest**: 
        - This is an ensemble-based method that works by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature. The logic is that anomalies are few and different, so they can be isolated easily.
        """)
    elif method == "One-Class SVM":
        st.markdown("""
        **One-Class SVM**:
        - This is a variant of Support Vector Machine (SVM) used for anomaly detection in an unsupervised manner. It learns a decision function for a dataset with only one class and tries to map the data into a higher dimension to create a decision boundary.
        """)
    elif method == "Local Outlier Factor":
        st.markdown("""
        **Local Outlier Factor (LOF)**: 
        - This method evaluates the local density of data points. It compares the density of a point to that of its neighbors. A point with a significantly lower density compared to its neighbors is considered an anomaly.
        """)
    elif method == "Point Anomaly":
        st.markdown("""
        **Point Anomaly**:
        - This method detects single points in the data that are far away from the rest of the points. It is a simple yet effective technique for detecting rare outliers in high-dimensional datasets.
        """)
    elif method == "Contextual Anomaly":
        st.markdown("""
        **Contextual Anomaly**:
        - This type of anomaly detection works best for time series or sensor data, where the context of data points plays a significant role in their interpretation. A data point might be considered an anomaly only in the context of a specific period or condition.
        """)
    elif method == "Duration Anomaly":
        st.markdown("""
        **Duration Anomaly**:
        - This method applies specifically to time-series data where anomalies occur over a time period. It detects long-duration deviations or short-term bursts that are statistically rare.
        """)

    # Select anomaly detection method
    if method == "Isolation Forest":
        model = IsolationForest(contamination=0.1, random_state=42)
        preds = model.fit_predict(X)
    elif method == "One-Class SVM":
        model = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
        preds = model.fit_predict(X)
    elif method == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        preds = model.fit_predict(X)

    elif method == "Point Anomaly":
        threshold = st.slider("Z-score Threshold", min_value=2.0, max_value=5.0, step=0.1, value=3.0)
        z_scores = np.abs(zscore(X))
        max_z = np.max(z_scores, axis=1)
        preds = np.where(max_z > threshold, "Outlier", "Inlier")

        # # Compute Z-scores across features
        # z_scores = np.abs(zscore(X))  # shape: (n_samples, n_features)

        # # Reduce to single value per row (e.g., max z-score)
        # max_z = np.max(z_scores, axis=1)  # shape: (n_samples,)

        # # Label as outlier if any feature's z > 3
        # preds = np.where(max_z > 3, "Outlier", "Inlier")  # ✅ this is correct and final

    elif method == "Contextual Anomaly":
        if dataset_type == "Time Series":
            z_scores = np.abs(zscore(X))  # shape = (n_samples, n_features)
            max_z = np.max(z_scores, axis=1)  # shape = (n_samples,)
            preds = np.where(max_z > 2, "Outlier", "Inlier")
        else:
            st.error("Contextual anomaly detection is only applicable for time series data.")
            return

    elif method == "Duration Anomaly":
        if dataset_type == "Time Series":
            if 'Signal' in data.columns:
                signal = data['Signal'].values
                threshold = 2.0  # Example threshold for anomaly
                z_scores = zscore(signal)
                anomalies = np.where(np.abs(z_scores) > threshold, "Anomaly", "Normal")
                preds = anomalies
            else:
                st.error("Duration anomaly detection requires a 'Signal' column in time series data.")
                return
        else:
            st.error("Duration anomaly detection is only applicable for time series data.")
            return

    st.write(f"Shape of predictions: {preds.shape}")

    if len(preds) == len(data):
        try:
            data['Anomaly'] = pd.Series(preds, index=data.index)
        except ValueError as e:
            st.error(f"❌ Failed to assign 'Anomaly' column: {e}")
            return
    else:
        st.error(f"❌ Shape mismatch: preds shape {np.shape(preds)}, expected {data.shape[0]}")
        return
    # Safe assignment based on method
    # if method in ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]:
    #     if len(preds) == len(data):
    #         data['Anomaly'] = np.where(preds == -1, "Outlier", "Inlier")
    #     else:
    #         st.error("Prediction length does not match number of data rows.")
    #         return
    # elif method in ["Point Anomaly", "Contextual Anomaly", "Duration Anomaly"]:
    #     if len(preds) == len(data):
    #         st.write("🔍 Debug Info:")
    #         st.write(f"type(preds): {type(preds)}")
    #         st.write(f"preds.shape: {getattr(preds, 'shape', 'N/A')}")
    #         st.write(f"len(preds): {len(preds)}")
    #         st.write(f"data.shape: {data.shape}")
    #         st.code(f"First 5 preds: {preds[:5]}")

    #         try:
    #             # Safely assign preds to Anomaly column using aligned Series
    #             data['Anomaly'] = pd.Series(preds, index=data.index)
    #         except ValueError as e:
    #             st.error(f"❌ Failed to assign 'Anomaly' column: {e}")
    #             return
    #     else:
    #         st.error(f"❌ Shape mismatch: preds shape {np.shape(preds)}, expected {data.shape[0]}")
    #         return



    st.subheader("📈 Visualization")

    if dataset_type == "Time Series" and "Time" in data.columns and len(features) == 1:
        fig = px.line(data, x="Time", y=features[0], color='Anomaly', title=f"Anomaly Detection using {method}")
    else:
        # 🛡️ 防御：如果未选择 features，就自动选择数值列（排除标签列）
        if features is None or len(features) == 0:
            st.warning("No features selected. Using all numeric columns.")
            features = data.select_dtypes(include=["number"]).columns.drop("Label", errors="ignore").tolist()

        if len(features) > 2:
            x_vals, y_vals = apply_pca_for_plotting(data, features)
            x_label, y_label = "PCA 1", "PCA 2"
        elif len(features) == 2:
            x_vals, y_vals = data[features[0]], data[features[1]]
            x_label, y_label = features[0], features[1]
        else:
            st.error("Need at least 1 or 2 features for visualization.")
            st.stop()

        fig = px.scatter(
            x=x_vals,
            y=y_vals,
            color=data['Anomaly'],
            symbol=data['Anomaly'],
            title=f"Anomaly Detection using {method}",
            labels={"x": x_label, "y": y_label}
        )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Anomaly Counts")
    st.write(data['Anomaly'].value_counts())
