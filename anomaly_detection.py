# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM
# from sklearn.neighbors import LocalOutlierFactor
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px


# def anomaly_detection_ui():
#     st.header("🔍 Anomaly Detection")

#     # Sample datasets
#     dataset_name = st.selectbox("Select Dataset", ["Synthetic", "Wine", "Iris"])

#     if dataset_name == "Synthetic":
#         from sklearn.datasets import make_blobs
#         X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)
#         outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
#         X = np.vstack([X, outliers])
#         y_true = np.array([1]*300 + [-1]*20)
#         data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
#         data['Label'] = y_true

#     elif dataset_name == "Wine":
#         from sklearn.datasets import load_wine
#         raw = load_wine()
#         data = pd.DataFrame(raw.data, columns=raw.feature_names)
#         data['Label'] = raw.target

#     elif dataset_name == "Iris":
#         from sklearn.datasets import load_iris
#         raw = load_iris()
#         data = pd.DataFrame(raw.data, columns=raw.feature_names)
#         data['Label'] = raw.target

#     st.write(f"Original Data Shape: {data.shape}")
#     features = st.multiselect("Select features for detection", data.columns[:-1], default=list(data.columns[:2]))

#     X = data[features].values
#     X = StandardScaler().fit_transform(X)

#     method = st.selectbox("Choose Detection Method", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])

#     if method == "Isolation Forest":
#         model = IsolationForest(contamination=0.1, random_state=42)
#         preds = model.fit_predict(X)
#     elif method == "One-Class SVM":
#         model = OneClassSVM(nu=0.1, kernel="rbf", gamma='scale')
#         preds = model.fit_predict(X)
#     elif method == "Local Outlier Factor":
#         model = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
#         preds = model.fit_predict(X)

#     data['Anomaly'] = np.where(preds == -1, "Outlier", "Inlier")

#     st.subheader("📊 Visualization")
#     fig = px.scatter(data, x=features[0], y=features[1], color='Anomaly', symbol='Anomaly',
#                      title=f"Anomaly Detection using {method}")
#     st.plotly_chart(fig, use_container_width=True)

#     st.subheader("📋 Anomaly Counts")
#     st.write(data['Anomaly'].value_counts())



import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.datasets import make_classification, load_digits, fetch_openml
from sklearn.metrics import pairwise_distances
from scipy.stats import zscore
from sklearn.decomposition import PCA


def load_datasets(dataset_name):
    # Loading datasets
    if dataset_name == "Synthetic":
        from sklearn.datasets import make_blobs
        X, _ = make_blobs(n_samples=300, centers=1, cluster_std=0.6, random_state=42)
        outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
        X = np.vstack([X, outliers])
        y_true = np.array([1]*300 + [-1]*20)
        data = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        data['Label'] = y_true
        dataset_type = "Tabular"
        
    elif dataset_name == "Wine":
        from sklearn.datasets import load_wine
        raw = load_wine()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['Label'] = raw.target
        dataset_type = "Tabular"
        
    elif dataset_name == "Iris":
        from sklearn.datasets import load_iris
        raw = load_iris()
        data = pd.DataFrame(raw.data, columns=raw.feature_names)
        data['Label'] = raw.target
        dataset_type = "Tabular"
        
    elif dataset_name == "Time Series":
        # Generate synthetic time series data
        time = np.arange(0, 100)
        signal = np.sin(time) + 0.1 * np.random.randn(100)
        data = pd.DataFrame({"Time": time, "Signal": signal})
        # Introduce some anomalies
        data.iloc[20:25, 1] = 3  # Anomaly in the signal
        data.iloc[60:65, 1] = -3  # Another anomaly
        dataset_type = "Time Series"
        
    elif dataset_name == "MNIST":
        from sklearn.datasets import load_digits
        raw = load_digits()
        data = pd.DataFrame(raw.data)
        data['Label'] = raw.target
        dataset_type = "Tabular"
        
    elif dataset_name == "KDDCup":
        data = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        data = pd.DataFrame(data[0], columns=[f"Feature {i}" for i in range(1, 21)])
        data['Label'] = np.random.choice([0, 1], size=1000, p=[0.95, 0.05])
        dataset_type = "Tabular"
        
    elif dataset_name == "UCI Adult":
        data = fetch_openml(name="adult", version=2)
        df = data.frame
        df['Label'] = df['class']
        df.drop(columns=["class"], inplace=True)
        dataset_type = "Tabular"
        
    elif dataset_name == "Titanic":
        data = fetch_openml(name="titanic", version=1)
        df = data.frame
        df['Label'] = df['survived']
        df.drop(columns=["survived"], inplace=True)
        dataset_type = "Tabular"
        
    elif dataset_name == "Fashion MNIST":
        from tensorflow.keras.datasets import fashion_mnist
        (X_train, _), (_, _) = fashion_mnist.load_data()
        data = pd.DataFrame(X_train.reshape(X_train.shape[0], -1))  # Flatten images
        dataset_type = "Image"
        
    elif dataset_name == "Air Quality":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/Air%20Quality.zip"
        df = pd.read_csv(url, header=0, sep=";", decimal=",")
        df = df.dropna(axis=0, how="any")  # Remove rows with NaN
        dataset_type = "Time Series"
    
    return data, dataset_type


def anomaly_detection_ui():
    st.header("🔍 Anomaly Detection")

    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["Synthetic", "Wine", "Iris", "Time Series", "MNIST", "KDDCup", "UCI Adult", "Titanic", "Fashion MNIST", "Air Quality"])

    data, dataset_type = load_datasets(dataset_name)

    st.write(f"Original Data Shape: {data.shape}")
    st.write(f"Dataset Type: {dataset_type}")
    features = st.multiselect("Select features for detection", data.columns[:-1], default=list(data.columns[:2]))

    X = data[features].values
    X = StandardScaler().fit_transform(X)

    # Method selection for anomaly detection
    method = st.selectbox("Choose Detection Method", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor", "Point Anomaly", "Contextual Anomaly", "Duration Anomaly"])

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
        z_scores = np.abs(zscore(X))  # Simple point anomaly detection via Z-score
        preds = np.where(z_scores > 3, "Anomaly", "Normal")
    elif method == "Contextual Anomaly":
        if dataset_type == "Time Series":
            # Contextual anomaly detection for time series data (using sliding window approach)
            window_size = 10
            z_scores = np.abs(zscore(X))  # Z-score to detect deviations
            preds = np.where(z_scores > 2, "Anomaly", "Normal")
        else:
            st.error("Contextual anomaly detection is only applicable for time series data.")
            return
    elif method == "Duration Anomaly":
        if dataset_type == "Time Series":
            # Duration anomaly detection (for time-series only)
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

    data['Anomaly'] = np.where(preds == -1, "Outlier", "Inlier")

    st.subheader("📊 Visualization")
    fig = px.scatter(data, x=features[0], y=features[1], color='Anomaly', symbol='Anomaly',
                     title=f"Anomaly Detection using {method}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Anomaly Counts")
    st.write(data['Anomaly'].value_counts())


