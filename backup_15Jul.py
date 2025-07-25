# === Full Modular Streamlit App ===
# Split into: 
# - Supervised Learning (Regression / Classification)
# - Optimizer Playground
# Reuses all user-defined logic and keeps existing function structure

import streamlit as st

st.set_page_config(page_title="ML + Optimizer Visualizer", layout="wide")

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score, log_loss
)
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from sympy import symbols, lambdify
from io import BytesIO
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict


import os
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, SystemMessage
import openai

import streamlit as st
from langchain_community.chat_models import ChatOpenAI

# from local_llm import query_local_llm
import requests
# COLAB_LLM_URL = "https://c193262ee1c4.ngrok-free.app/generate"


COLAB_LLM_URL = "https://your-llm-app-name.loca.lt/generate" 

def query_local_llm(prompt):
    try:
        response = requests.post(COLAB_LLM_URL, json={"query": prompt})
        if response.ok:
            return response.json().get("response", "").strip()
        else:
            return f"⚠️ LLM API error: {response.status_code}"
    except Exception as e:
        return f"❌ Connection to LLM failed: {e}"
    
    
# client = openai  
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# llm = ChatOpenAI(
#     temperature=0.3,
#     openai_api_key=st.secrets["OPENAI_API_KEY"]
# )

# memory = ConversationBufferMemory()

# === Define shared query_llm for use across all modes ===
import os
from langchain.chat_models import ChatOpenAI

if "llm_backend" not in st.session_state:
    st.session_state.llm_backend = "OpenAI"  # default
if "openai_key" not in st.session_state:
    st.session_state.openai_key = os.getenv("OPENAI_API_KEY", "")  # fallback to environment

def query_llm(prompt: str) -> str:
    if st.session_state.llm_backend == "OpenAI" and st.session_state.openai_key:
        try:
            llm = ChatOpenAI(temperature=0.3, openai_api_key=st.session_state.openai_key)
            return llm.predict(prompt)
        except Exception as e:
            return f"❌ LLM Error: {e}"
    else:
        return "⚠️ LLM is not configured. Please check your API key or select OpenAI backend."


st.markdown("""
    <style>
    .stMarkdown h4 {
        margin-bottom: 0.2em;
    }
    .stRadio > div {
        margin-top: 0;
        margin-bottom: 0.5em;
    }
    .stSelectbox, .stTextInput, .stSlider {
        padding-top: 0;
        padding-bottom: 0;
        margin-bottom: 0.3em;
    }
    </style>
""", unsafe_allow_html=True)

# === UI Config ===
# st.set_page_config(page_title="ML + Optimizer Visualizer", layout="wide")
st.title("🧠 Machine Learning & Optimization Visual Explorer")

# === Top-level Mode Switch ===
mode = st.sidebar.radio("Choose Mode", ["📈 Supervised Learning", "🌋 Optimization Playground", "🧠 Unsupervised Learning", "🤖 LLM Assistant"])

# === Helper: Built-in Dataset Loader ===
def load_builtin_dataset(name):
    loaders = {
        "Iris": load_iris(as_frame=True),
        "Wine": load_wine(as_frame=True),
        "Breast Cancer": load_breast_cancer(as_frame=True)
    }
    data = loaders[name]
    df = data.frame.copy()
    df["target"] = data.target
    return df, data.target_names

# === Shared: Encode Target if Needed ===
def encode_target_column(df, target):
    if df[target].dtype == 'object' or df[target].dtype.name == 'category':
        df[target] = df[target].astype('category').cat.codes
        return df, df[target].astype('category').cat.categories.tolist()
    return df, None

# === Supervised Learning Section ===
if mode == "📈 Supervised Learning":
    st.subheader("📈 Supervised Learning Playground")
    st.sidebar.markdown("## Dataset Source")
    dataset_choice = st.sidebar.selectbox("Select Dataset", ["Iris", "Wine", "Breast Cancer", "Upload Your Own"])

    if dataset_choice != "Upload Your Own":
        df, label_names = load_builtin_dataset(dataset_choice)
        st.sidebar.success(f"Loaded **{dataset_choice}** dataset with shape {df.shape}")
        st.sidebar.markdown("#### 🧪 Sample Preview")
        st.sidebar.dataframe(df.head())

        if "target" in df.columns:
            st.sidebar.markdown("#### 🎯 Target Distribution")
            st.sidebar.write(df["target"].value_counts())
        
        st.sidebar.markdown("#### 🔍 Feature Correlation")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        st.sidebar.pyplot(fig)
    else:
        uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            label_names = None
        else:
            st.stop()

    st.dataframe(df.head())

    # Target + Feature selection
    st.sidebar.markdown("## Feature Selection")
    # Default target column if available
    default_target = "target" if "target" in df.columns else df.columns[0]
    target = st.sidebar.selectbox("🎯 Target Column", df.columns, index=df.columns.get_loc(default_target))

    df, target_labels = encode_target_column(df, target)
    feature_candidates = [col for col in df.columns if col != target]

    # Auto feature defaults based on dataset
    default_features = {
        "Iris": ["petal length (cm)", "petal width (cm)"],
        "Wine": ["alcohol", "malic_acid", "color_intensity"],
        "Breast Cancer": ["mean radius", "mean texture", "mean perimeter"]
    }
    initial_features = default_features.get(dataset_choice, [])
    features = st.sidebar.multiselect("🧩 Feature Columns", feature_candidates, default=initial_features)


    st.markdown("### 📊 Feature Distributions")
    st.caption("Boxplots by class (if categorical target) or histograms otherwise")

    n_cols = min(3, len(features))
    rows = (len(features) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for i in range(n_cols):
            idx = r * n_cols + i
            if idx < len(features):
                col = features[idx]
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(3.5, 3))
                    if df[target].nunique() < 10:
                        sns.boxplot(x=df[target], y=df[col], ax=ax)
                        ax.set_title(f"{col} by {target}", fontsize=10)
                    else:
                        sns.histplot(df[col], kde=True, ax=ax)
                        ax.set_title(f"{col} Distribution", fontsize=10)
                    st.pyplot(fig)
        
    if target and features:
        X = df[features].select_dtypes(include=[np.number])
        y = pd.to_numeric(df[target], errors="coerce")
        X = X[~y.isna()]
        y = y.dropna()
        y_class = y.round().astype(int)

        task_type = st.radio("Select Task", ["Linear Regression", "Logistic Regression", "Classification"])

        if task_type == "Linear Regression":
            tool = st.selectbox("Tool", ["Simple", "Polynomial", "Multi-Feature", "Diagnostics"])
            if tool == "Simple":
                if len(features) == 1:
                    feature = features[0]
                    X_sm = sm.add_constant(X[feature])
                    model = sm.OLS(y, X_sm).fit()
                    y_pred = model.predict(X_sm)
                    fig = plt.figure()
                    plt.scatter(X[feature], y, label="Actual")
                    plt.plot(X[feature], y_pred, label="Predicted", color="red")
                    plt.xlabel(feature)
                    plt.ylabel(target)
                    plt.title("Linear Fit")
                    plt.legend()
                    st.pyplot(fig)
                    st.text(model.summary())
                else:
                    st.warning("Please select exactly 1 feature.")

            elif tool == "Polynomial":
                if len(features) == 1:
                    feature = features[0]
                    degree = st.slider("Select Degree", 1, 5, 2)
                    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    poly_model.fit(X[[feature]], y)
                    y_pred = poly_model.predict(X[[feature]])
                    fig = plt.figure()
                    plt.scatter(X[feature], y, color="gray", label="Actual")
                    plt.plot(X[feature].sort_values(), y_pred[np.argsort(X[feature].values)], color="green", label="Poly Fit")
                    plt.legend()
                    plt.title(f"Polynomial Fit (degree {degree})")
                    st.pyplot(fig)
                    st.write("**MSE:**", mean_squared_error(y, y_pred))
                    st.write("**R² Score:**", r2_score(y, y_pred))
                else:
                    st.warning("Select only 1 feature.")

            elif tool == "Multi-Feature":
                model_type = st.radio("Model", ["Linear", "Ridge", "Lasso"], horizontal=True)
                alpha_val = st.slider("Regularization α", 0.0, 10.0, 0.0)
                X_const = sm.add_constant(X)
                if model_type == "Linear":
                    model = sm.OLS(y, X_const).fit()
                    y_pred = model.predict(X_const)
                    st.text(model.summary())
                elif model_type == "Ridge":
                    model = Ridge(alpha=alpha_val).fit(X, y)
                    y_pred = model.predict(X)
                else:
                    model = Lasso(alpha=alpha_val).fit(X, y)
                    y_pred = model.predict(X)
                fig = plt.figure()
                plt.scatter(y, y_pred, alpha=0.6)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.title(f"{model_type} Regression Results")
                st.pyplot(fig)

            elif tool == "Diagnostics":
                X_multi = sm.add_constant(X)
                model = sm.OLS(y, X_multi).fit()
                y_pred = model.predict(X_multi)
                residuals = y - y_pred
                rss = np.sum(residuals ** 2)
                rse = np.sqrt(rss / (len(y) - len(X.columns) - 1))
                st.write(f"**RSS:** {rss:.4f}")
                st.write(f"**RSE:** {rse:.4f}")
                fig1 = plt.figure()
                plt.scatter(y_pred, residuals)
                plt.axhline(0, color="red", linestyle="--")
                plt.xlabel("Fitted Values")
                plt.ylabel("Residuals")
                plt.title("Residuals vs Fitted")
                st.pyplot(fig1)

                fig2 = plt.figure()
                sns.histplot(residuals, kde=True)
                plt.title("Distribution of Residuals")
                st.pyplot(fig2)


        elif task_type == "Logistic Regression":
            C_val = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            max_iter_val = st.slider("Max Iterations", 100, 1000, 300)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            penalty = st.selectbox("Penalty", ["l2", "l1"])
            model = LogisticRegression(C=C_val, max_iter=max_iter_val, solver=solver, penalty=penalty)
            model.fit(X, y_class)
            y_pred = model.predict(X)
            st.metric("Accuracy", f"{accuracy_score(y_class, y_pred):.4f}")

            # Debug info
            st.write("Unique classes:", np.unique(y_class))

            # Ensure ROC only for binary
            if len(np.unique(y_class)) == 2:
                y_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_class, y_proba)
                roc_auc = roc_auc_score(y_class, y_proba)
                fig = plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.title("ROC Curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                st.pyplot(fig)
            else:
                st.info("ROC Curve only supported for binary classification.")
                
            fig = plt.figure()
            sns.heatmap(confusion_matrix(y_class, y_pred), annot=True, fmt="d", cmap="Blues")
            st.pyplot(fig)

            st.write("📋 Classification Report")
            st.dataframe(pd.DataFrame(classification_report(y_class, y_pred, output_dict=True)).transpose())



        elif task_type == "Classification":
            classifier = st.radio("Classifier", ["Naive Bayes", "Decision Tree", "K-Nearest Neighbors", "Random Forest", "MLP", "XGBoost",  "SVM"])
            if classifier == "Naive Bayes":
                model = GaussianNB()
                model.fit(X, y_class)
                y_pred = model.predict(X)

                acc = accuracy_score(y_class, y_pred)
                report = classification_report(y_class, y_pred, output_dict=True)
                labels = [f"Class {i}" for i in sorted(np.unique(y_class))]

                st.metric("Accuracy", f"{acc:.4f}")

                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn", 
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title(f"Confusion Matrix - Naive Bayes (Acc: {acc:.2f})")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # ROC/PR Curves
                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    y_proba = model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y_class, y_proba)
                    roc_auc = roc_auc_score(y_class, y_proba)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.title("ROC Curve")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.legend()
                    st.pyplot(fig)

                    # Precision-Recall
                    precision, recall, _ = precision_recall_curve(y_class, y_proba)
                    avg_prec = average_precision_score(y_class, y_proba)
                    fig_pr = plt.figure(figsize=(4, 3))
                    plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve")
                    plt.legend()
                    st.pyplot(fig_pr)

                elif hasattr(model, "predict_proba") and len(np.unique(y_class)) > 2:
                    classes = np.unique(y_class)
                    y_bin = label_binarize(y_class, classes=classes)
                    ovr_model = OneVsRestClassifier(model)
                    y_score = ovr_model.fit(X, y_class).predict_proba(X)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    for i in range(len(classes)):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_title("Multiclass ROC Curve")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)

                # Decision Boundary (if 2D)
                if X.shape[1] == 2:
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                        np.linspace(y_min, y_max, 200))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(plt.cm.Pastel2.colors))
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, cmap=ListedColormap(plt.cm.Dark2.colors), edgecolors="k")
                    ax.set_title("Naive Bayes Decision Boundary")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.grid(True)
                    ax.legend(handles=scatter.legend_elements()[0], labels=[f"Class {i}" for i in np.unique(y_class)], title="Classes")
                    st.pyplot(fig)

            elif classifier == "Decision Tree":
                max_depth = st.slider("Max Depth", 1, 20, 3)
                criterion = st.selectbox("Criterion", ["gini", "entropy"])
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)

                model.fit(X, y_class)
                y_pred = model.predict(X)

                # Scores
                acc = accuracy_score(y_class, y_pred)
                report = classification_report(y_class, y_pred, output_dict=True)
                labels = [f"Class {i}" for i in sorted(np.unique(y_class))]

                st.metric("Accuracy", f"{acc:.4f}")

                # Confusion Matrix Heatmap with Labels and Accuracy in Title
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", 
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title(f"Confusion Matrix - Decision Tree (Acc: {acc:.2f})")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

                # Classification report
                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # === ROC Curve (Binary only) ===
                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    y_proba = model.predict_proba(X)[:, 1]  # use probability for class 1
                    fpr, tpr, _ = roc_curve(y_class, y_proba)
                    roc_auc = roc_auc_score(y_class, y_proba)

                    fig_roc = plt.figure(figsize=(4, 3))
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.title("ROC Curve - Decision Tree")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve")
                    plt.legend()
                    st.pyplot(fig_roc)

                    st.markdown("""
                    📘 **Interpretation**:  
                    - AUC (Area Under Curve) close to 1.0 indicates a strong classifier.  
                    - ROC shows trade-off between sensitivity (TPR) and 1-specificity (FPR).  
                    """)
                elif hasattr(model, "predict_proba") and len(np.unique(y_class)) > 2:
                    classes = np.unique(y_class)
                    y_bin = label_binarize(y_class, classes=classes)
                    ovr_model = OneVsRestClassifier(model)
                    y_score = ovr_model.fit(X, y_class).predict_proba(X)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    for i in range(len(classes)):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_title("Multiclass ROC Curve")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)

                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    precision, recall, _ = precision_recall_curve(y_class, y_proba)
                    avg_prec = average_precision_score(y_class, y_proba)

                    fig_pr = plt.figure(figsize=(4, 3))
                    plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve")
                    plt.legend()
                    st.pyplot(fig_pr)


                # === Decision Boundary Plot (only if 2D) ===
                if X.shape[1] == 2:
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                        np.linspace(y_min, y_max, 200))

                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    bg_cmap = ListedColormap(plt.cm.Pastel2.colors[:len(np.unique(y_class))])
                    pt_cmap = ListedColormap(plt.cm.Dark2.colors[:len(np.unique(y_class))])

                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=bg_cmap)
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, edgecolors='k', cmap=pt_cmap)

                    ax.set_title("2D Decision Boundary")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.grid(True)

                    labels = [f"Class {i}" for i in np.unique(y_class)]
                    ax.legend(handles=scatter.legend_elements()[0], labels=labels, title="Classes")

                    st.pyplot(fig)

                    st.markdown("""
                    📘 **Interpretation**:  
                    - The shaded regions represent how the decision tree splits the input space.  
                    - Each color denotes a class region.  
                    - Dots represent actual samples — misclassified ones fall in the "wrong" region.  
                    """)
                else:
                    st.info("ℹ️ To show 2D decision boundaries, please select exactly **2 numeric features**.")


            elif classifier == "K-Nearest Neighbors":
                k = st.slider("Number of Neighbors (k)", 1, 20, 5)
                weights = st.selectbox("Weights", ["uniform", "distance"])
                model = KNeighborsClassifier(n_neighbors=k, weights=weights)
                model.fit(X, y_class)
                y_pred = model.predict(X)

                acc = accuracy_score(y_class, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")

                # 📊 Confusion Matrix
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                labels = [f"Class {i}" for i in np.unique(y_class)]
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", 
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix - KNN")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # 📋 Classification Report
                report = classification_report(y_class, y_pred, output_dict=True)
                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # 📈 ROC Curve
                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    y_proba = model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y_class, y_proba)
                    roc_auc = roc_auc_score(y_class, y_proba)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.title("ROC Curve")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.legend()
                    st.pyplot(fig)

                elif hasattr(model, "predict_proba") and len(np.unique(y_class)) > 2:
                    y_bin = label_binarize(y_class, classes=np.unique(y_class))
                    ovr_model = OneVsRestClassifier(model)
                    y_score = ovr_model.fit(X, y_class).predict_proba(X)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    for i in range(len(np.unique(y_class))):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_title("Multiclass ROC Curve")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)

                # 🔍 PR Curve (binary)
                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    precision, recall, _ = precision_recall_curve(y_class, y_proba)
                    avg_prec = average_precision_score(y_class, y_proba)
                    fig = plt.figure()
                    plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve")
                    plt.legend()
                    st.pyplot(fig)

                # 🌈 Decision Boundary (2D only)
                if X.shape[1] == 2:
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                        np.linspace(y_min, y_max, 200))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

                    bg_cmap = ListedColormap(plt.cm.Pastel1.colors[:len(np.unique(y_class))])
                    point_cmap = ListedColormap(plt.cm.Set1.colors[:len(np.unique(y_class))])

                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.contourf(xx, yy, Z, cmap=bg_cmap, alpha=0.5)
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, cmap=point_cmap, edgecolors='k')
                    ax.set_title("Decision Boundary - KNN")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.legend(handles=scatter.legend_elements()[0], 
                            labels=[f"Class {i}" for i in np.unique(y_class)],
                            title="Classes")
                    st.pyplot(fig)

            elif classifier == "MLP (Neural Network)":
                hidden_layer_sizes = st.text_input("Hidden Layers (e.g., 100 or 50,30)", "100")
                max_iter = st.slider("Max Iterations", 100, 1000, 300)
                alpha = st.slider("L2 Regularization (alpha)", 0.0001, 1.0, 0.001)
                try:
                    layer_tuple = tuple(map(int, hidden_layer_sizes.strip().split(",")))
                except:
                    layer_tuple = (100,)
                    st.warning("Invalid hidden layer format. Defaulted to (100,)")

                model = MLPClassifier(hidden_layer_sizes=layer_tuple, alpha=alpha, max_iter=max_iter, random_state=42)
                model.fit(X, y_class)
                y_pred = model.predict(X)
                acc = accuracy_score(y_class, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")

                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                labels = [f"Class {i}" for i in sorted(np.unique(y_class))]
                sns.heatmap(cm, annot=True, fmt="d", cmap="PuBuGn", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix - MLP")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

                # Classification Report
                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(classification_report(y_class, y_pred, output_dict=True)).transpose())

                # ROC & PR Curves
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X)

                    if len(np.unique(y_class)) == 2:
                        fpr, tpr, _ = roc_curve(y_class, y_proba[:, 1])
                        roc_auc = roc_auc_score(y_class, y_proba[:, 1])
                        fig = plt.figure()
                        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                        plt.plot([0, 1], [0, 1], "k--")
                        plt.title("ROC Curve")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.legend()
                        st.pyplot(fig)

                        precision, recall, _ = precision_recall_curve(y_class, y_proba[:, 1])
                        avg_prec = average_precision_score(y_class, y_proba[:, 1])
                        fig_pr = plt.figure()
                        plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
                        plt.title("Precision-Recall Curve")
                        plt.xlabel("Recall")
                        plt.ylabel("Precision")
                        plt.legend()
                        st.pyplot(fig_pr)

                    else:
                        classes = np.unique(y_class)
                        y_bin = label_binarize(y_class, classes=classes)
                        fig, ax = plt.subplots()
                        for i in range(len(classes)):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
                        ax.plot([0, 1], [0, 1], "k--")
                        ax.set_title("Multiclass ROC Curve")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.legend()
                        st.pyplot(fig)


                # 2D Decision Boundary
                if X.shape[1] == 2:
                    from matplotlib.colors import ListedColormap
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                        np.linspace(y_min, y_max, 200))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                    bg_cmap = ListedColormap(plt.cm.Pastel2.colors[:len(np.unique(y_class))])
                    pt_cmap = ListedColormap(plt.cm.Dark2.colors[:len(np.unique(y_class))])
                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=bg_cmap)
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, edgecolors="k", cmap=pt_cmap)
                    ax.set_title("Decision Boundary - MLP Classifier")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.legend(handles=scatter.legend_elements()[0], labels=labels, title="Classes")
                    st.pyplot(fig)

            elif classifier == "XGBoost":
                n_estimators = st.slider("Number of Estimators", 50, 300, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
                max_depth = st.slider("Max Depth", 1, 10, 3)

                model = XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    max_depth=max_depth)
                model.fit(X, y_class)
                y_pred = model.predict(X)

                acc = accuracy_score(y_class, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")
                report = classification_report(y_class, y_pred, output_dict=True)
                labels = [f"Class {i}" for i in sorted(np.unique(y_class))]

                # Confusion Matrix
                fig_cm, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="OrRd",
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title(f"Confusion Matrix - XGBoost (Acc: {acc:.2f})")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig_cm)

                # Classification Report
                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # ROC or PR Curves
                if hasattr(model, "predict_proba"):

                    unique_classes = np.unique(y_class)
                    if len(X) == 0 or len(unique_classes) == 0:
                        st.warning("No valid samples or classes for XGBoost ROC.")
                    elif len(unique_classes) == 2:
                        # Binary ROC
                        y_proba = model.predict_proba(X)[:, 1]
                        fpr, tpr, _ = roc_curve(y_class, y_proba)
                        roc_auc = roc_auc_score(y_class, y_proba)

                        fig = plt.figure(figsize=(4, 3))
                        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                        plt.plot([0, 1], [0, 1], "k--")
                        plt.title("ROC Curve")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.legend()
                        st.pyplot(fig)

                        # PR Curve
                        precision, recall, _ = precision_recall_curve(y_class, y_proba)
                        avg_prec = average_precision_score(y_class, y_proba)

                        fig_pr = plt.figure(figsize=(4, 3))
                        plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
                        plt.xlabel("Recall")
                        plt.ylabel("Precision")
                        plt.title("Precision-Recall Curve")
                        plt.legend()
                        st.pyplot(fig_pr)

                    elif len(unique_classes) > 2:
                        try:
                            y_bin = label_binarize(y_class, classes=unique_classes)
                            if y_bin.shape[1] < 2:
                                st.warning("Need at least 2 classes for ROC curve.")
                            else:
                                # 🔧 RE-define XGBoost model with num_class
                                model_ovr = XGBClassifier(use_label_encoder=False,
                                                        eval_metric="logloss",
                                                        n_estimators=n_estimators,
                                                        learning_rate=learning_rate,
                                                        max_depth=max_depth,
                                                        objective="multi:softprob",
                                                        num_class=len(unique_classes))

                                ovr_model = OneVsRestClassifier(model_ovr)
                                y_score = ovr_model.fit(X, y_class).predict_proba(X)

                                fig, ax = plt.subplots(figsize=(5, 4))
                                for i in range(len(unique_classes)):
                                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                                    roc_auc = auc(fpr, tpr)
                                    ax.plot(fpr, tpr, label=f"Class {unique_classes[i]} (AUC={roc_auc:.2f})")
                                ax.plot([0, 1], [0, 1], "k--")
                                ax.set_title("Multiclass ROC Curve")
                                ax.set_xlabel("False Positive Rate")
                                ax.set_ylabel("True Positive Rate")
                                ax.legend()
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Multiclass ROC skipped: {e}")

                        

                    else:
                        classes = np.unique(y_class)
                        y_bin = label_binarize(y_class, classes=classes)
                        ovr_model = OneVsRestClassifier(model)
                        y_score = ovr_model.fit(X, y_class).predict_proba(X)

                        fig_multi = plt.figure(figsize=(5, 4))
                        for i in range(len(classes)):
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                            roc_auc = auc(fpr, tpr)
                            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
                        plt.plot([0, 1], [0, 1], "k--")
                        plt.title("Multiclass ROC Curve")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.legend()
                        st.pyplot(fig_multi)

                # Feature Importance
                fig_imp, ax = plt.subplots(figsize=(5, 3))
                importances = model.feature_importances_
                ax.barh(X.columns, importances, color="teal")
                ax.set_title("Feature Importances - XGBoost")
                st.pyplot(fig_imp)

                # Decision Boundary (only for 2 features)
                if X.shape[1] == 2:
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                                        np.linspace(y_min, y_max, 300))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    bg = ListedColormap(plt.cm.Pastel1.colors[:len(np.unique(y_class))])
                    points = ListedColormap(plt.cm.Set1.colors[:len(np.unique(y_class))])
                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=bg)
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, cmap=points, edgecolors='k')
                    ax.set_title("2D Decision Boundary - XGBoost")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.grid(True)
                    labels = [f"Class {i}" for i in np.unique(y_class)]
                    ax.legend(handles=scatter.legend_elements()[0], labels=labels, title="Classes")
                    st.pyplot(fig)

            elif classifier == "Random Forest":
                st.subheader("🌲 Random Forest Classifier")

                # Parameter options
                enable_tuning = st.checkbox("Enable Auto-Tuning")
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 3, 5, 10],
                    'min_samples_split': [2, 5, 10]
                }

                base_model = RandomForestClassifier(random_state=42)

                if enable_tuning:
                    tuning_method = st.radio("Tuning Method", ["GridSearchCV", "RandomizedSearchCV"])
                    if tuning_method == "GridSearchCV":
                        model = GridSearchCV(base_model, param_grid, cv=3, scoring='accuracy')
                    else:
                        model = RandomizedSearchCV(base_model, param_distributions=param_grid, n_iter=10, cv=3, scoring='accuracy')
                else:
                    n_estimators = st.slider("n_estimators", 50, 300, 100)
                    max_depth = st.slider("max_depth", 1, 20, 5)
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

                model.fit(X, y_class)
                best_model = model.best_estimator_ if enable_tuning else model
                y_pred = best_model.predict(X)

                acc = accuracy_score(y_class, y_pred)
                st.metric("Accuracy", f"{acc:.4f}")
                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(classification_report(y_class, y_pred, output_dict=True)).transpose())

                # Confusion Matrix
                fig_cm, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                labels = [f"Class {i}" for i in sorted(np.unique(y_class))]
                sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrBr", xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title("Confusion Matrix - Random Forest")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig_cm)

                # Feature Importances
                fig_imp, ax = plt.subplots(figsize=(5, 3))
                importances = best_model.feature_importances_
                ax.barh(X.columns, importances, color="forestgreen")
                ax.set_title("Feature Importances - Random Forest")
                st.pyplot(fig_imp)

                # ROC (binary) or multiclass workaround
                if hasattr(best_model, "predict_proba"):
                    unique_classes = np.unique(y_class)
                    if len(unique_classes) == 2:
                        y_proba = best_model.predict_proba(X)[:, 1]
                        fpr, tpr, _ = roc_curve(y_class, y_proba)
                        roc_auc = auc(fpr, tpr)
                        fig_roc = plt.figure(figsize=(4, 3))
                        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                        plt.plot([0, 1], [0, 1], "k--")
                        plt.title("ROC Curve")
                        plt.xlabel("False Positive Rate")
                        plt.ylabel("True Positive Rate")
                        plt.legend()
                        st.pyplot(fig_roc)
                    elif len(unique_classes) > 2:
                        try:
                            y_bin = label_binarize(y_class, classes=unique_classes)
                            ovr_model = OneVsRestClassifier(best_model)
                            y_score = ovr_model.fit(X, y_class).predict_proba(X)

                            fig, ax = plt.subplots(figsize=(5, 4))
                            for i in range(len(unique_classes)):
                                fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                                roc_auc = auc(fpr, tpr)
                                ax.plot(fpr, tpr, label=f"Class {unique_classes[i]} (AUC={roc_auc:.2f})")
                            ax.plot([0, 1], [0, 1], "k--")
                            ax.set_title("Multiclass ROC Curve")
                            ax.set_xlabel("False Positive Rate")
                            ax.set_ylabel("True Positive Rate")
                            ax.legend()
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Multiclass ROC skipped: {e}")

                # Optional: Decision boundary for 2D
                if X.shape[1] == 2:
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                        np.linspace(y_min, y_max, 200))
                    Z = best_model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    cmap_bg = ListedColormap(plt.cm.Pastel2.colors[:len(np.unique(y_class))])
                    cmap_pts = ListedColormap(plt.cm.Dark2.colors[:len(np.unique(y_class))])
                    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.4)
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, cmap=cmap_pts, edgecolor="k")
                    ax.set_title("2D Decision Boundary - Random Forest")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.legend(handles=scatter.legend_elements()[0], labels=labels, title="Classes")
                    st.pyplot(fig)

            # elif classifier == "Random Forest":
            #     n_estimators = st.slider("Number of Trees", 10, 200, 100)
            #     max_depth = st.slider("Max Depth", 1, 20, 5)
            #     model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            #     model.fit(X, y_class)
            #     y_pred = model.predict(X)
            #     acc = accuracy_score(y_class, y_pred)
            #     st.metric("Accuracy", f"{acc:.4f}")

            #     # Confusion Matrix
            #     fig, ax = plt.subplots(figsize=(4, 3))
            #     cm = confusion_matrix(y_class, y_pred)
            #     labels = [f"Class {i}" for i in sorted(np.unique(y_class))]
            #     sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", xticklabels=labels, yticklabels=labels, ax=ax)
            #     ax.set_title("Confusion Matrix - Random Forest")
            #     ax.set_xlabel("Predicted Label")
            #     ax.set_ylabel("True Label")
            #     st.pyplot(fig)

            #     st.markdown("### 📋 Classification Report")
            #     st.dataframe(pd.DataFrame(classification_report(y_class, y_pred, output_dict=True)).transpose())

            #     # ROC or PR Curve
            #     if hasattr(model, "predict_proba"):
            #         y_proba = model.predict_proba(X)

            #         if len(np.unique(y_class)) == 2:
            #             # Binary ROC
            #             fpr, tpr, _ = roc_curve(y_class, y_proba[:, 1])
            #             roc_auc = roc_auc_score(y_class, y_proba[:, 1])
            #             fig = plt.figure()
            #             plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            #             plt.plot([0, 1], [0, 1], "k--")
            #             plt.xlabel("False Positive Rate")
            #             plt.ylabel("True Positive Rate")
            #             plt.title("ROC Curve")
            #             plt.legend()
            #             st.pyplot(fig)

            #             # PR Curve
            #             precision, recall, _ = precision_recall_curve(y_class, y_proba[:, 1])
            #             avg_prec = average_precision_score(y_class, y_proba[:, 1])
            #             fig_pr = plt.figure()
            #             plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
            #             plt.xlabel("Recall")
            #             plt.ylabel("Precision")
            #             plt.title("Precision-Recall Curve")
            #             plt.legend()
            #             st.pyplot(fig_pr)

            #         else:
            #             # Multiclass ROC (One-vs-Rest)
            #             classes = np.unique(y_class)
            #             y_bin = label_binarize(y_class, classes=classes)
            #             fig, ax = plt.subplots()
            #             for i in range(len(classes)):
            #                 fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            #                 roc_auc = auc(fpr, tpr)
            #                 ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
            #             ax.plot([0, 1], [0, 1], "k--")
            #             ax.set_title("Multiclass ROC Curve")
            #             ax.set_xlabel("False Positive Rate")
            #             ax.set_ylabel("True Positive Rate")
            #             ax.legend()
            #             st.pyplot(fig)

            #     # Feature Importance
            #     st.markdown("### 🔍 Feature Importance")
            #     importances = model.feature_importances_
            #     sorted_idx = np.argsort(importances)[::-1]
            #     fig_imp = plt.figure(figsize=(5, 3))
            #     plt.bar(range(len(importances)), importances[sorted_idx], align="center")
            #     plt.xticks(range(len(importances)), [X.columns[i] for i in sorted_idx], rotation=45)
            #     plt.ylabel("Importance")
            #     plt.title("Feature Importance")
            #     st.pyplot(fig_imp)

            #     # 2D Decision Boundary
            #     if X.shape[1] == 2:
            #         from matplotlib.colors import ListedColormap
            #         x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
            #         y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
            #         xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
            #                             np.linspace(y_min, y_max, 200))
            #         Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
            #         bg_cmap = ListedColormap(plt.cm.Pastel1.colors[:len(np.unique(y_class))])
            #         pt_cmap = ListedColormap(plt.cm.Dark2.colors[:len(np.unique(y_class))])
            #         fig, ax = plt.subplots(figsize=(4, 3))
            #         ax.contourf(xx, yy, Z, alpha=0.5, cmap=bg_cmap)
            #         scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, edgecolors="k", cmap=pt_cmap)
            #         ax.set_title("Decision Boundary - Random Forest")
            #         ax.set_xlabel(X.columns[0])
            #         ax.set_ylabel(X.columns[1])
            #         ax.legend(handles=scatter.legend_elements()[0], labels=labels, title="Classes")
            #         st.pyplot(fig)

            else:
                C_val = st.slider("C", 0.01, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                gamma = st.selectbox("Gamma", ["scale", "auto"])
                model = SVC(C=C_val, kernel=kernel, gamma=gamma, probability=True)

                model.fit(X, y_class)
                y_pred = model.predict(X)

                acc = accuracy_score(y_class, y_pred)
                report = classification_report(y_class, y_pred, output_dict=True)
                labels = [f"Class {i}" for i in sorted(np.unique(y_class))]

                st.metric("Accuracy", f"{acc:.4f}")

                # Confusion Matrix
                fig, ax = plt.subplots(figsize=(4, 3))
                cm = confusion_matrix(y_class, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", 
                            xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title(f"Confusion Matrix - SVM (Acc: {acc:.2f})")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                st.pyplot(fig)

                st.markdown("### 📋 Classification Report")
                st.dataframe(pd.DataFrame(report).transpose())

                # ROC Curve (binary only)
                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    y_proba = model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y_class, y_proba)
                    roc_auc = roc_auc_score(y_class, y_proba)

                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("ROC Curve (SVM)")
                    plt.legend()
                    st.pyplot(fig)

                elif hasattr(model, "predict_proba") and len(np.unique(y_class)) > 2:
                    classes = np.unique(y_class)
                    y_bin = label_binarize(y_class, classes=classes)
                    ovr_model = OneVsRestClassifier(model)
                    y_score = ovr_model.fit(X, y_class).predict_proba(X)

                    fig, ax = plt.subplots(figsize=(5, 4))
                    for i in range(len(classes)):
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")
                    ax.plot([0, 1], [0, 1], "k--")
                    ax.set_title("Multiclass ROC Curve")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    st.pyplot(fig)

                if hasattr(model, "predict_proba") and len(np.unique(y_class)) == 2:
                    precision, recall, _ = precision_recall_curve(y_class, y_proba)
                    avg_prec = average_precision_score(y_class, y_proba)

                    fig_pr = plt.figure(figsize=(4, 3))
                    plt.plot(recall, precision, label=f"Avg Precision = {avg_prec:.2f}")
                    plt.xlabel("Recall")
                    plt.ylabel("Precision")
                    plt.title("Precision-Recall Curve")
                    plt.legend()
                    st.pyplot(fig_pr)


                # Decision Boundary (2D only)
                if X.shape[1] == 2:
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                        np.linspace(y_min, y_max, 200))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)

                    fig, ax = plt.subplots(figsize=(4, 3))
                    ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(plt.cm.Pastel1.colors))
                    scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class,
                                        cmap=ListedColormap(plt.cm.Set1.colors), edgecolors='k')
                    ax.set_title("SVM Decision Boundary")
                    ax.set_xlabel(X.columns[0])
                    ax.set_ylabel(X.columns[1])
                    ax.grid(True)
                    ax.legend(handles=scatter.legend_elements()[0],
                            labels=[f"Class {i}" for i in np.unique(y_class)], title="Classes")
                    st.pyplot(fig)

                    st.markdown("""
                    📘 **Interpretation**:  
                    - SVM separates classes using a maximum margin hyperplane.  
                    - This plot shows the decision regions predicted by SVM.  
                    - Colored background = predicted class.  
                    - Dots = actual data points.  
                    - Overlaps mean misclassification or boundary limitations.
                    """)



# === Optimizer Playground ===
elif mode == "🌋 Optimization Playground":
    # Inserted full optimizer playground block here
    import sympy as sp
    import time
    from sympy import symbols, lambdify
    from io import BytesIO
    from PIL import Image
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D


    st.title("🧪 Optimizer Visual Playground")

    x, y, w = sp.symbols("x y w")
    x_sym, y_sym, w_sym = sp.symbols("x y w")

    predefined_funcs = {
        "Quadratic Bowl": (x**2 + y**2, [], "Convex bowl, global min at origin."),
        "Saddle": (x**2 - y**2, [], "Saddle point at origin, non-convex."),
        "Rosenbrock": ((1 - x)**2 + 100 * (y - x**2)**2, [], "Banana-shaped curved valley, classic test function."),
        "Constrained Circle": (x * y, [x + y - 1], "Constrained optimization with line x + y = 1."),
        "Double Constraint": (x**2 + y**2, [x + y - 1, x**2 + y**2 - 4], "Circular + linear intersection constraints."),
        "Multi-Objective": (w * ((x - 1)**2 + (y - 2)**2) + (1 - w) * ((x + 2)**2 + (y + 1)**2), [], "Weighted sum of two loss terms."),
        "Ackley": (-20*sp.exp(-0.2*sp.sqrt(0.5*(x**2 + y**2))) - sp.exp(0.5*(sp.cos(2*sp.pi*x) + sp.cos(2*sp.pi*y))) + sp.E + 20, [], "Multimodal non-convex function."),
        "Rastrigin": (20 + x**2 - 10*sp.cos(2*sp.pi*x) + y**2 - 10*sp.cos(2*sp.pi*y), [], "Many local minima, non-convex."),
        "Styblinski-Tang": (0.5*((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y)), [], "Non-convex with multiple minima."),
        "Sphere": (x**2 + y**2, [], "Simple convex function."),
        "Himmelblau": ((x**2 + y - 11)**2 + (x + y**2 - 7)**2, [], "Multiple global minima, non-convex."),
        "Booth": ((x + 2*y - 7)**2 + (2*x + y - 5)**2, [], "Simple convex function."),
        "Beale": ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2, [], "Non-convex with multiple minima.")

    }

            
    with st.expander("💬 Ask the LLM about Optimizers or Math", expanded=False):
        user_question = st.text_input("Ask anything (e.g., What is BFGS? Why is Newton unstable?)")
        if user_question:
            with st.spinner("🤖 Thinking..."):
                full_prompt = (
                    "You are an expert on numerical optimization methods such as Gradient Descent, "
                    "Adam, Newton’s Method, Simulated Annealing, and Genetic Algorithms.\n\n"
                    f"Question: {user_question}\nAnswer:"
                )
                response = query_llm(full_prompt)
                st.markdown(
                    f"<div style='background-color:#f4f9ff;padding:10px;border-radius:6px;'>{response}</div>",
                    unsafe_allow_html=True,
                )

    with st.expander("🧠 Taylor Series & Optimizer Foundations", expanded=True):
        st.markdown("### 📚 How Taylor Series Explains Optimizers")

        st.markdown("Many optimization methods are based on the **Taylor series expansion** of a function. This helps us approximate the function locally using its derivatives:")

        st.latex(r"""
        f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
        """)
        st.markdown("- This is the **1st-order Taylor expansion**, which is the basis of **Gradient Descent**. It only uses the slope (gradient) to decide the direction to move.")

        st.latex(r"""
        f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
        """)
        st.markdown("- This is the **2nd-order Taylor expansion**, used in **Newton's Method**. It adds the **Hessian** (curvature) to adjust the step size and improve convergence.")

        st.markdown("### ✍️ Summary")
        st.markdown("""
        You can imagine it like this:
        - ✅ **1st-order**: Approximates with a **tangent line** (local slope).
        - ✅ **2nd-order**: Approximates with a **parabola** (slope + curvature).

        This helps students connect abstract equations with **optimizer logic**.
                        
        - **Gradient Descent** is a 1st-order method → uses slope only.
        - **Newton's Method** is a 2nd-order method → uses slope and curvature.
        - Understanding these expansions builds strong intuition about how optimizers move in the loss landscape.
        """)

        # === 🧠 Interactive Taylor expansion plot ===
        # try:
        #     a = st.slider("Choose expansion point x = a", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

        #     x = np.linspace(-3, 3, 500)
        #     fx = np.cos(x)

        #     # Values at x = a
        #     f_a = np.cos(a)
        #     f1_a = -np.sin(a)  # First derivative at a
        #     f2_a = -np.cos(a)  # Second derivative at a

        #     # Taylor approximations
        #     taylor_1 = f_a + f1_a * (x - a)  # 1st-order (linear)
        #     taylor_2 = f_a + f1_a * (x - a) + 0.5 * f2_a * (x - a)**2  # 2nd-order (parabola)

        #     fig, ax = plt.subplots(figsize=(8, 5))
        #     ax.plot(x, fx, label=r'$f(x) = \cos x$', linewidth=2, color='blue')
        #     ax.plot(x, taylor_1, label='1st-order (linear)', linestyle='--', color='red')
        #     ax.plot(x, taylor_2, label='2nd-order (parabola)', linestyle='--', color='orange')
        #     ax.scatter(a, f_a, color='black', zorder=5)
        #     ax.axhline(0, color='black', linewidth=0.8)
        #     ax.axvline(a, color='gray', linestyle=':')
        #     ax.set_title(f'✅ LIVE: Taylor Approximations of $\\cos x$ at $x = {a}$')
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('f(x)')
        #     ax.legend()
        #     ax.grid(True)
        #     ax.set_ylim(-2, 2)

        #     st.pyplot(fig)

        # except Exception as e:
        #     st.error(f"Plot rendering failed: {e}")

        try:
            func_choice = st.selectbox("Choose a function to expand:", ["cos(x)", "exp(x)", "ln(1+x)", "tanh(x)"])
            show_3rd_4th = st.checkbox("➕ Show 3rd and 4th-order terms", value=False)
            show_linear = st.checkbox("Show 1st-order (linear)", value=True)
            show_parabola = st.checkbox("Show 2nd-order (parabola)", value=True)

            def get_function_package(choice):
                x = sp.symbols('x')
                if choice == "cos(x)":
                    f_sym = sp.cos(x)
                    domain = (-3, 3)
                elif choice == "exp(x)":
                    f_sym = sp.exp(x)
                    domain = (-3, 3)
                elif choice == "ln(1+x)":
                    f_sym = sp.ln(1 + x)
                    domain = (-0.9, 3)
                elif choice == "tanh(x)":
                    f_sym = sp.tanh(x)
                    domain = (-3, 3)
                return f_sym, domain

            # Symbolic expression and Taylor expansion
            f_sym, (xmin, xmax) = get_function_package(func_choice)
            x_sym, a_sym = sp.symbols('x a')
            f_taylor = f_sym.series(x_sym, x0=a_sym, n=5).removeO()
            
            # Compute symbolic derivatives at x = a
            x_sym, a_sym = sp.symbols('x a')
            h = x_sym - a_sym

            f1_sym = sp.diff(f_sym, x_sym)
            f2_sym = sp.diff(f_sym, x_sym, 2)
            f3_sym = sp.diff(f_sym, x_sym, 3)
            f4_sym = sp.diff(f_sym, x_sym, 4)

            # Taylor terms at x = a
            T1 = f_sym.subs(x_sym, a_sym) + f1_sym.subs(x_sym, a_sym) * h
            T2 = T1 + (1/2) * f2_sym.subs(x_sym, a_sym) * h**2
            T4 = T2 + (1/6) * f3_sym.subs(x_sym, a_sym) * h**3 + (1/24) * f4_sym.subs(x_sym, a_sym) * h**4

            # Display formulas
            st.markdown("#### 🧮 Symbolic Taylor Expansion at $x = a$")
            st.markdown("**1st-order (Gradient Descent basis):**")
            st.latex(f"f(x) \\approx {sp.latex(T1)}")

            st.markdown("**2nd-order (Newton's Method basis):**")
            st.latex(f"f(x) \\approx {sp.latex(T2)}")

            if show_3rd_4th:
                st.markdown("**Full Expansion (up to 4th-order):**")
                st.latex(f"f(x) \\approx {sp.latex(T4)}")


            # Get lambdified numeric functions
            f_np = sp.lambdify(x_sym, f_sym, "numpy")
            f1_np = sp.lambdify(x_sym, sp.diff(f_sym, x_sym), "numpy")
            f2_np = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 2), "numpy")
            f3_np = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 3), "numpy")
            f4_np = sp.lambdify(x_sym, sp.diff(f_sym, x_sym, 4), "numpy")

            # UI slider
            a = st.slider("Choose expansion point x = a", min_value=float(xmin), max_value=float(xmax), value=0.0, step=0.1)

            x = np.linspace(xmin, xmax, 500)
            f_a, f1_a, f2_a = f_np(a), f1_np(a), f2_np(a)
            f3_a = f3_np(a) if show_3rd_4th else 0
            f4_a = f4_np(a) if show_3rd_4th else 0

            # Build Taylor series
            taylor_1 = f_a + f1_a * (x - a)
            taylor_2 = taylor_1 + 0.5 * f2_a * (x - a)**2
            taylor_4 = taylor_2 + (1/6) * f3_a * (x - a)**3 + (1/24) * f4_a * (x - a)**4

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, f_np(x), label=fr'$f(x) = {func_choice}$', linewidth=2, color='blue')
            if show_linear:
                ax.plot(x, taylor_1, label='1st-order (linear)', linestyle='--', color='red')
            if show_parabola:
                ax.plot(x, taylor_2, label='2nd-order (parabola)', linestyle='--', color='orange')
            if show_3rd_4th:
                ax.plot(x, taylor_4, label='3rd/4th-order', linestyle='--', color='green')

            ax.scatter(a, f_a, color='black', zorder=5)
            ax.axhline(0, color='black', linewidth=0.8)
            ax.axvline(a, color='gray', linestyle=':')
            ax.set_title(f'✅ LIVE: Taylor Approximations of {func_choice} at $x = {a}$')
            ax.set_xlabel('x')
            ax.set_ylabel('f(x)')
            ax.legend()
            ax.grid(True)
            ax.set_ylim(np.nanmin(f_np(x)) - 1, np.nanmax(f_np(x)) + 1)

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Rendering failed: {e}")


    # 🪄 Optimizer Category Info Block (Outside main expander)
    with st.expander("🧠 Optimizer Category Info & Usage Tips", expanded=False):
        st.markdown("""
        ### 🤖 Optimizer Categories

        **Gradient-based Optimizers** use derivatives (gradients) to guide updates:

        - **GradientDescent**: Basic steepest descent using learning rate.
        - **Adam**: Adaptive learning rates with momentum (recommended for noisy or sparse gradients).
        - **RMSProp**: Like Adam but simpler; adjusts step size based on recent gradient magnitudes.
        - **Newton's Method**: Uses second-order derivatives (Hessian) for faster convergence, but may be unstable near saddle points.

        **Heuristic Optimizers** use stochastic or evolutionary strategies:

        - **Simulated Annealing**: Explores search space with temperature-based random steps. Good for escaping local minima.
        - **Genetic Algorithm**: Population-based method inspired by natural evolution. Effective for complex, non-differentiable functions.

        ---
        ### 🎮 How to Use the Playground

        - **1. Choose a function** from the dropdown.
        - **2. Select an optimizer** and tune parameters like learning rate or mutation strength.
        - **3. Try enabling _Auto-Tune_** to automatically pick good learning rate + steps.
        - **4. Adjust initial x/y** starting positions to see how descent paths change.
        - **5. Toggle _"Animate Descent"_** to visualize how the optimizer moves step-by-step.
        - **6. Compare optimizers** in the **"Diagnostic Tools"** section using convergence plots and summary tables.

        > 🧪 Try Newton's Method on **Quadratic Bowl** for fast convergence, or test **Genetic Algorithm** on **Rastrigin** to explore multimodal search.

        ---
        ⚠️ *Note:* Gradient-based methods require a smooth function. Use heuristic optimizers for discontinuous or non-differentiable objectives.
        """)



    with st.expander("🚀 Optimizer Visual Playground", expanded=True):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            mode = st.radio("Function Source", ["Predefined", "Custom"])
            func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
            expr_str = st.text_input("Enter function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""
            w_val = st.slider("Weight w (Multi-Objective)", 0.0, 1.0, 0.5) if func_name == "Multi-Objective" else None

            optimizers = ["GradientDescent", "Adam", "RMSProp", "Newton's Method", "Simulated Annealing", "Genetic Algorithm"]
            optimizer = st.selectbox("Optimizer", optimizers)
            

            options = {}

            if optimizer == "Newton's Method":
                newton_variant = st.selectbox("Newton Variant", ["Classic Newton", "Numerical Newton", "BFGS", "L-BFGS"])
                options["newton_variant"] = newton_variant
            elif optimizer == "Simulated Annealing":
                options["T"] = st.slider("Initial Temperature (T)", 0.1, 10.0, 2.0)
                options["cooling"] = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
            elif optimizer == "Genetic Algorithm":
                options["pop_size"] = st.slider("Population Size", 10, 100, 20)
                options["mutation_std"] = st.slider("Mutation Std Dev", 0.1, 1.0, 0.3)

            auto_tune = False

            if optimizer in ["GradientDescent", "Adam", "RMSProp"]:
                if optimizer == "GradientDescent":
                    use_backtracking = st.checkbox("🔍 Use Backtracking Line Search", value=False)
                    options["use_backtracking"] = use_backtracking

                    if use_backtracking:
                        st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
                        auto_tune = False
                        if "auto_tune_checkbox" in st.session_state:
                            st.session_state["auto_tune_checkbox"] = False
                        st.caption("ℹ️ Disabled because backtracking search dynamically adjusts step size.")
                    else:
                        auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")
                else:
                    auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")

            elif optimizer == "Newton's Method":
                st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
                auto_tune = False
                if "auto_tune_checkbox" in st.session_state:
                    st.session_state["auto_tune_checkbox"] = False
                st.caption("ℹ️ Auto-tune is not applicable to Newton’s Method.")
                
            start_xy_defaults = {
                "Quadratic Bowl": (-3.0, 3.0), "Saddle": (-2.0, 2.0), "Rosenbrock": (-1.5, 1.5),
                "Constrained Circle": (0.5, 0.5), "Double Constraint": (-1.5, 1.5),
                "Multi-Objective": (0.0, 0.0), "Ackley": (2.0, -2.0), "Rastrigin": (3.0, 3.0),
                "Styblinski-Tang": (-2.5, -2.5), "Sphere": (-3.0, 3.0), "Himmelblau": (0.0, 0.0),
                "Booth": (1.0, 1.0), "Beale": (-2.0, 2.0)
            }
            default_x, default_y = start_xy_defaults.get(func_name, (-3.0, 3.0))
            default_lr = 0.005
            default_steps = 50

            def run_auto_tuning_simulation(f_func, optimizer, x0, y0):
                lr_grid = list(np.logspace(-4, -1, 6))
                step_grid = [20, 30, 40, 50, 60, 80]
                best_score = float("inf")
                best_lr, best_steps = lr_grid[0], step_grid[0]
                logs = []

                for lr in lr_grid:
                    for steps in step_grid:
                        x_t, y_t = x0, y0
                        m, v = np.zeros(2), np.zeros(2)
                        beta1, beta2, eps = 0.9, 0.999, 1e-8
                        for t in range(1, steps + 1):
                            dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
                            dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
                            grad = np.array([dx, dy])
                            if np.linalg.norm(grad) < 1e-3:
                                break
                            if optimizer == "Adam":
                                m = beta1 * m + (1 - beta1) * grad
                                v = beta2 * v + (1 - beta2) * (grad ** 2)
                                m_hat = m / (1 - beta1 ** t)
                                v_hat = v / (1 - beta2 ** t)
                                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                            elif optimizer == "RMSProp":
                                v = beta2 * v + (1 - beta2) * (grad ** 2)
                                update = lr * grad / (np.sqrt(v) + eps)
                            else:
                                update = lr * grad
                            x_t -= update[0]
                            y_t -= update[1]
                        loss = f_func(x_t, y_t)
                        score = loss + 0.01 * steps
                        logs.append({"lr": lr, "steps": steps, "loss": loss, "score": score})
                        if score < best_score:
                            best_score, best_lr, best_steps = score, lr, steps
                st.session_state.df_log = pd.DataFrame(logs)
                return best_lr, best_steps

            if auto_tune:
                # Use fresh sympy symbols to avoid conflict with Taylor's x
                x_sym, y_sym, w_sym = sp.symbols("x y w")
                symbolic_expr = predefined_funcs[func_name][0]

                if func_name == "Multi-Objective" and w_val is not None:
                    symbolic_expr = symbolic_expr.subs(w_sym, w_val)

                f_lambdified = sp.lambdify((x_sym, y_sym), symbolic_expr, modules="numpy")

                best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)
                st.success(f"✅ Auto-tuned: lr={best_lr}, steps={best_steps}, start=({default_x},{default_y})")
                default_lr, default_steps = best_lr, best_steps

            # Set in session_state
            if 'params_set' not in st.session_state or st.button("🔄 Reset to Auto-Tuned"):
                st.session_state.lr = default_lr
                st.session_state.steps = default_steps
                st.session_state.start_x = default_x
                st.session_state.start_y = default_y
                st.session_state.params_set = True

            # Final user inputs

            # ✅ Only show lr and steps when not using backtracking or Newton
            if not (
                optimizer == "GradientDescent" and options.get("use_backtracking", False)
            ) and optimizer != "Newton's Method":
                lr = st.selectbox("Learning Rate", sorted(set([0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, default_lr])), index=0, key="lr")
                steps = st.slider("Steps", 10, 100, value=st.session_state.get("steps", 50), key="steps")
            elif optimizer == "Newton's Method":
                lr = None
                steps = None
                st.info("📌 Newton’s Method computes its own step size using the Hessian inverse — learning rate is not needed.")
            elif optimizer == "GradientDescent" and options.get("use_backtracking", False):
                lr = None
                steps = None
                st.info("📌 Using Backtracking Line Search — no need to set learning rate or step count.")



            # Ensure session_state values exist to avoid AttributeError
            if "start_x" not in st.session_state:
                st.session_state["start_x"] = -3.0
            if "start_y" not in st.session_state:
                st.session_state["start_y"] = 3.0


            st.slider("Initial x", -5.0, 5.0, st.session_state.start_x, key="start_x")
            st.slider("Initial y", -5.0, 5.0, st.session_state.start_y, key="start_y")
            # st.checkbox("🎮 Animate Descent Steps")
            show_animation = st.checkbox("🎮 Animate Descent Steps", key="show_animation")

        if auto_tune and optimizer in ["GradientDescent", "Adam", "RMSProp"]:
            with col_right:
                st.markdown("### 📊 Auto-Tuning Trial Log")
                if "df_log" in st.session_state:
                    st.dataframe(st.session_state.df_log.sort_values("score").reset_index(drop=True))
                    st.markdown("""
                    **🧠 How to Read Score:**
                    - `score = final_loss + penalty × steps`
                    - ✅ Lower score is better (fast and accurate convergence).
                    """)
                else:
                    st.info("Auto-tuning not yet triggered.")
                #             st.info("Auto-tuning not yet triggered.")


        if mode == "Predefined":
            f_expr, constraints, description = predefined_funcs[func_name]
            if func_name == "Multi-Objective" and w_val is not None:
                f_expr = f_expr.subs(w, w_val)
        else:
            try:
                f_expr = sp.sympify(expr_str)
                constraints = []
                description = "Custom function."
            except:
                st.error("Invalid expression.")
                st.stop()



        st.markdown(f"### 📘 Function Description:\n> {description}")

        # Setup for symbolic Lagrangian and KKT (if needed)
        L_expr = f_expr + sum(sp.Symbol(f"lambda{i+1}") * g for i, g in enumerate(constraints))
        # grad_L = [sp.diff(L_expr, v) for v in (x, y)]
        grad_L = [sp.diff(L_expr, v) for v in (x_sym, y_sym)]
        kkt_conditions = grad_L + constraints

        def backtracking_line_search_sym(f_sym, grad_f_sym, x0, y0, alpha0=1.0, beta=0.5, c=1e-4, max_iters=100):
            # Declare standard symbols for lambdify
            x, y = sp.symbols("x y")

            # Ensure all expressions use x, y (not x_sym, y_sym)
            f_sym = f_sym.subs({sp.Symbol("x"): x, sp.Symbol("y"): y})
            grad_f_sym = [g.subs({sp.Symbol("x"): x, sp.Symbol("y"): y}) for g in grad_f_sym]

            f_lambd = sp.lambdify((x, y), f_sym, modules='numpy')
            grad_f_lambd = [sp.lambdify((x, y), g, modules='numpy') for g in grad_f_sym]

            xk, yk = x0, y0
            path = [(xk, yk)]
            alphas = []

            for _ in range(max_iters):
                gx, gy = grad_f_lambd[0](xk, yk), grad_f_lambd[1](xk, yk)
                grad_norm = gx**2 + gy**2

                if grad_norm < 1e-10:
                    break

                alpha = alpha0
                while True:
                    x_new = xk - alpha * gx
                    y_new = yk - alpha * gy
                    lhs = f_lambd(x_new, y_new)
                    rhs = f_lambd(xk, yk) - c * alpha * grad_norm
                    if lhs <= rhs or alpha < 1e-8:
                        break
                    alpha *= beta

                xk, yk = xk - alpha * gx, yk - alpha * gy
                path.append((xk, yk))
                alphas.append(alpha)

            return path, alphas


        # def backtracking_line_search_sym(f_sym, grad_f_sym, x0, y0, alpha0=1.0, beta=0.5, c=1e-4, max_iters=100):
        #     f_lambd = sp.lambdify((x, y), f_sym, modules='numpy')
        #     grad_f_lambd = [sp.lambdify((x, y), g, modules='numpy') for g in grad_f_sym]

        #     xk, yk = x0, y0
        #     path = [(xk, yk)]
        #     alphas = []

        #     for _ in range(max_iters):
        #         gx, gy = grad_f_lambd[0](xk, yk), grad_f_lambd[1](xk, yk)
        #         grad_norm = gx**2 + gy**2

        #         if grad_norm < 1e-10:
        #             break

        #         alpha = alpha0
        #         while True:
        #             x_new = xk - alpha * gx
        #             y_new = yk - alpha * gy
        #             lhs = f_lambd(x_new, y_new)
        #             rhs = f_lambd(xk, yk) - c * alpha * grad_norm
        #             if lhs <= rhs or alpha < 1e-8:
        #                 break
        #             alpha *= beta

        #         xk, yk = xk - alpha * gx, yk - alpha * gy
        #         path.append((xk, yk))
        #         alphas.append(alpha)

        #     return path, alphas


        def optimize_path(x0, y0, optimizer, lr, steps, f_func, grad_f=None, hessian_f=None, options=None):
            path = [(x0, y0)]
            options = options or {}
            meta = {} 

            # --- Special Cases: Return early ---
            if optimizer == "GradientDescent" and options.get("use_backtracking", False):
                grad_f_expr = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
                path, alphas = backtracking_line_search_sym(f_expr, grad_f_expr, x0, y0)
                meta["callback_steps"] = len(path)
                return path, alphas, meta


            if optimizer == "Newton's Method" and options.get("newton_variant") in ["BFGS", "L-BFGS"]:
                from scipy.optimize import minimize

                path_coords = []
                losses = []

                def loss_vec(v):
                    return f_func(v[0], v[1])

                def grad_vec(v):
                    return np.atleast_1d(grad_f(v[0], v[1]))

                def callback(vk):
                    path_coords.append((vk[0], vk[1]))
                    losses.append(f_func(vk[0], vk[1]))

                x0_vec = np.array([x0, y0])
                method = "L-BFGS-B" if options["newton_variant"] == "L-BFGS" else "BFGS"

                res = minimize(loss_vec, x0_vec, method=method, jac=grad_vec, callback=callback, options={"maxiter": steps})

                # Fallback if callback was never triggered (e.g., early convergence)
                if not path_coords:
                    path_coords = [tuple(res.x)]
                    losses = [f_func(*res.x)]

                meta["res_nit"] = res.nit
                meta["callback_steps"] = len(path_coords)
                return path_coords, losses, meta


            if optimizer == "Simulated Annealing":
                T, cooling = options.get("T", 2.0), options.get("cooling", 0.95)
                current = f_func(x0, y0)
                for _ in range(steps):
                    xn, yn = x0 + np.random.randn(), y0 + np.random.randn()
                    candidate = f_func(xn, yn)
                    if candidate < current or np.random.rand() < np.exp(-(candidate - current)/T):
                        x0, y0 = xn, yn
                        current = candidate
                        path.append((x0, y0))
                    T *= cooling
                meta["callback_steps"] = len(path)
                return path, None, meta
            
            if optimizer == "Genetic Algorithm":
                pop_size = options.get("pop_size", 20)
                mutation_std = options.get("mutation_std", 0.3)
                pop = [np.random.uniform(-5, 5, 2) for _ in range(pop_size)]
                for _ in range(steps // 2):
                    pop = sorted(pop, key=lambda p: f_func(p[0], p[1]))[:pop_size // 2]
                    children = [np.mean([pop[i], pop[j]], axis=0) + np.random.normal(0, mutation_std, 2)
                                for i in range(len(pop)) for j in range(i+1, len(pop))][:pop_size // 2]
                    pop += children
                best = sorted(pop, key=lambda p: f_func(p[0], p[1]))[0]
                path = [tuple(best)]
                meta["callback_steps"] = len(path)
                return path, None, meta

            # --- Standard Optimizer Loop ---
            m, v = np.zeros(2), np.zeros(2)
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for t in range(1, steps + 1):
                x_t, y_t = path[-1]
                grad = grad_f(x_t, y_t)

                if optimizer == "Adam":
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    update = lr * m_hat / (np.sqrt(v_hat) + eps)

                elif optimizer == "RMSProp":
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    update = lr * grad / (np.sqrt(v) + eps)

                elif optimizer == "Newton's Method":
                    variant = options.get("newton_variant", "Classic Newton")
                    if variant == "Classic Newton":
                        try:
                            H = hessian_f(x_t, y_t)
                            H_inv = np.linalg.inv(H)
                            update = H_inv @ grad
                        except:
                            update = grad
                    elif variant == "Numerical Newton":
                        eps_num = 1e-4
                        def fx(x_, y_): return f_func(x_, y_)
                        def second_partial(f, x, y, i, j):
                            h = eps_num
                            if i == 0 and j == 0:
                                return (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
                            elif i == 1 and j == 1:
                                return (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
                            else:
                                return (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
                        H = np.array([
                            [second_partial(fx, x_t, y_t, 0, 0), second_partial(fx, x_t, y_t, 0, 1)],
                            [second_partial(fx, x_t, y_t, 1, 0), second_partial(fx, x_t, y_t, 1, 1)]
                        ])
                        try:
                            update = np.linalg.inv(H) @ grad
                        except:
                            update = grad

                else:  # Default: GradientDescent
                    update = lr * grad

                update = np.asarray(update)
                new_x, new_y = x_t - update[0], y_t - update[1]
                path.append((new_x, new_y))

            meta["callback_steps"] = len(path)
            return path, None, meta


        if optimizer == "Newton's Method":
            with st.expander("🧠 Newton Method Variants Explained", expanded=False):
                st.markdown("### 📘 Classic Newton vs. Numerical vs. Quasi-Newton")
                st.markdown("Newton's Method is a powerful optimization technique that uses **second-order derivatives** or their approximations to accelerate convergence.")

                st.markdown("#### 🧮 Classic Newton (Symbolic)")
                st.markdown("- Uses the **symbolic Hessian matrix** from calculus:")
                st.latex(r"\nabla^2 f(x, y)")
                st.markdown("- ✅ Very efficient and accurate for simple analytic functions (e.g., quadratic, convex).")
                st.markdown("- ⚠️ Can fail or be unstable if the Hessian is singular or badly conditioned.")

                st.markdown("#### 🔢 Numerical Newton")
                st.markdown("- Uses **finite differences** to approximate the Hessian.")
                st.markdown("- No need for symbolic derivatives.")
                st.markdown("- ✅ More robust for complex or unknown functions.")
                st.markdown("- 🐢 Slightly slower due to extra evaluations.")

                st.markdown("#### 🔁 BFGS / L-BFGS (Quasi-Newton)")
                st.markdown("- ✅ Avoids computing the full Hessian.")
                st.markdown("- Builds curvature estimate using gradients:")
                st.latex(r"""
                H_{k+1} = H_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{H_k s_k s_k^T H_k}{s_k^T H_k s_k}
                """)
                st.markdown("Where:")
                st.latex(r"s_k = x_{k+1} - x_k")
                st.latex(r"y_k = \nabla f(x_{k+1}) - \nabla f(x_k)")
                st.markdown("- 🧠 **BFGS**: High accuracy, stores full matrix.")
                st.markdown("- 🪶 **L-BFGS**: Stores only a few recent updates — ideal for high-dimensional problems.")
                st.markdown("💡 Quasi-Newton methods **approximate** curvature and still converge fast — especially useful for functions like Rosenbrock!")

                st.markdown("---")
                st.markdown("### ✏️ Why No Learning Rate?")
                st.markdown("Newton’s Method computes:")
                st.latex(r"x_{t+1} = x_t - H^{-1} \nabla f(x_t)")
                st.markdown("So it **naturally determines the best step direction and size** — no need for manual tuning like in gradient descent.")

        # === Shared simulation function ===
        def simulate_optimizer(opt_name, f_expr, lr=0.01, steps=50):
            f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
            # f_func = sp.lambdify((x, y), f_expr, modules="numpy")
            x0, y0 = -3, 3
            path = [(x0, y0)]
            m, v = 0, 0
            for t in range(1, steps + 1):
                x_t, y_t = path[-1]
                dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
                dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
                g = np.array([dx, dy])
                if opt_name == "Adam":
                    m = 0.9 * m + 0.1 * g
                    v = 0.999 * v + 0.001 * (g ** 2)
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                elif opt_name == "RMSProp":
                    v = 0.999 * v + 0.001 * (g ** 2)
                    update = lr * g / (np.sqrt(v) + 1e-8)
                elif opt_name == "Newton's Method":
                    hess = sp.hessian(f_expr, (x, y))
                    hess_func = sp.lambdify((x, y), hess, modules="numpy")
                    try:
                        H = np.array(hess_func(x_t, y_t))
                        H_inv = np.linalg.inv(H)
                        update = H_inv @ g
                    except:
                        update = g
                else:
                    update = lr * g
                x_new, y_new = x_t - update[0], y_t - update[1]
                path.append((x_new, y_new))
            final_x, final_y = path[-1]
            grad_norm = np.linalg.norm(g)
            return {
                "Optimizer": opt_name,
                "Final Value": round(f_func(final_x, final_y), 4),
                "Gradient Norm": round(grad_norm, 4),
                "Steps": len(path) - 1
            }
        g_funcs = [sp.lambdify((x_sym, y_sym), g, modules=["numpy"]) for g in constraints]  
        # g_funcs = [sp.lambdify((x, y), g, modules=["numpy"]) for g in constraints]
        f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
        # f_func = sp.lambdify((x, y), f_expr, modules=["numpy"])
        grad_f = lambda x0, y0: np.array([
            (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
            (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
        ])

        def hessian_f(x0, y0):
            hess_expr = sp.hessian(f_expr, (x, y))
            hess_func = sp.lambdify((x, y), hess_expr, modules=["numpy"])
            return np.array(hess_func(x0, y0))

        # === Pull final values from session_state
        start_x = st.session_state.get("start_x", -3.0)
        start_y = st.session_state.get("start_y", 3.0)
        lr = st.session_state.get("lr", 0.01)
        steps = st.session_state.get("steps", 50)


        path, alpha_log, meta = optimize_path(
            start_x, start_y,
            optimizer=optimizer,
            lr=lr,
            steps=steps,
            f_func=f_func,
            grad_f=grad_f,
            hessian_f=hessian_f,
            options=options
        )

        xs, ys = zip(*path)
        Z_path = [f_func(xp, yp) for xp, yp in path]

        x_vals = np.linspace(-5, 5, 200)
        y_vals = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_func(X, Y)

        col1, col2 = st.columns(2)
        with col1:
            fig = plt.figure(figsize=(4, 3))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
            ax.plot(xs, ys, Z_path, 'r*-')
            ax.set_title("3D Descent Path")
            st.pyplot(fig)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.contour(X, Y, Z, levels=30, cmap='viridis')
            if constraints:
                for g_f in g_funcs:
                    G = g_f(X, Y)
                    ax2.contour(X, Y, G, levels=[0], colors='red', linewidths=2)
            ax2.plot(xs, ys, 'r*-', label='Path')
            ax2.legend()
            ax2.set_title("2D Contour + Constraints")
            st.pyplot(fig2)

        # # ✅ Show alpha values if using backtracking
        # if optimizer == "GradientDescent" and options.get("use_backtracking", False) and alpha_log:
        #     st.markdown("### 📉 Backtracking Step Sizes (α values)")
        #     fig_alpha, ax_alpha = plt.subplots(figsize=(5, 3))
        #     ax_alpha.plot(alpha_log, marker='o', linestyle='-')
        #     ax_alpha.set_title("Backtracking Step Sizes (α)")
        #     ax_alpha.set_xlabel("Step")
        #     ax_alpha.set_ylabel("Alpha")
        #     ax_alpha.grid(True)
        #     st.pyplot(fig_alpha)

        if show_animation:
            frames = []
            fig_anim, ax_anim = plt.subplots(figsize=(5, 4))

            for i in range(1, len(path) + 1):
                ax_anim.clear()
                ax_anim.contour(X, Y, Z, levels=30, cmap="viridis")
                ax_anim.plot(*zip(*path[:i]), 'r*-')
                ax_anim.set_xlim([-5, 5])
                ax_anim.set_ylim([-5, 5])
                ax_anim.set_title(f"Step {i}/{len(path)-1}")

                buf = BytesIO()
                fig_anim.savefig(buf, format='png', dpi=100)  # optional: set dpi
                buf.seek(0)
                frames.append(Image.open(buf).convert("P"))  # convert to palette for GIF efficiency
                buf.close()

            gif_buf = BytesIO()
            frames[0].save(
                gif_buf, format="GIF", save_all=True,
                append_images=frames[1:], duration=300, loop=0
            )
            gif_buf.seek(0)
            st.image(gif_buf, caption="📽️ Animated Descent Path", use_container_width=True)



    with st.expander("🧰 Optimizer Diagnostic Tools", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Optimizer Comparison")

            start_x = st.session_state.get("start_x", 0.0)
            start_y = st.session_state.get("start_y", 0.0)

            selected_opts = st.multiselect(
                "Optimizers",
                ["GradientDescent", "Adam", "RMSProp", "Newton's Method"],
                default=["GradientDescent"],
                key="compare"
            )
            fig_comp, ax_comp = plt.subplots(figsize=(4, 3))

            results = []
            summary_results = []

            for opt in selected_opts:
                path_opt, losses, meta= optimize_path(  # ✅ unpack both outputs
                    start_x,
                    start_y,
                    optimizer=opt,
                    lr=lr,
                    steps=steps,
                    f_func=f_func,
                    grad_f=grad_f,
                    hessian_f=hessian_f,
                    options=options
                )

                zs_coords = path_opt
                zs_vals = losses if losses is not None and len(losses) > 0 else [f_func(xp, yp) for xp, yp in zs_coords]
                grad_norm = float(np.linalg.norm(grad_f(*zs_coords[-1])))

                results.append((opt, zs_vals))

                summary_results.append({
                    "Optimizer": opt,
                    "Final Value": np.round(zs_vals[-1], 4),
                    "Gradient Norm": np.round(grad_norm, 4),
                    "Steps": len(zs_vals),
                    "Actual Steps": meta.get("res_nit", "N/A"),   # ✅ 使用 meta.get()
                    "Logged Steps": meta.get("callback_steps", "N/A")
                })

            # Sort results by final loss
            results.sort(key=lambda x: x[1][-1])

            for opt, zs in results:
                if zs is not None and len(zs) > 0:
                    ax_comp.plot(zs, label=f"{opt} ({len(zs)} steps)", marker="o", markersize=2)
                else:
                    st.warning(f"⚠️ No convergence data for {opt}")

            ax_comp.set_title("Convergence")
            ax_comp.set_xlabel("Step")
            ax_comp.set_ylabel("f(x, y)")
            ax_comp.set_ylim(bottom=0)
            ax_comp.legend()
            st.pyplot(fig_comp)

            # Show summary table
            st.markdown("#### 📋 Optimizer Summary Table")
            df_summary = pd.DataFrame(summary_results)
            st.dataframe(df_summary)



            st.markdown("#### 🔥 Gradient Norm Heatmap")
            # norm_grad = np.sqrt((np.gradient(Z, axis=0))**2 + (np.gradient(Z, axis=1))**2)
            # Temporary definition in case not computed above
            if 'Z_loss' not in locals():
                x_range = np.linspace(-5, 5, 50)
                y_range = np.linspace(-5, 5, 50)
                X_loss, Y_loss = np.meshgrid(x_range, y_range)
                Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2  # Default MSE-like loss

            norm_grad = np.sqrt((np.gradient(Z_loss, axis=0))**2 + (np.gradient(Z_loss, axis=1))**2)

            fig3, ax3 = plt.subplots()
            heat = ax3.imshow(norm_grad, extent=[-5, 5, -5, 5], origin='lower', cmap='plasma')
            fig3.colorbar(heat, ax=ax3, label="‖∇f‖")
            ax3.set_title("‖∇f(x, y)‖")
            st.pyplot(fig3)

        with col2:
            st.markdown("#### 🌄 Loss Surface")

            loss_type = st.radio("Loss Type", ["MSE", "Log Loss", "Cross Entropy", "Custom"])

            # Create input grid
            x_range = np.linspace(-5, 5, 50)  # reduced resolution for arrows
            y_range = np.linspace(-5, 5, 50)
            X_loss, Y_loss = np.meshgrid(x_range, y_range)

            # Compute Z surface and manually define minimum
            if loss_type == "MSE":
                Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2
                min_x, min_y = 2, -1
            elif loss_type == "Log Loss":
                Z_loss = np.log(1 + np.exp(-(X_loss + Y_loss)))
                min_x, min_y = 5, -5
            elif loss_type == "Cross Entropy":
                p = 1 / (1 + np.exp(-(X_loss + Y_loss)))
                Z_loss = -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
                min_x, min_y = 5, -5
            else:
                Z_loss = np.sin(X_loss) * np.cos(Y_loss)
                min_x, min_y = 0, 0

            # Compute gradients (numerical partial derivatives)
            dZ_dx, dZ_dy = np.gradient(Z_loss, x_range, y_range)

            # === Side-by-side plots ===
            col1, col2 = st.columns(2)

            with col1:
                fig3d = plt.figure(figsize=(5, 4))
                ax3d = fig3d.add_subplot(111, projection='3d')
                ax3d.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis', edgecolor='none', alpha=0.9)
                ax3d.scatter(min_x, min_y, np.min(Z_loss), color='red', s=50, label='Min')
                ax3d.set_title(f"{loss_type} Surface")
                ax3d.legend()
                st.pyplot(fig3d)

            with col2:
                fig2d, ax2d = plt.subplots(figsize=(5, 4))
                contour = ax2d.contourf(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis')
                ax2d.plot(min_x, min_y, 'ro', label='Min')
                # Overlay gradient arrows (negative for descent)
                ax2d.quiver(X_loss, Y_loss, -dZ_dx, -dZ_dy, color='white', alpha=0.7, scale=50)
                fig2d.colorbar(contour, ax=ax2d, label="Loss")
                ax2d.set_title(f"{loss_type} Contour View + Gradient Field")
                ax2d.legend()
                st.pyplot(fig2d)

            st.markdown("#### ✅ Constraint Checker")

            start_x = st.session_state.get("start_x", 0.0)
            start_y = st.session_state.get("start_y", 0.0)


            # ⛳ Ensure path is updated
            path, losses, meta = optimize_path(
                start_x,
                start_y,
                optimizer=selected_opts[0],  # or st.session_state.get("optimizer", "GradientDescent")
                lr=lr,
                steps=steps,
                f_func=f_func,
                grad_f=grad_f,
                hessian_f=hessian_f,
                options=options
            )
            if constraints:
                fig_con, ax_con = plt.subplots(figsize=(4, 3))
                for i, g_func in enumerate(g_funcs):
                    violations = [g_func(xp, yp) for xp, yp in path]
                    ax_con.plot(violations, label=f"g{i+1}(x, y)")
                ax_con.axhline(0, color="red", linestyle="--")
                ax_con.set_xlabel("Step")
                ax_con.set_ylabel("g(x, y)")
                ax_con.legend()
                st.pyplot(fig_con)
            else:
                st.info("No constraints defined.")
    


    # === Symbolic Analysis: KKT, Gradient & Hessian ===
    with st.expander("📐 Symbolic Analysis: KKT, Gradient & Hessian", expanded=False):
        st.markdown("#### 🎯 Objective & Lagrangian")
        st.latex(r"f(x, y) = " + sp.latex(f_expr))
        st.latex(r"\mathcal{L}(x, y, \lambda) = " + sp.latex(L_expr))

        st.markdown("#### ✅ KKT Conditions")
        for i, cond in enumerate(kkt_conditions):
            st.latex(fr"\text{{KKT}}_{{{i+1}}} = {sp.latex(cond)}")

        st.markdown("#### 🧮 Gradient & Hessian")
        grad = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
        # grad = [sp.diff(f_expr, v) for v in (x, y)]
        # hessian = sp.hessian(f_expr, (x, y))
        hessian = sp.hessian(f_expr, (x_sym, y_sym))

        st.latex("Gradient: " + sp.latex(sp.Matrix(grad)))
        st.latex("Hessian: " + sp.latex(hessian))

        if optimizer == "Newton's Method":
            st.markdown("#### 🧠 Newton Method Diagnostics")        
            hess = sp.hessian(f_expr, (x_sym, y_sym))
            hess_func = sp.lambdify((x_sym, y_sym), hess, modules="numpy")
            H_val = np.array(hess_func(start_x, start_y))
            det_val = np.linalg.det(H_val)

            st.markdown("#### 📐 Hessian Matrix")
            st.latex(r"\text{H}(x, y) = " + sp.latex(hess))

            st.markdown("#### 📏 Determinant")
            st.latex(r"\det(\text{H}) = " + sp.latex(sp.det(hess)))

            if np.isclose(det_val, 0, atol=1e-6):
                st.error("❌ Determinant is zero — Newton's Method cannot proceed (singular Hessian).")
            elif det_val < 0:
                st.warning("⚠️ Negative determinant — may indicate a saddle point or non-convex region.")
            else:
                st.success("✅ Hessian is suitable for Newton's Method descent.")

    # === Newton Method Info ===

# # === Mode: LLM Assistant ===
# elif mode == "🤖 LLM Assistant":
#     st.subheader("🤖 LLM Assistant: Explore Your Data Intelligently")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     uploaded_file = st.file_uploader("📁 Upload a dataset (CSV)", type=["csv"])
#     if uploaded_file:
#         st.session_state.uploaded_file = uploaded_file

#     df = None
#     if "uploaded_file" in st.session_state:
#         try:
#             if st.session_state.uploaded_file is not None:
#                 content = st.session_state.uploaded_file.read()
#                 if content.strip() == b"":
#                     raise ValueError("Uploaded file is empty.")
#                 st.session_state.uploaded_file.seek(0)
#                 df = pd.read_csv(st.session_state.uploaded_file)
#             else:
#                 raise ValueError("No file uploaded.")
#         except Exception as e:
#             st.error(f"❌ Failed to read uploaded CSV: {e}")
#             st.session_state.uploaded_file = None
#             df = None

#     if df is not None:
#         st.write("### 📄 Data Preview", df.head())
#         st.write("### 📊 Summary Statistics")
#         st.dataframe(df.describe(include='all'))

#         st.markdown("### 💡 Suggested Prompts")
#         st.markdown("""
#         - What are the most correlated features?
#         - Show a summary of missing values
#         - Which features influence the target most?
#         - What kind of plot would help visualize X vs Y?
#         - Can you generate a histogram of column X?
#         - Show pairwise plots for selected features
#         - Predict the target using linear regression
#         - Detect outliers or anomalies in the data
#         """)

#     # === LLM Chat ===

#     llm_choice = st.radio("Choose LLM Backend", ["OpenAI", "Local LLM"])
#     if llm_choice == "OpenAI":
#         def query_llm(prompt: str) -> str:
#             resp = client.ChatCompletion.create(
#                 model="gpt-4o",
#                 messages=[ ... ],
#             )
#             return resp.choices[0].message.content.strip()
#     else:
#         def query_llm(prompt: str) -> str:
#             return query_local_llm(prompt)

#     # ─── user input / querying ────────────────────────────────
#     user_input = st.text_input("💬 Ask something (about your data):")
#     if user_input:
#         with st.spinner("🤖 Thinking..."):
#             # build your prompt exactly once
#             if df is not None:
#                 summary = df.describe(include="all").to_string()
#                 corr    = df.corr(numeric_only=True).round(3).to_string()
#                 full_prompt = (
#                     f"You are a data analysis assistant.\n\n"
#                     f"Dataset summary:\n{summary}\n\n"
#                     f"Correlation matrix:\n{corr}\n\n"
#                     f"Q: {user_input}\nA:"
#                 )
#             else:
#                 full_prompt = user_input

#             try:
#                 answer = query_llm(full_prompt)
#                 st.session_state.chat_history.append((user_input, answer))
#             except Exception as e:
#                 st.error(f"❌ LLM Error: {e}")
#                 answer = None

#         if answer:
#             st.markdown(
#                 f"<div style='background-color:#e8f5e9;padding:10px;border-radius:8px;'>{answer}</div>",
#                 unsafe_allow_html=True,
#             )

#     if df is not None:             
#         st.markdown("### 📈 Custom Chart Generator")
#         chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
#         x_col = st.selectbox("X-axis Column", df.columns)
#         y_col = st.selectbox("Y-axis Column", df.columns)
#         if st.button("Generate Chart"):
#             fig, ax = plt.subplots()
#             if chart_type == "Line":
#                 ax.plot(df[x_col], df[y_col])
#             elif chart_type == "Bar":
#                 ax.bar(df[x_col], df[y_col])
#             elif chart_type == "Scatter":
#                 ax.scatter(df[x_col], df[y_col])
#             elif chart_type == "Histogram":
#                 ax.hist(df[x_col], bins=20)
#             ax.set_xlabel(x_col)
#             ax.set_ylabel(y_col)
#             st.pyplot(fig, use_container_width=True)

#     st.markdown("### 💾 Export Data")
#     file_name = st.text_input("Output file name (without extension)", "my_data")
#     if st.button("Download as CSV"):
#         tmp_csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             label="📥 Download Processed CSV",
#             data=tmp_csv,
#             file_name=f"{file_name}.csv",
#             mime="text/csv"
#         )



#     if st.session_state.chat_history:
#         st.markdown("### 📜 Chat History")
#         for q, a in st.session_state.chat_history[::-1]:
#             st.markdown(f"**You:** {q}")
#             st.markdown(f"**Assistant:** {a}")

#     if "uploaded_file" not in st.session_state or df is None:
#         st.info("📂 Upload a dataset to explore insights with the assistant.")


# === Mode: LLM Assistant ===


elif mode == "🤖 LLM Assistant":
    st.subheader("🤖 LLM Assistant: Explore Your Data Intelligently")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("📁 Upload a dataset (CSV)", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    df = None
    if "uploaded_file" in st.session_state:
        try:
            if st.session_state.uploaded_file is not None:
                content = st.session_state.uploaded_file.read()
                if content.strip() == b"":
                    raise ValueError("Uploaded file is empty.")
                st.session_state.uploaded_file.seek(0)
                df = pd.read_csv(st.session_state.uploaded_file)
            else:
                raise ValueError("No file uploaded.")
        except Exception as e:
            st.error(f"❌ Failed to read uploaded CSV: {e}")
            st.session_state.uploaded_file = None
            df = None

    if df is not None:
        st.write("### 📄 Data Preview", df.head())
        st.write("### 📊 Summary Statistics")
        st.dataframe(df.describe(include='all'))

        st.markdown("### 💡 Suggested Prompts")
        st.markdown("""
        - What are the most correlated features?
        - Show a summary of missing values
        - Which features influence the target most?
        - What kind of plot would help visualize X vs Y?
        - Can you generate a histogram of column X?
        - Show pairwise plots for selected features
        - Predict the target using linear regression
        - Detect outliers or anomalies in the data
        """)

    # === 🔐 API Key Selection ===
    st.markdown("### 🔑 LLM Settings")
    llm_choice = st.radio("Choose LLM Backend", ["OpenAI", "Local LLM"])
    user_key = st.text_input("Enter your OpenAI API key (optional):", type="password")

    final_key = user_key if user_key else st.secrets.get("OPENAI_API_KEY")

    query_llm = None

    if llm_choice == "OpenAI":
        st.warning("⚠️ Using OpenAI's GPT may incur charges.")
        st.markdown("[💳 View billing](https://platform.openai.com/account/billing)", unsafe_allow_html=True)
        allow_openai_use = st.checkbox("✅ I understand I may be billed for OpenAI usage", value=True)

        if allow_openai_use:
            from langchain.chat_models import ChatOpenAI
            from langchain.memory import ConversationBufferMemory

            llm = ChatOpenAI(temperature=0.3, openai_api_key=final_key)
            memory = ConversationBufferMemory()
            query_llm = lambda prompt: llm.predict(prompt)
        else:
            query_llm = lambda prompt: "⚠️ OpenAI API key missing or billing not confirmed."
    else:
        query_llm = lambda prompt: query_local_llm(prompt)

    # === 🧠 User Input ===
    user_input = st.text_input("💬 Ask something (about your data):")
    if user_input:
        with st.spinner("🤖 Thinking..."):
            if df is not None:
                summary = df.describe(include="all").to_string()
                corr = df.corr(numeric_only=True).round(3).to_string()
                full_prompt = (
                    f"You are a data analysis assistant.\n\n"
                    f"Dataset summary:\n{summary}\n\n"
                    f"Correlation matrix:\n{corr}\n\n"
                    f"Q: {user_input}\nA:"
                )
            else:
                full_prompt = user_input

            try:
                st.code(full_prompt, language="text")  # ✅ show prompt for debug
                st.info("📡 Sending prompt to LLM...")
                answer = query_llm(full_prompt)

                if isinstance(answer, str) and answer.strip():
                    st.session_state.chat_history.append((user_input, answer))
                else:
                    st.warning("⚠️ LLM returned an empty or invalid response.")


            except Exception as e:
                st.error(f"❌ LLM Error: {e}")
                answer = None

        if answer and isinstance(answer, str) and answer.strip():
            st.markdown(
                f"<div style='background-color:#e8f5e9;padding:10px;border-radius:8px;'>{answer}</div>",
                unsafe_allow_html=True,
            )


    if df is not None:
        st.markdown("### 📈 Custom Chart Generator")
        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
        x_col = st.selectbox("X-axis Column", df.columns)
        y_col = st.selectbox("Y-axis Column", df.columns)
        if st.button("Generate Chart"):
            fig, ax = plt.subplots()
            if chart_type == "Line":
                ax.plot(df[x_col], df[y_col])
            elif chart_type == "Bar":
                ax.bar(df[x_col], df[y_col])
            elif chart_type == "Scatter":
                ax.scatter(df[x_col], df[y_col])
            elif chart_type == "Histogram":
                ax.hist(df[x_col], bins=20)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig, use_container_width=True)

    st.markdown("### 💾 Export Data")
    file_name = st.text_input("Output file name (without extension)", "my_data")
    if st.button("Download as CSV"):
        tmp_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Processed CSV",
            data=tmp_csv,
            file_name=f"{file_name}.csv",
            mime="text/csv"
        )

    if st.session_state.chat_history:
        st.markdown("### 📜 Chat History")
        for q, a in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")

    if "uploaded_file" not in st.session_state or df is None:
        st.info("📂 Upload a dataset to explore insights with the assistant.")





elif mode == "🧠 Unsupervised Learning":
    st.subheader("🧠 Unsupervised Learning Explorer")

    # Submodule selection in sidebar
    unsup_option = st.sidebar.radio("Select Unsupervised Module", [
        "Clustering",
        "Dimensionality Reduction",
        # "Anomaly Detection",
        # Add more modules as needed
    ])

    # Render selected module
    if unsup_option == "Clustering":
        from clustering import clustering_ui
        clustering_ui()
    
    elif unsup_option == "Dimensionality Reduction":
        from dim_reduction import dim_reduction_ui
        dim_reduction_ui()








# Footer
st.markdown("---")
st.info("Switch between ML tasks and optimization demos using the sidebar.")







# === Temp to modulise the optimisation functions ===

elif mode == "🌋 Optimization Playground":
    # Inserted full optimizer playground block here
    import sympy as sp
    import time
    from sympy import symbols, lambdify
    from io import BytesIO
    from PIL import Image
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    st.title("🧪 Optimizer Visual Playground")

    x, y, w = sp.symbols("x y w")
    x_sym, y_sym, w_sym = sp.symbols("x y w")

    predefined_funcs = {
        "Quadratic Bowl": (x**2 + y**2, [], "Convex bowl, global min at origin."),
        "Saddle": (x**2 - y**2, [], "Saddle point at origin, non-convex."),
        "Rosenbrock": ((1 - x)**2 + 100 * (y - x**2)**2, [], "Banana-shaped curved valley, classic test function."),
        "Constrained Circle": (x * y, [x + y - 1], "Constrained optimization with line x + y = 1."),
        "Double Constraint": (x**2 + y**2, [x + y - 1, x**2 + y**2 - 4], "Circular + linear intersection constraints."),
        "Multi-Objective": (w * ((x - 1)**2 + (y - 2)**2) + (1 - w) * ((x + 2)**2 + (y + 1)**2), [], "Weighted sum of two loss terms."),
        "Ackley": (-20*sp.exp(-0.2*sp.sqrt(0.5*(x**2 + y**2))) - sp.exp(0.5*(sp.cos(2*sp.pi*x) + sp.cos(2*sp.pi*y))) + sp.E + 20, [], "Multimodal non-convex function."),
        "Rastrigin": (20 + x**2 - 10*sp.cos(2*sp.pi*x) + y**2 - 10*sp.cos(2*sp.pi*y), [], "Many local minima, non-convex."),
        "Styblinski-Tang": (0.5*((x**4 - 16*x**2 + 5*x) + (y**4 - 16*y**2 + 5*y)), [], "Non-convex with multiple minima."),
        "Sphere": (x**2 + y**2, [], "Simple convex function."),
        "Himmelblau": ((x**2 + y - 11)**2 + (x + y**2 - 7)**2, [], "Multiple global minima, non-convex."),
        "Booth": ((x + 2*y - 7)**2 + (2*x + y - 5)**2, [], "Simple convex function."),
        "Beale": ((1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2, [], "Non-convex with multiple minima.")

    }

            
    with st.expander("💬 Ask the LLM about Optimizers or Math", expanded=False):
        user_question = st.text_input("Ask anything (e.g., What is BFGS? Why is Newton unstable?)")
        if user_question:
            with st.spinner("🤖 Thinking..."):
                full_prompt = (
                    "You are an expert on numerical optimization methods such as Gradient Descent, "
                    "Adam, Newton’s Method, Simulated Annealing, and Genetic Algorithms.\n\n"
                    f"Question: {user_question}\nAnswer:"
                )
                response = query_llm(full_prompt)
                st.markdown(
                    f"<div style='background-color:#f4f9ff;padding:10px;border-radius:6px;'>{response}</div>",
                    unsafe_allow_html=True,
                )

    with st.expander("🧠 Taylor Series & Optimizer Foundations", expanded=True):
        st.markdown("### 📚 How Taylor Series Explains Optimizers")

        st.markdown("Many optimization methods are based on the **Taylor series expansion** of a function. This helps us approximate the function locally using its derivatives:")

        st.latex(r"""
        f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
        """)
        st.markdown("- This is the **1st-order Taylor expansion**, which is the basis of **Gradient Descent**. It only uses the slope (gradient) to decide the direction to move.")

        st.latex(r"""
        f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H(x) \Delta x
        """)
        st.markdown("- This is the **2nd-order Taylor expansion**, used in **Newton's Method**. It adds the **Hessian** (curvature) to adjust the step size and improve convergence.")

        st.markdown("### ✍️ Summary")
        st.markdown("""
        You can imagine it like this:
        - ✅ **1st-order**: Approximates with a **tangent line** (local slope).
        - ✅ **2nd-order**: Approximates with a **parabola** (slope + curvature).

        This helps students connect abstract equations with **optimizer logic**.
                        
        - **Gradient Descent** is a 1st-order method → uses slope only.
        - **Newton's Method** is a 2nd-order method → uses slope and curvature.
        - Understanding these expansions builds strong intuition about how optimizers move in the loss landscape.
        """)

        st.markdown("---")
        st.markdown("### 🔍 Univariate Taylor Expansion (1D Preview)")

        try:
            func_choice = st.selectbox("Choose a function:", ["cos(x)", "exp(x)", "ln(1+x)", "tanh(x)", "Custom"])
            show_3rd_4th = st.checkbox("➕ Show 3rd & 4th-order", value=False)
            show_linear = st.checkbox("Show 1st-order (Linear)", value=True)
            show_parabola = st.checkbox("Show 2nd-order (Parabola)", value=True)

            x_sym = sp.symbols('x')

            def get_function(choice):
                if choice == "cos(x)": return sp.cos(x_sym), (-3, 3)
                if choice == "exp(x)": return sp.exp(x_sym), (-3, 3)
                if choice == "ln(1+x)": return sp.ln(1 + x_sym), (-0.9, 3)
                if choice == "tanh(x)": return sp.tanh(x_sym), (-3, 3)
                if choice == "Custom":
                    user_input = st.text_input("Enter function f(x):", "x**2 * sin(x)")
                    try:
                        return sp.sympify(user_input), (-3, 3)
                    except Exception as e:
                        st.error(f"Invalid input: {e}")
                        st.stop()

            f_sym, (xmin, xmax) = get_function(func_choice)
            x_sym, a_sym = sp.symbols('x a')
            h = x_sym - a_sym

            # Derivatives and Taylor terms
            f1, f2, f3, f4 = [sp.diff(f_sym, x_sym, i) for i in range(1, 5)]
            T1 = f_sym.subs(x_sym, a_sym) + f1.subs(x_sym, a_sym) * h
            T2 = T1 + (1/2) * f2.subs(x_sym, a_sym) * h**2
            T4 = T2 + (1/6) * f3.subs(x_sym, a_sym) * h**3 + (1/24) * f4.subs(x_sym, a_sym) * h**4

            st.markdown("### ✏️ Taylor Expansion at $x = a$")
            st.latex(f"f(x) \\approx {sp.latex(T1)}")
            st.latex(f"f(x) \\approx {sp.latex(T2)}")
            if show_3rd_4th:
                st.latex(f"f(x) \\approx {sp.latex(T4)}")

            # Numeric plotting
            f_np = sp.lambdify(x_sym, f_sym, "numpy")
            derivs = [sp.lambdify(x_sym, d, "numpy") for d in [f1, f2, f3, f4]]
            a = st.slider("Expansion point a:", float(xmin), float(xmax), 0.0, 0.1)
            x = np.linspace(xmin, xmax, 500)

            f_vals = [d(a) for d in derivs]
            t1 = f_np(a) + f_vals[0] * (x - a)
            t2 = t1 + 0.5 * f_vals[1] * (x - a)**2
            t4 = t2 + (1/6) * f_vals[2] * (x - a)**3 + (1/24) * f_vals[3] * (x - a)**4

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, f_np(x), label=f"f(x) = {func_choice}", color='blue')
            if show_linear: ax.plot(x, t1, '--', label='1st-order', color='red')
            if show_parabola: ax.plot(x, t2, '--', label='2nd-order', color='orange')
            if show_3rd_4th: ax.plot(x, t4, '--', label='3rd/4th-order', color='green')
            ax.axvline(a, color='gray', linestyle=':')
            ax.axhline(0, color='black', linewidth=0.8)
            ax.scatter(a, f_np(a), color='black')
            ax.set_title(f"Taylor Approximations at x = {a}")
            ax.legend(); ax.grid(True)
            st.pyplot(fig)

            # === Optional Animation ===
            # === Optional Animation ===
            if st.checkbox("🎬 Animate 1st & 2nd-order Approximation"):
                st.markdown("### 🎬 Animation: 1st & 2nd-Order Taylor Approximation")
                fig_anim, ax_anim = plt.subplots(figsize=(8, 4))

                line_true, = ax_anim.plot(x, f_np(x), label="f(x)", color='blue')
                line_taylor1, = ax_anim.plot([], [], '--', label="1st-order", color='red')
                line_taylor2, = ax_anim.plot([], [], '--', label="2nd-order", color='orange')
                point, = ax_anim.plot([], [], 'ko')

                ax_anim.set_xlim(xmin, xmax)
                y_vals = f_np(x)
                buffer = 0.4 * (np.max(y_vals) - np.min(y_vals))
                ax_anim.set_ylim(np.min(y_vals) - buffer, np.max(y_vals) + buffer)
                ax_anim.axhline(0, color='gray', lw=0.5)
                ax_anim.grid(True)
                ax_anim.legend()

                a_vals = np.linspace(xmin + 0.1, xmax - 0.1, 60)

                def update(frame):
                    a_val = a_vals[frame]
                    f_a = f_np(a_val)
                    f1_a = derivs[0](a_val)
                    f2_a = derivs[1](a_val)

                    t1_anim = f_a + f1_a * (x - a_val)
                    t2_anim = t1_anim + 0.5 * f2_a * (x - a_val)**2

                    line_taylor1.set_data(x, t1_anim)
                    line_taylor2.set_data(x, t2_anim)
                    point.set_data([a_val], [f_a])
                    ax_anim.set_title(f"Taylor Approx at a = {a_val:.2f}")
                    return line_taylor1, line_taylor2, point

                ani = FuncAnimation(fig_anim, update, frames=len(a_vals), interval=100, blit=True)

                buf = BytesIO()
                writer = PillowWriter(fps=20)
                with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                    ani.save(tmpfile.name, writer=writer)
                    tmpfile.seek(0)
                    gif_base64 = base64.b64encode(tmpfile.read()).decode("utf-8")

                components.html(f'<img src="data:image/gif;base64,{gif_base64}" width="100%">', height=350)

        except Exception as e:
            st.error(f"Rendering error: {e}")

            

# --- SECTION: Multivariable Taylor Expansion (2D Preview) ---
        st.markdown("---")

        st.markdown("### 🌐 Multivariable Taylor Expansion (2D Preview)")

        multi_func = st.selectbox("Choose function:", ["Quadratic Bowl", "Rosenbrock"])
        x, y, a, b = sp.symbols('x y a b')

        if multi_func == "Quadratic Bowl":
            fxy = x**2 + y**2
        else:
            fxy = (1 - x)**2 + 100 * (y - x**2)**2

        # Derivatives
        grad = [sp.diff(fxy, v) for v in (x, y)]
        hess = [[sp.diff(g, v) for v in (x, y)] for g in grad]

        # Center sliders
        a_val = st.slider("Center a (x)", -5.0, 5.0, 0.0)
        b_val = st.slider("Center b (y)", -5.0, 5.0, 0.0)

        # Zoom toggle
        zoom_in = st.checkbox("🔍 Zoom into local neighborhood", value=False)

        zoom_center_x, zoom_center_y = 0.0, 0.0  # ✅ Fixed center for meshgrid limits

        if zoom_in:
            xlim = (zoom_center_x - 1, zoom_center_x + 1)
            ylim = (zoom_center_y - 1, zoom_center_y + 1)
        else:
            xlim = (-5, 5)
            ylim = (-5, 5)


        # Compute Taylor series
        f_a = fxy.subs({x: a, y: b})
        grad_val = [g.subs({x: a, y: b}) for g in grad]
        T1 = f_a + grad_val[0]*(x - a) + grad_val[1]*(y - b)

        hess_val = [[h.subs({x: a, y: b}) for h in row] for row in hess]
        T2 = T1 + 0.5 * (
            hess_val[0][0]*(x - a)**2 +
            2*hess_val[0][1]*(x - a)*(y - b) +
            hess_val[1][1]*(y - b)**2
        )

        st.markdown("### ✏️ Expansion at $(x, y) = (a, b)$")
        st.latex(f"f(x, y) \\approx {sp.latex(T1)}")
        st.latex(f"f(x, y) \\approx {sp.latex(T2)}")

        # Evaluate
        f_np = sp.lambdify((x, y), fxy, "numpy")
        T2_np = sp.lambdify((x, y, a, b), T2, "numpy")

        X, Y = np.meshgrid(np.linspace(xlim[0], xlim[1], 100), np.linspace(ylim[0], ylim[1], 100))
        Z_true = f_np(X, Y)
        Z_taylor = T2_np(X, Y, a_val, b_val)

        # Plot both surfaces
        fig_true = go.Figure(data=[go.Surface(z=Z_true, x=X, y=Y, colorscale='Viridis')])
        fig_true.update_layout(title="True Function", scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='f(x,y)'
        ), margin=dict(l=0, r=0, b=0, t=40))

        fig_taylor = go.Figure(data=[go.Surface(z=Z_taylor, x=X, y=Y, colorscale='RdBu')])
        fig_taylor.update_layout(title="2nd-Order Taylor Approx", scene=dict(
            xaxis_title='x', yaxis_title='y', zaxis_title='Approx'
        ), margin=dict(l=0, r=0, b=0, t=40))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_true, use_container_width=True)
        with col2:
            st.plotly_chart(fig_taylor, use_container_width=True)

        # --- SECTION: Animate ---
        st.markdown("---")
        st.markdown("### 🎬 Animate Taylor Approximation Surface")

        animate_mode = st.radio("Animate path:", ["a only", "b only", "both a & b"], index=0)
        param_vals = np.linspace(-1.0, 1.0, 30)
        frames = []

        for val in param_vals:
            if animate_mode == "a only":
                Z_frame = T2_np(X, Y, val, b_val)
                label = f"a = {val:.2f}"
            elif animate_mode == "b only":
                Z_frame = T2_np(X, Y, a_val, val)
                label = f"b = {val:.2f}"
            else:  # both a and b
                Z_frame = T2_np(X, Y, val, val)
                label = f"(a, b) = ({val:.2f}, {val:.2f})"

            frames.append(go.Frame(data=[
                go.Surface(z=Z_frame, x=X, y=Y, colorscale='RdBu')
            ], name=label))

        # Initial Z
        if animate_mode == "a only":
            Z0 = T2_np(X, Y, param_vals[0], b_val)
        elif animate_mode == "b only":
            Z0 = T2_np(X, Y, a_val, param_vals[0])
        else:
            Z0 = T2_np(X, Y, param_vals[0], param_vals[0])

        fig_anim = go.Figure(
            data=[go.Surface(z=Z0, x=X, y=Y, colorscale='RdBu')],
            layout=go.Layout(
                title="Animated 2nd-Order Taylor Approximation",
                scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='Approx'),
                updatemenus=[dict(
                    type="buttons",
                    showactive=True,
                    buttons=[dict(label="▶ Play", method="animate", args=[None])]
                )],
                sliders=[{
                    "steps": [{"args": [[f.name]], "label": f.name, "method": "animate"} for f in frames],
                    "currentvalue": {"prefix": "Center: "}
                }]
            ),
            frames=frames
        )

        st.plotly_chart(fig_anim, use_container_width=True)


    # 🪄 Optimizer Category Info Block (Outside main expander)
    with st.expander("🧠 Optimizer Category Info & Usage Tips", expanded=False):
        st.markdown("""
        ### 🤖 Optimizer Categories

        **Gradient-based Optimizers** use derivatives (gradients) to guide updates:

        - **GradientDescent**: Basic steepest descent using learning rate.
        - **Adam**: Adaptive learning rates with momentum (recommended for noisy or sparse gradients).
        - **RMSProp**: Like Adam but simpler; adjusts step size based on recent gradient magnitudes.
        - **Newton's Method**: Uses second-order derivatives (Hessian) for faster convergence, but may be unstable near saddle points.

        **Heuristic Optimizers** use stochastic or evolutionary strategies:

        - **Simulated Annealing**: Explores search space with temperature-based random steps. Good for escaping local minima.
        - **Genetic Algorithm**: Population-based method inspired by natural evolution. Effective for complex, non-differentiable functions.

        ---
        ### 🎮 How to Use the Playground

        - **1. Choose a function** from the dropdown.
        - **2. Select an optimizer** and tune parameters like learning rate or mutation strength.
        - **3. Try enabling _Auto-Tune_** to automatically pick good learning rate + steps.
        - **4. Adjust initial x/y** starting positions to see how descent paths change.
        - **5. Toggle _"Animate Descent"_** to visualize how the optimizer moves step-by-step.
        - **6. Compare optimizers** in the **"Diagnostic Tools"** section using convergence plots and summary tables.

        > 🧪 Try Newton's Method on **Quadratic Bowl** for fast convergence, or test **Genetic Algorithm** on **Rastrigin** to explore multimodal search.

        ---
        ⚠️ *Note:* Gradient-based methods require a smooth function. Use heuristic optimizers for discontinuous or non-differentiable objectives.
        """)



    with st.expander("🚀 Optimizer Visual Playground", expanded=True):
        col_left, col_right = st.columns([1, 1])

        with col_left:
            mode = st.radio("Function Source", ["Predefined", "Custom"])
            func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
            expr_str = st.text_input("Enter function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""
            w_val = st.slider("Weight w (Multi-Objective)", 0.0, 1.0, 0.5) if func_name == "Multi-Objective" else None

            optimizers = ["GradientDescent", "Adam", "RMSProp", "Newton's Method", "Simulated Annealing", "Genetic Algorithm"]
            optimizer = st.selectbox("Optimizer", optimizers)
            

            options = {}

            if optimizer == "Newton's Method":
                newton_variant = st.selectbox("Newton Variant", ["Classic Newton", "Numerical Newton", "BFGS", "L-BFGS"])
                options["newton_variant"] = newton_variant
            elif optimizer == "Simulated Annealing":
                options["T"] = st.slider("Initial Temperature (T)", 0.1, 10.0, 2.0)
                options["cooling"] = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
            elif optimizer == "Genetic Algorithm":
                options["pop_size"] = st.slider("Population Size", 10, 100, 20)
                options["mutation_std"] = st.slider("Mutation Std Dev", 0.1, 1.0, 0.3)

            auto_tune = False

            if optimizer in ["GradientDescent", "Adam", "RMSProp"]:
                if optimizer == "GradientDescent":
                    use_backtracking = st.checkbox("🔍 Use Backtracking Line Search", value=False)
                    options["use_backtracking"] = use_backtracking

                    if use_backtracking:
                        st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
                        auto_tune = False
                        if "auto_tune_checkbox" in st.session_state:
                            st.session_state["auto_tune_checkbox"] = False
                        st.caption("ℹ️ Disabled because backtracking search dynamically adjusts step size.")
                    else:
                        auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")
                else:
                    auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")

            elif optimizer == "Newton's Method":
                st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=False, disabled=True, key="auto_tune_disabled")
                auto_tune = False
                if "auto_tune_checkbox" in st.session_state:
                    st.session_state["auto_tune_checkbox"] = False
                st.caption("ℹ️ Auto-tune is not applicable to Newton’s Method.")
                
            start_xy_defaults = {
                "Quadratic Bowl": (-3.0, 3.0), "Saddle": (-2.0, 2.0), "Rosenbrock": (-1.5, 1.5),
                "Constrained Circle": (0.5, 0.5), "Double Constraint": (-1.5, 1.5),
                "Multi-Objective": (0.0, 0.0), "Ackley": (2.0, -2.0), "Rastrigin": (3.0, 3.0),
                "Styblinski-Tang": (-2.5, -2.5), "Sphere": (-3.0, 3.0), "Himmelblau": (0.0, 0.0),
                "Booth": (1.0, 1.0), "Beale": (-2.0, 2.0)
            }
            default_x, default_y = start_xy_defaults.get(func_name, (-3.0, 3.0))
            default_lr = 0.005
            default_steps = 50

            def run_auto_tuning_simulation(f_func, optimizer, x0, y0):
                lr_grid = list(np.logspace(-4, -1, 6))
                step_grid = [20, 30, 40, 50, 60, 80]
                best_score = float("inf")
                best_lr, best_steps = lr_grid[0], step_grid[0]
                logs = []

                for lr in lr_grid:
                    for steps in step_grid:
                        x_t, y_t = x0, y0
                        m, v = np.zeros(2), np.zeros(2)
                        beta1, beta2, eps = 0.9, 0.999, 1e-8
                        for t in range(1, steps + 1):
                            dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
                            dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
                            grad = np.array([dx, dy])
                            if np.linalg.norm(grad) < 1e-3:
                                break
                            if optimizer == "Adam":
                                m = beta1 * m + (1 - beta1) * grad
                                v = beta2 * v + (1 - beta2) * (grad ** 2)
                                m_hat = m / (1 - beta1 ** t)
                                v_hat = v / (1 - beta2 ** t)
                                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                            elif optimizer == "RMSProp":
                                v = beta2 * v + (1 - beta2) * (grad ** 2)
                                update = lr * grad / (np.sqrt(v) + eps)
                            else:
                                update = lr * grad
                            x_t -= update[0]
                            y_t -= update[1]
                        loss = f_func(x_t, y_t)
                        score = loss + 0.01 * steps
                        logs.append({"lr": lr, "steps": steps, "loss": loss, "score": score})
                        if score < best_score:
                            best_score, best_lr, best_steps = score, lr, steps
                st.session_state.df_log = pd.DataFrame(logs)
                return best_lr, best_steps

            if auto_tune:
                # Use fresh sympy symbols to avoid conflict with Taylor's x
                x_sym, y_sym, w_sym = sp.symbols("x y w")
                symbolic_expr = predefined_funcs[func_name][0]

                if func_name == "Multi-Objective" and w_val is not None:
                    symbolic_expr = symbolic_expr.subs(w_sym, w_val)

                f_lambdified = sp.lambdify((x_sym, y_sym), symbolic_expr, modules="numpy")

                best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)
                st.success(f"✅ Auto-tuned: lr={best_lr}, steps={best_steps}, start=({default_x},{default_y})")
                default_lr, default_steps = best_lr, best_steps

            # Set in session_state
            if 'params_set' not in st.session_state or st.button("🔄 Reset to Auto-Tuned"):
                st.session_state.lr = default_lr
                st.session_state.steps = default_steps
                st.session_state.start_x = default_x
                st.session_state.start_y = default_y
                st.session_state.params_set = True

            # Final user inputs

            # ✅ Only show lr and steps when not using backtracking or Newton
            if not (
                optimizer == "GradientDescent" and options.get("use_backtracking", False)
            ) and optimizer != "Newton's Method":
                lr = st.selectbox("Learning Rate", sorted(set([0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, default_lr])), index=0, key="lr")
                steps = st.slider("Steps", 10, 100, value=st.session_state.get("steps", 50), key="steps")
            elif optimizer == "Newton's Method":
                lr = None
                steps = None
                st.info("📌 Newton’s Method computes its own step size using the Hessian inverse — learning rate is not needed.")
            elif optimizer == "GradientDescent" and options.get("use_backtracking", False):
                lr = None
                steps = None
                st.info("📌 Using Backtracking Line Search — no need to set learning rate or step count.")



            # Ensure session_state values exist to avoid AttributeError
            if "start_x" not in st.session_state:
                st.session_state["start_x"] = -3.0
            if "start_y" not in st.session_state:
                st.session_state["start_y"] = 3.0


            st.slider("Initial x", -5.0, 5.0, st.session_state.start_x, key="start_x")
            st.slider("Initial y", -5.0, 5.0, st.session_state.start_y, key="start_y")
            # st.checkbox("🎮 Animate Descent Steps")
            show_animation = st.checkbox("🎮 Animate Descent Steps", key="show_animation")

        if auto_tune and optimizer in ["GradientDescent", "Adam", "RMSProp"]:
            with col_right:
                st.markdown("### 📊 Auto-Tuning Trial Log")
                if "df_log" in st.session_state:
                    st.dataframe(st.session_state.df_log.sort_values("score").reset_index(drop=True))
                    st.markdown("""
                    **🧠 How to Read Score:**
                    - `score = final_loss + penalty × steps`
                    - ✅ Lower score is better (fast and accurate convergence).
                    """)
                else:
                    st.info("Auto-tuning not yet triggered.")
                #             st.info("Auto-tuning not yet triggered.")


        if mode == "Predefined":
            f_expr, constraints, description = predefined_funcs[func_name]
            if func_name == "Multi-Objective" and w_val is not None:
                f_expr = f_expr.subs(w, w_val)
        else:
            try:
                f_expr = sp.sympify(expr_str)
                constraints = []
                description = "Custom function."
            except:
                st.error("Invalid expression.")
                st.stop()



        st.markdown(f"### 📘 Function Description:\n> {description}")

        # Setup for symbolic Lagrangian and KKT (if needed)
        L_expr = f_expr + sum(sp.Symbol(f"lambda{i+1}") * g for i, g in enumerate(constraints))
        # grad_L = [sp.diff(L_expr, v) for v in (x, y)]
        grad_L = [sp.diff(L_expr, v) for v in (x_sym, y_sym)]
        kkt_conditions = grad_L + constraints

        def backtracking_line_search_sym(f_sym, grad_f_sym, x0, y0, alpha0=1.0, beta=0.5, c=1e-4, max_iters=100):
            # Declare standard symbols for lambdify
            x, y = sp.symbols("x y")

            # Ensure all expressions use x, y (not x_sym, y_sym)
            f_sym = f_sym.subs({sp.Symbol("x"): x, sp.Symbol("y"): y})
            grad_f_sym = [g.subs({sp.Symbol("x"): x, sp.Symbol("y"): y}) for g in grad_f_sym]

            f_lambd = sp.lambdify((x, y), f_sym, modules='numpy')
            grad_f_lambd = [sp.lambdify((x, y), g, modules='numpy') for g in grad_f_sym]

            xk, yk = x0, y0
            path = [(xk, yk)]
            alphas = []

            for _ in range(max_iters):
                gx, gy = grad_f_lambd[0](xk, yk), grad_f_lambd[1](xk, yk)
                grad_norm = gx**2 + gy**2

                if grad_norm < 1e-10:
                    break

                alpha = alpha0
                while True:
                    x_new = xk - alpha * gx
                    y_new = yk - alpha * gy
                    lhs = f_lambd(x_new, y_new)
                    rhs = f_lambd(xk, yk) - c * alpha * grad_norm
                    if lhs <= rhs or alpha < 1e-8:
                        break
                    alpha *= beta

                xk, yk = xk - alpha * gx, yk - alpha * gy
                path.append((xk, yk))
                alphas.append(alpha)

            return path, alphas


        # def backtracking_line_search_sym(f_sym, grad_f_sym, x0, y0, alpha0=1.0, beta=0.5, c=1e-4, max_iters=100):
        #     f_lambd = sp.lambdify((x, y), f_sym, modules='numpy')
        #     grad_f_lambd = [sp.lambdify((x, y), g, modules='numpy') for g in grad_f_sym]

        #     xk, yk = x0, y0
        #     path = [(xk, yk)]
        #     alphas = []

        #     for _ in range(max_iters):
        #         gx, gy = grad_f_lambd[0](xk, yk), grad_f_lambd[1](xk, yk)
        #         grad_norm = gx**2 + gy**2

        #         if grad_norm < 1e-10:
        #             break

        #         alpha = alpha0
        #         while True:
        #             x_new = xk - alpha * gx
        #             y_new = yk - alpha * gy
        #             lhs = f_lambd(x_new, y_new)
        #             rhs = f_lambd(xk, yk) - c * alpha * grad_norm
        #             if lhs <= rhs or alpha < 1e-8:
        #                 break
        #             alpha *= beta

        #         xk, yk = xk - alpha * gx, yk - alpha * gy
        #         path.append((xk, yk))
        #         alphas.append(alpha)

        #     return path, alphas


        def optimize_path(x0, y0, optimizer, lr, steps, f_func, grad_f=None, hessian_f=None, options=None):
            path = [(x0, y0)]
            options = options or {}
            meta = {} 

            # --- Special Cases: Return early ---
            if optimizer == "GradientDescent" and options.get("use_backtracking", False):
                grad_f_expr = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
                path, alphas = backtracking_line_search_sym(f_expr, grad_f_expr, x0, y0)
                meta["callback_steps"] = len(path)
                return path, alphas, meta


            if optimizer == "Newton's Method" and options.get("newton_variant") in ["BFGS", "L-BFGS"]:
                from scipy.optimize import minimize

                path_coords = []
                losses = []

                def loss_vec(v):
                    return f_func(v[0], v[1])

                def grad_vec(v):
                    return np.atleast_1d(grad_f(v[0], v[1]))

                def callback(vk):
                    path_coords.append((vk[0], vk[1]))
                    losses.append(f_func(vk[0], vk[1]))

                x0_vec = np.array([x0, y0])
                method = "L-BFGS-B" if options["newton_variant"] == "L-BFGS" else "BFGS"

                res = minimize(loss_vec, x0_vec, method=method, jac=grad_vec, callback=callback, options={"maxiter": steps})

                # Fallback if callback was never triggered (e.g., early convergence)
                if not path_coords:
                    path_coords = [tuple(res.x)]
                    losses = [f_func(*res.x)]

                meta["res_nit"] = res.nit
                meta["callback_steps"] = len(path_coords)
                return path_coords, losses, meta


            if optimizer == "Simulated Annealing":
                T, cooling = options.get("T", 2.0), options.get("cooling", 0.95)
                current = f_func(x0, y0)
                for _ in range(steps):
                    xn, yn = x0 + np.random.randn(), y0 + np.random.randn()
                    candidate = f_func(xn, yn)
                    if candidate < current or np.random.rand() < np.exp(-(candidate - current)/T):
                        x0, y0 = xn, yn
                        current = candidate
                        path.append((x0, y0))
                    T *= cooling
                meta["callback_steps"] = len(path)
                return path, None, meta
            
            if optimizer == "Genetic Algorithm":
                pop_size = options.get("pop_size", 20)
                mutation_std = options.get("mutation_std", 0.3)
                pop = [np.random.uniform(-5, 5, 2) for _ in range(pop_size)]
                for _ in range(steps // 2):
                    pop = sorted(pop, key=lambda p: f_func(p[0], p[1]))[:pop_size // 2]
                    children = [np.mean([pop[i], pop[j]], axis=0) + np.random.normal(0, mutation_std, 2)
                                for i in range(len(pop)) for j in range(i+1, len(pop))][:pop_size // 2]
                    pop += children
                best = sorted(pop, key=lambda p: f_func(p[0], p[1]))[0]
                path = [tuple(best)]
                meta["callback_steps"] = len(path)
                return path, None, meta

            # --- Standard Optimizer Loop ---
            m, v = np.zeros(2), np.zeros(2)
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for t in range(1, steps + 1):
                x_t, y_t = path[-1]
                grad = grad_f(x_t, y_t)

                if optimizer == "Adam":
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    update = lr * m_hat / (np.sqrt(v_hat) + eps)

                elif optimizer == "RMSProp":
                    v = beta2 * v + (1 - beta2) * (grad ** 2)
                    update = lr * grad / (np.sqrt(v) + eps)

                elif optimizer == "Newton's Method":
                    variant = options.get("newton_variant", "Classic Newton")
                    if variant == "Classic Newton":
                        try:
                            H = hessian_f(x_t, y_t)
                            H_inv = np.linalg.inv(H)
                            update = H_inv @ grad
                        except:
                            update = grad
                    elif variant == "Numerical Newton":
                        eps_num = 1e-4
                        def fx(x_, y_): return f_func(x_, y_)
                        def second_partial(f, x, y, i, j):
                            h = eps_num
                            if i == 0 and j == 0:
                                return (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
                            elif i == 1 and j == 1:
                                return (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
                            else:
                                return (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
                        H = np.array([
                            [second_partial(fx, x_t, y_t, 0, 0), second_partial(fx, x_t, y_t, 0, 1)],
                            [second_partial(fx, x_t, y_t, 1, 0), second_partial(fx, x_t, y_t, 1, 1)]
                        ])
                        try:
                            update = np.linalg.inv(H) @ grad
                        except:
                            update = grad

                else:  # Default: GradientDescent
                    update = lr * grad

                update = np.asarray(update)
                new_x, new_y = x_t - update[0], y_t - update[1]
                path.append((new_x, new_y))

            meta["callback_steps"] = len(path)
            return path, None, meta


        if optimizer == "Newton's Method":
            with st.expander("🧠 Newton Method Variants Explained", expanded=False):
                st.markdown("### 📘 Classic Newton vs. Numerical vs. Quasi-Newton")
                st.markdown("Newton's Method is a powerful optimization technique that uses **second-order derivatives** or their approximations to accelerate convergence.")

                st.markdown("#### 🧮 Classic Newton (Symbolic)")
                st.markdown("- Uses the **symbolic Hessian matrix** from calculus:")
                st.latex(r"\nabla^2 f(x, y)")
                st.markdown("- ✅ Very efficient and accurate for simple analytic functions (e.g., quadratic, convex).")
                st.markdown("- ⚠️ Can fail or be unstable if the Hessian is singular or badly conditioned.")

                st.markdown("#### 🔢 Numerical Newton")
                st.markdown("- Uses **finite differences** to approximate the Hessian.")
                st.markdown("- No need for symbolic derivatives.")
                st.markdown("- ✅ More robust for complex or unknown functions.")
                st.markdown("- 🐢 Slightly slower due to extra evaluations.")

                st.markdown("#### 🔁 BFGS / L-BFGS (Quasi-Newton)")
                st.markdown("- ✅ Avoids computing the full Hessian.")
                st.markdown("- Builds curvature estimate using gradients:")
                st.latex(r"""
                H_{k+1} = H_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{H_k s_k s_k^T H_k}{s_k^T H_k s_k}
                """)
                st.markdown("Where:")
                st.latex(r"s_k = x_{k+1} - x_k")
                st.latex(r"y_k = \nabla f(x_{k+1}) - \nabla f(x_k)")
                st.markdown("- 🧠 **BFGS**: High accuracy, stores full matrix.")
                st.markdown("- 🪶 **L-BFGS**: Stores only a few recent updates — ideal for high-dimensional problems.")
                st.markdown("💡 Quasi-Newton methods **approximate** curvature and still converge fast — especially useful for functions like Rosenbrock!")

                st.markdown("---")
                st.markdown("### ✏️ Why No Learning Rate?")
                st.markdown("Newton’s Method computes:")
                st.latex(r"x_{t+1} = x_t - H^{-1} \nabla f(x_t)")
                st.markdown("So it **naturally determines the best step direction and size** — no need for manual tuning like in gradient descent.")

        # === Shared simulation function ===
        def simulate_optimizer(opt_name, f_expr, lr=0.01, steps=50):
            f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
            # f_func = sp.lambdify((x, y), f_expr, modules="numpy")
            x0, y0 = -3, 3
            path = [(x0, y0)]
            m, v = 0, 0
            for t in range(1, steps + 1):
                x_t, y_t = path[-1]
                dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
                dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
                g = np.array([dx, dy])
                if opt_name == "Adam":
                    m = 0.9 * m + 0.1 * g
                    v = 0.999 * v + 0.001 * (g ** 2)
                    m_hat = m / (1 - 0.9 ** t)
                    v_hat = v / (1 - 0.999 ** t)
                    update = lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                elif opt_name == "RMSProp":
                    v = 0.999 * v + 0.001 * (g ** 2)
                    update = lr * g / (np.sqrt(v) + 1e-8)
                elif opt_name == "Newton's Method":
                    hess = sp.hessian(f_expr, (x, y))
                    hess_func = sp.lambdify((x, y), hess, modules="numpy")
                    try:
                        H = np.array(hess_func(x_t, y_t))
                        H_inv = np.linalg.inv(H)
                        update = H_inv @ g
                    except:
                        update = g
                else:
                    update = lr * g
                x_new, y_new = x_t - update[0], y_t - update[1]
                path.append((x_new, y_new))
            final_x, final_y = path[-1]
            grad_norm = np.linalg.norm(g)
            return {
                "Optimizer": opt_name,
                "Final Value": round(f_func(final_x, final_y), 4),
                "Gradient Norm": round(grad_norm, 4),
                "Steps": len(path) - 1
            }
        g_funcs = [sp.lambdify((x_sym, y_sym), g, modules=["numpy"]) for g in constraints]  
        # g_funcs = [sp.lambdify((x, y), g, modules=["numpy"]) for g in constraints]
        f_func = sp.lambdify((x_sym, y_sym), f_expr, modules=["numpy"])
        # f_func = sp.lambdify((x, y), f_expr, modules=["numpy"])
        grad_f = lambda x0, y0: np.array([
            (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
            (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
        ])

        def hessian_f(x0, y0):
            hess_expr = sp.hessian(f_expr, (x, y))
            hess_func = sp.lambdify((x, y), hess_expr, modules=["numpy"])
            return np.array(hess_func(x0, y0))

        # === Pull final values from session_state
        start_x = st.session_state.get("start_x", -3.0)
        start_y = st.session_state.get("start_y", 3.0)
        lr = st.session_state.get("lr", 0.01)
        steps = st.session_state.get("steps", 50)


        path, alpha_log, meta = optimize_path(
            start_x, start_y,
            optimizer=optimizer,
            lr=lr,
            steps=steps,
            f_func=f_func,
            grad_f=grad_f,
            hessian_f=hessian_f,
            options=options
        )

        xs, ys = zip(*path)
        Z_path = [f_func(xp, yp) for xp, yp in path]

        x_vals = np.linspace(-5, 5, 200)
        y_vals = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = f_func(X, Y)

        # --- Taylor Expansion Toggle ---
        show_taylor = st.checkbox("📐 Show Taylor Approximation at (a, b)", value=False)

        show_2nd = False
        Z_t1 = None
        Z_t2 = None

        expansion_point = (a_val, b_val) if show_taylor else None

        if show_taylor:
            st.markdown("**Taylor Expansion Center (a, b)**")
            a_val = st.slider("a (expansion x)", -5.0, 5.0, float(start_x), step=0.1)
            b_val = st.slider("b (expansion y)", -5.0, 5.0, float(start_y), step=0.1)
            show_2nd = st.checkbox("Include 2nd-order terms", value=True)

            # --- Symbolic derivatives ---
            grad_fx = [sp.diff(f_expr, var) for var in (x_sym, y_sym)]
            hess_fx = sp.hessian(f_expr, (x_sym, y_sym))
            subs = {x_sym: a_val, y_sym: b_val}
            dx, dy = x_sym - a_val, y_sym - b_val

            f_ab = f_expr.subs(subs)
            grad_vals = [g.subs(subs) for g in grad_fx]
            hess_vals = hess_fx.subs(subs)

            # 1st and 2nd-order Taylor expressions
            T1_expr = f_ab + grad_vals[0]*dx + grad_vals[1]*dy
            T2_expr = T1_expr + 0.5 * (
                hess_vals[0, 0]*dx**2 + 2*hess_vals[0, 1]*dx*dy + hess_vals[1, 1]*dy**2
            )

            t1_np = sp.lambdify((x_sym, y_sym), T1_expr, "numpy")
            t2_np = sp.lambdify((x_sym, y_sym), T2_expr, "numpy") if show_2nd else None
            Z_t1 = t1_np(X, Y)
            Z_t2 = t2_np(X, Y) if show_2nd else None

        from visualizer import plot_3d_descent, plot_2d_contour  # Or use inline if not modularized


        st.markdown("### 📈 3D View")
        plot_3d_descent(
            x_vals=x_vals,
            y_vals=y_vals,
            Z=Z,
            path=path,
            Z_path=Z_path,
            Z_t1=Z_t1,
            Z_t2=Z_t2,
            show_taylor=show_taylor,
            show_2nd=show_2nd,
            expansion_point=expansion_point,
            f_func=f_func
        )

        st.markdown("### 🗺️ 2D View")
        plot_2d_contour(
            x_vals=x_vals,
            y_vals=y_vals,
            Z=Z,
            path=path,
            g_funcs=g_funcs if constraints else None,
            X=X, Y=Y,
            Z_t2=Z_t2,
            show_2nd=show_2nd,
            expansion_point=expansion_point
        )

        if show_taylor:
            st.caption("🔺 Red = 1st-order Taylor, 🔷 Blue = 2nd-order Taylor, 🟢 Green = true surface")

        if show_animation:
            frames = []
            fig_anim, ax_anim = plt.subplots(figsize=(5, 4))

            for i in range(1, len(path) + 1):
                ax_anim.clear()
                ax_anim.contour(X, Y, Z, levels=30, cmap="viridis")
                ax_anim.plot(*zip(*path[:i]), 'r*-')
                ax_anim.set_xlim([-5, 5])
                ax_anim.set_ylim([-5, 5])
                ax_anim.set_title(f"Step {i}/{len(path)-1}")

                buf = BytesIO()
                fig_anim.savefig(buf, format='png', dpi=100)  # optional: set dpi
                buf.seek(0)
                frames.append(Image.open(buf).convert("P"))  # convert to palette for GIF efficiency
                buf.close()

            gif_buf = BytesIO()
            frames[0].save(
                gif_buf, format="GIF", save_all=True,
                append_images=frames[1:], duration=300, loop=0
            )
            gif_buf.seek(0)
            st.image(gif_buf, caption="📽️ Animated Descent Path", use_container_width=True)



    with st.expander("🧰 Optimizer Diagnostic Tools", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Optimizer Comparison")

            start_x = st.session_state.get("start_x", 0.0)
            start_y = st.session_state.get("start_y", 0.0)

            selected_opts = st.multiselect(
                "Optimizers",
                ["GradientDescent", "Adam", "RMSProp", "Newton's Method"],
                default=["GradientDescent"],
                key="compare"
            )
            fig_comp, ax_comp = plt.subplots(figsize=(4, 3))

            results = []
            summary_results = []

            for opt in selected_opts:
                path_opt, losses, meta= optimize_path(  # ✅ unpack both outputs
                    start_x,
                    start_y,
                    optimizer=opt,
                    lr=lr,
                    steps=steps,
                    f_func=f_func,
                    grad_f=grad_f,
                    hessian_f=hessian_f,
                    options=options
                )

                zs_coords = path_opt
                zs_vals = losses if losses is not None and len(losses) > 0 else [f_func(xp, yp) for xp, yp in zs_coords]
                grad_norm = float(np.linalg.norm(grad_f(*zs_coords[-1])))

                results.append((opt, zs_vals))

                summary_results.append({
                    "Optimizer": opt,
                    "Final Value": np.round(zs_vals[-1], 4),
                    "Gradient Norm": np.round(grad_norm, 4),
                    "Steps": len(zs_vals),
                    "Actual Steps": meta.get("res_nit", "N/A"),   # ✅ 使用 meta.get()
                    "Logged Steps": meta.get("callback_steps", "N/A")
                })

            # Sort results by final loss
            results.sort(key=lambda x: x[1][-1])

            for opt, zs in results:
                if zs is not None and len(zs) > 0:
                    ax_comp.plot(zs, label=f"{opt} ({len(zs)} steps)", marker="o", markersize=2)
                else:
                    st.warning(f"⚠️ No convergence data for {opt}")

            ax_comp.set_title("Convergence")
            ax_comp.set_xlabel("Step")
            ax_comp.set_ylabel("f(x, y)")
            ax_comp.set_ylim(bottom=0)
            ax_comp.legend()
            st.pyplot(fig_comp)

            # Show summary table
            st.markdown("#### 📋 Optimizer Summary Table")
            df_summary = pd.DataFrame(summary_results)
            st.dataframe(df_summary)



            st.markdown("#### 🔥 Gradient Norm Heatmap")
            # norm_grad = np.sqrt((np.gradient(Z, axis=0))**2 + (np.gradient(Z, axis=1))**2)
            # Temporary definition in case not computed above
            if 'Z_loss' not in locals():
                x_range = np.linspace(-5, 5, 50)
                y_range = np.linspace(-5, 5, 50)
                X_loss, Y_loss = np.meshgrid(x_range, y_range)
                Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2  # Default MSE-like loss

            norm_grad = np.sqrt((np.gradient(Z_loss, axis=0))**2 + (np.gradient(Z_loss, axis=1))**2)

            fig3, ax3 = plt.subplots()
            heat = ax3.imshow(norm_grad, extent=[-5, 5, -5, 5], origin='lower', cmap='plasma')
            fig3.colorbar(heat, ax=ax3, label="‖∇f‖")
            ax3.set_title("‖∇f(x, y)‖")
            st.pyplot(fig3)

        with col2:
            st.markdown("#### 🌄 Loss Surface")

            loss_type = st.radio("Loss Type", ["MSE", "Log Loss", "Cross Entropy", "Custom"])

            # Create input grid
            x_range = np.linspace(-5, 5, 50)  # reduced resolution for arrows
            y_range = np.linspace(-5, 5, 50)
            X_loss, Y_loss = np.meshgrid(x_range, y_range)

            # Compute Z surface and manually define minimum
            if loss_type == "MSE":
                Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2
                min_x, min_y = 2, -1
            elif loss_type == "Log Loss":
                Z_loss = np.log(1 + np.exp(-(X_loss + Y_loss)))
                min_x, min_y = 5, -5
            elif loss_type == "Cross Entropy":
                p = 1 / (1 + np.exp(-(X_loss + Y_loss)))
                Z_loss = -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
                min_x, min_y = 5, -5
            else:
                Z_loss = np.sin(X_loss) * np.cos(Y_loss)
                min_x, min_y = 0, 0

            # Compute gradients (numerical partial derivatives)
            dZ_dx, dZ_dy = np.gradient(Z_loss, x_range, y_range)

            # === Side-by-side plots ===
            col1, col2 = st.columns(2)

            with col1:
                fig3d = plt.figure(figsize=(5, 4))
                ax3d = fig3d.add_subplot(111, projection='3d')
                ax3d.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis', edgecolor='none', alpha=0.9)
                ax3d.scatter(min_x, min_y, np.min(Z_loss), color='red', s=50, label='Min')
                ax3d.set_title(f"{loss_type} Surface")
                ax3d.legend()
                st.pyplot(fig3d)

            with col2:
                fig2d, ax2d = plt.subplots(figsize=(5, 4))
                contour = ax2d.contourf(X_loss, Y_loss, Z_loss, levels=30, cmap='viridis')
                ax2d.plot(min_x, min_y, 'ro', label='Min')
                # Overlay gradient arrows (negative for descent)
                ax2d.quiver(X_loss, Y_loss, -dZ_dx, -dZ_dy, color='white', alpha=0.7, scale=50)
                fig2d.colorbar(contour, ax=ax2d, label="Loss")
                ax2d.set_title(f"{loss_type} Contour View + Gradient Field")
                ax2d.legend()
                st.pyplot(fig2d)

            st.markdown("#### ✅ Constraint Checker")

            start_x = st.session_state.get("start_x", 0.0)
            start_y = st.session_state.get("start_y", 0.0)


            # ⛳ Ensure path is updated
            path, losses, meta = optimize_path(
                start_x,
                start_y,
                optimizer=selected_opts[0],  # or st.session_state.get("optimizer", "GradientDescent")
                lr=lr,
                steps=steps,
                f_func=f_func,
                grad_f=grad_f,
                hessian_f=hessian_f,
                options=options
            )
            if constraints:
                fig_con, ax_con = plt.subplots(figsize=(4, 3))
                for i, g_func in enumerate(g_funcs):
                    violations = [g_func(xp, yp) for xp, yp in path]
                    ax_con.plot(violations, label=f"g{i+1}(x, y)")
                ax_con.axhline(0, color="red", linestyle="--")
                ax_con.set_xlabel("Step")
                ax_con.set_ylabel("g(x, y)")
                ax_con.legend()
                st.pyplot(fig_con)
            else:
                st.info("No constraints defined.")
    


    # === Symbolic Analysis: KKT, Gradient & Hessian ===
    with st.expander("📐 Symbolic Analysis: KKT, Gradient & Hessian", expanded=False):
        st.markdown("#### 🎯 Objective & Lagrangian")
        st.latex(r"f(x, y) = " + sp.latex(f_expr))
        st.latex(r"\mathcal{L}(x, y, \lambda) = " + sp.latex(L_expr))

        st.markdown("#### ✅ KKT Conditions")
        for i, cond in enumerate(kkt_conditions):
            st.latex(fr"\text{{KKT}}_{{{i+1}}} = {sp.latex(cond)}")

        st.markdown("#### 🧮 Gradient & Hessian")
        grad = [sp.diff(f_expr, v) for v in (x_sym, y_sym)]
        # grad = [sp.diff(f_expr, v) for v in (x, y)]
        # hessian = sp.hessian(f_expr, (x, y))
        hessian = sp.hessian(f_expr, (x_sym, y_sym))

        st.latex("Gradient: " + sp.latex(sp.Matrix(grad)))
        st.latex("Hessian: " + sp.latex(hessian))

        if optimizer == "Newton's Method":
            st.markdown("#### 🧠 Newton Method Diagnostics")        
            hess = sp.hessian(f_expr, (x_sym, y_sym))
            hess_func = sp.lambdify((x_sym, y_sym), hess, modules="numpy")
            H_val = np.array(hess_func(start_x, start_y))
            det_val = np.linalg.det(H_val)

            st.markdown("#### 📐 Hessian Matrix")
            st.latex(r"\text{H}(x, y) = " + sp.latex(hess))

            st.markdown("#### 📏 Determinant")
            st.latex(r"\det(\text{H}) = " + sp.latex(sp.det(hess)))

            if np.isclose(det_val, 0, atol=1e-6):
                st.error("❌ Determinant is zero — Newton's Method cannot proceed (singular Hessian).")
            elif det_val < 0:
                st.warning("⚠️ Negative determinant — may indicate a saddle point or non-convex region.")
            else:
                st.success("✅ Hessian is suitable for Newton's Method descent.")

    # === Newton Method Info ===

# # === Mode: LLM Assistant ===
# elif mode == "🤖 LLM Assistant":
#     st.subheader("🤖 LLM Assistant: Explore Your Data Intelligently")

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = []

#     uploaded_file = st.file_uploader("📁 Upload a dataset (CSV)", type=["csv"])
#     if uploaded_file:
#         st.session_state.uploaded_file = uploaded_file

#     df = None
#     if "uploaded_file" in st.session_state:
#         try:
#             if st.session_state.uploaded_file is not None:
#                 content = st.session_state.uploaded_file.read()
#                 if content.strip() == b"":
#                     raise ValueError("Uploaded file is empty.")
#                 st.session_state.uploaded_file.seek(0)
#                 df = pd.read_csv(st.session_state.uploaded_file)
#             else:
#                 raise ValueError("No file uploaded.")
#         except Exception as e:
#             st.error(f"❌ Failed to read uploaded CSV: {e}")
#             st.session_state.uploaded_file = None
#             df = None

#     if df is not None:
#         st.write("### 📄 Data Preview", df.head())
#         st.write("### 📊 Summary Statistics")
#         st.dataframe(df.describe(include='all'))

#         st.markdown("### 💡 Suggested Prompts")
#         st.markdown("""
#         - What are the most correlated features?
#         - Show a summary of missing values
#         - Which features influence the target most?
#         - What kind of plot would help visualize X vs Y?
#         - Can you generate a histogram of column X?
#         - Show pairwise plots for selected features
#         - Predict the target using linear regression
#         - Detect outliers or anomalies in the data
#         """)

#     # === LLM Chat ===

#     llm_choice = st.radio("Choose LLM Backend", ["OpenAI", "Local LLM"])
#     if llm_choice == "OpenAI":
#         def query_llm(prompt: str) -> str:
#             resp = client.ChatCompletion.create(
#                 model="gpt-4o",
#                 messages=[ ... ],
#             )
#             return resp.choices[0].message.content.strip()
#     else:
#         def query_llm(prompt: str) -> str:
#             return query_local_llm(prompt)

#     # ─── user input / querying ────────────────────────────────
#     user_input = st.text_input("💬 Ask something (about your data):")
#     if user_input:
#         with st.spinner("🤖 Thinking..."):
#             # build your prompt exactly once
#             if df is not None:
#                 summary = df.describe(include="all").to_string()
#                 corr    = df.corr(numeric_only=True).round(3).to_string()
#                 full_prompt = (
#                     f"You are a data analysis assistant.\n\n"
#                     f"Dataset summary:\n{summary}\n\n"
#                     f"Correlation matrix:\n{corr}\n\n"
#                     f"Q: {user_input}\nA:"
#                 )
#             else:
#                 full_prompt = user_input

#             try:
#                 answer = query_llm(full_prompt)
#                 st.session_state.chat_history.append((user_input, answer))
#             except Exception as e:
#                 st.error(f"❌ LLM Error: {e}")
#                 answer = None

#         if answer:
#             st.markdown(
#                 f"<div style='background-color:#e8f5e9;padding:10px;border-radius:8px;'>{answer}</div>",
#                 unsafe_allow_html=True,
#             )

#     if df is not None:             
#         st.markdown("### 📈 Custom Chart Generator")
#         chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
#         x_col = st.selectbox("X-axis Column", df.columns)
#         y_col = st.selectbox("Y-axis Column", df.columns)
#         if st.button("Generate Chart"):
#             fig, ax = plt.subplots()
#             if chart_type == "Line":
#                 ax.plot(df[x_col], df[y_col])
#             elif chart_type == "Bar":
#                 ax.bar(df[x_col], df[y_col])
#             elif chart_type == "Scatter":
#                 ax.scatter(df[x_col], df[y_col])
#             elif chart_type == "Histogram":
#                 ax.hist(df[x_col], bins=20)
#             ax.set_xlabel(x_col)
#             ax.set_ylabel(y_col)
#             st.pyplot(fig, use_container_width=True)

#     st.markdown("### 💾 Export Data")
#     file_name = st.text_input("Output file name (without extension)", "my_data")
#     if st.button("Download as CSV"):
#         tmp_csv = df.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             label="📥 Download Processed CSV",
#             data=tmp_csv,
#             file_name=f"{file_name}.csv",
#             mime="text/csv"
#         )



#     if st.session_state.chat_history:
#         st.markdown("### 📜 Chat History")
#         for q, a in st.session_state.chat_history[::-1]:
#             st.markdown(f"**You:** {q}")
#             st.markdown(f"**Assistant:** {a}")

#     if "uploaded_file" not in st.session_state or df is None:
#         st.info("📂 Upload a dataset to explore insights with the assistant.")


# === Mode: LLM Assistant ===


elif mode == "🤖 LLM Assistant":
    st.subheader("🤖 LLM Assistant: Explore Your Data Intelligently")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_file = st.file_uploader("📁 Upload a dataset (CSV)", type=["csv"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file

    df = None
    if "uploaded_file" in st.session_state:
        try:
            if st.session_state.uploaded_file is not None:
                content = st.session_state.uploaded_file.read()
                if content.strip() == b"":
                    raise ValueError("Uploaded file is empty.")
                st.session_state.uploaded_file.seek(0)
                df = pd.read_csv(st.session_state.uploaded_file)
            else:
                raise ValueError("No file uploaded.")
        except Exception as e:
            st.error(f"❌ Failed to read uploaded CSV: {e}")
            st.session_state.uploaded_file = None
            df = None

    if df is not None:
        st.write("### 📄 Data Preview", df.head())
        st.write("### 📊 Summary Statistics")
        st.dataframe(df.describe(include='all'))

        st.markdown("### 💡 Suggested Prompts")
        st.markdown("""
        - What are the most correlated features?
        - Show a summary of missing values
        - Which features influence the target most?
        - What kind of plot would help visualize X vs Y?
        - Can you generate a histogram of column X?
        - Show pairwise plots for selected features
        - Predict the target using linear regression
        - Detect outliers or anomalies in the data
        """)

    # === 🔐 API Key Selection ===
    st.markdown("### 🔑 LLM Settings")
    llm_choice = st.radio("Choose LLM Backend", ["OpenAI", "Local LLM"])
    user_key = st.text_input("Enter your OpenAI API key (optional):", type="password")

    final_key = user_key if user_key else st.secrets.get("OPENAI_API_KEY")

    query_llm = None

    if llm_choice == "OpenAI":
        st.warning("⚠️ Using OpenAI's GPT may incur charges.")
        st.markdown("[💳 View billing](https://platform.openai.com/account/billing)", unsafe_allow_html=True)
        allow_openai_use = st.checkbox("✅ I understand I may be billed for OpenAI usage", value=True)

        if allow_openai_use:
            from langchain.chat_models import ChatOpenAI
            from langchain.memory import ConversationBufferMemory

            llm = ChatOpenAI(temperature=0.3, openai_api_key=final_key)
            memory = ConversationBufferMemory()
            query_llm = lambda prompt: llm.predict(prompt)
        else:
            query_llm = lambda prompt: "⚠️ OpenAI API key missing or billing not confirmed."
    else:
        query_llm = lambda prompt: query_local_llm(prompt)

    # === 🧠 User Input ===
    user_input = st.text_input("💬 Ask something (about your data):")
    if user_input:
        with st.spinner("🤖 Thinking..."):
            if df is not None:
                summary = df.describe(include="all").to_string()
                corr = df.corr(numeric_only=True).round(3).to_string()
                full_prompt = (
                    f"You are a data analysis assistant.\n\n"
                    f"Dataset summary:\n{summary}\n\n"
                    f"Correlation matrix:\n{corr}\n\n"
                    f"Q: {user_input}\nA:"
                )
            else:
                full_prompt = user_input

            try:
                st.code(full_prompt, language="text")  # ✅ show prompt for debug
                st.info("📡 Sending prompt to LLM...")
                answer = query_llm(full_prompt)

                if isinstance(answer, str) and answer.strip():
                    st.session_state.chat_history.append((user_input, answer))
                else:
                    st.warning("⚠️ LLM returned an empty or invalid response.")


            except Exception as e:
                st.error(f"❌ LLM Error: {e}")
                answer = None

        if answer and isinstance(answer, str) and answer.strip():
            st.markdown(
                f"<div style='background-color:#e8f5e9;padding:10px;border-radius:8px;'>{answer}</div>",
                unsafe_allow_html=True,
            )


    if df is not None:
        st.markdown("### 📈 Custom Chart Generator")
        chart_type = st.selectbox("Select Chart Type", ["Line", "Bar", "Scatter", "Histogram"])
        x_col = st.selectbox("X-axis Column", df.columns)
        y_col = st.selectbox("Y-axis Column", df.columns)
        if st.button("Generate Chart"):
            fig, ax = plt.subplots()
            if chart_type == "Line":
                ax.plot(df[x_col], df[y_col])
            elif chart_type == "Bar":
                ax.bar(df[x_col], df[y_col])
            elif chart_type == "Scatter":
                ax.scatter(df[x_col], df[y_col])
            elif chart_type == "Histogram":
                ax.hist(df[x_col], bins=20)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            st.pyplot(fig, use_container_width=True)

    st.markdown("### 💾 Export Data")
    file_name = st.text_input("Output file name (without extension)", "my_data")
    if st.button("Download as CSV"):
        tmp_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Processed CSV",
            data=tmp_csv,
            file_name=f"{file_name}.csv",
            mime="text/csv"
        )

    if st.session_state.chat_history:
        st.markdown("### 📜 Chat History")
        for q, a in st.session_state.chat_history[::-1]:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Assistant:** {a}")

    if "uploaded_file" not in st.session_state or df is None:
        st.info("📂 Upload a dataset to explore insights with the assistant.")





elif mode == "🧠 Unsupervised Learning":
    st.subheader("🧠 Unsupervised Learning Explorer")

    # Submodule selection in sidebar
    unsup_option = st.sidebar.radio("Select Unsupervised Module", [
        "Clustering",
        "Dimensionality Reduction",
        # "Anomaly Detection",
        # Add more modules as needed
    ])

    # Render selected module
    if unsup_option == "Clustering":
        from clustering import clustering_ui
        clustering_ui()
    
    elif unsup_option == "Dimensionality Reduction":
        from dim_reduction import dim_reduction_ui
        dim_reduction_ui()








# Footer
st.markdown("---")
st.info("Switch between ML tasks and optimization demos using the sidebar.")
