import streamlit as st

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
def supervised_ui():
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


