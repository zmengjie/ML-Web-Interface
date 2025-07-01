import streamlit as st
import pandas as pd
import numpy as np
st.set_page_config(page_title="Optimizer Playground", layout="wide")

import sympy as sp
import streamlit.components.v1 as components

import time
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score
)
from sympy import symbols, sympify, lambdify
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from io import BytesIO
from PIL import Image, ImageSequence


# st.set_page_config(layout="wide")
st.title("📊 Easy Visualizer")


# 📦 Future Features Block (Placeholder for Modular Expansion)
st.markdown("## 🧪 Experimental Modules (Coming Soon)")

with st.expander("🎮 ML Playground (No Upload Needed)"):
    st.markdown("""
    A hands-on playground to explore common ML models:
    - Try **Linear**, **Logistic**, **Naive Bayes**, **SVM**, **Decision Trees**
    - Adjust hyperparameters (e.g., alpha, C, kernel)
    - Visualize decision boundaries, performance metrics, and training curves
    - Great for non-computing users to experiment with models
    """)


with st.expander("📈 Supervised Learning Suite"):
    st.markdown("""
    Coming soon: classification & regression tools
    - 📌 **Naive Bayes Classifier** for categorical prediction
    - 📌 **Decision Trees & SVMs** with interactive hyperparameter sliders
    - 📌 Enhanced regression with error distributions
    """)

with st.expander("🧬 Unsupervised Learning Suite"):
    st.markdown("""
    Add structure discovery tools:
    - 🔍 **K-Means Clustering** (interactive cluster count)
    - 🔍 **PCA-based Dimensionality Reduction**
    - 🔍 Cluster visualizations in 2D and 3D
    """)

# Initialize sidebar guide state
if "show_guide" not in st.session_state:
    st.session_state.show_guide = False

# # Toggle button (place inside main layout, not sidebar)
# if st.button("📘 Toggle Guide (Sidebar)", key="toggle_guide_button"):
#     st.session_state.show_guide = not st.session_state.show_guide


with st.sidebar.expander("📘 Help & Guide", expanded=st.session_state.get("show_guide", False)):
    st.markdown("""
This interactive playground allows you to:
- Select from **predefined or custom mathematical functions**
- Choose optimizers like **Gradient Descent**, **Adam**, or **RMSProp**
- Explore **constraint-based optimization** using Lagrangian and KKT conditions
- Visualize the **3D surface**, **2D contours**, and **gradient norm heatmaps**
- Animate the optimization steps
- Symbolically check KKT and gradients for education and validation

---

### 🛠️ Instructions

1. **Choose a Function**  
   Select from built-in functions like *Quadratic Bowl*, *Rosenbrock*, or define your own with `f(x, y)`.

2. **Configure Optimizer Settings**  
   Adjust learning rate, optimizer type, and initial (x, y) point to test how descent behaves.

3. **Run and Observe**  
   Press run to watch animated optimizer steps on both 3D surface and 2D contours.

4. **Apply Constraints**  
   Toggle predefined constraints to explore constrained optimization via Lagrangian method.

5. **KKT Checker**  
   This displays the **Karush-Kuhn-Tucker (KKT)** conditions:
   - **Stationarity**: ∇L = 0
   - **Primal Feasibility**: Constraints g(x, y) ≤ 0 must hold
   - **Dual Feasibility**: λ ≥ 0 for all multipliers
   - **Complementary Slackness**: λᵢ·gᵢ(x) = 0 for each constraint

   These conditions are **necessary** for optimality in constrained problems (under convexity). Use them to validate symbolic setup and explore edge conditions.

---

### 💡 Tips & Recommendations
- Try *Quadratic Bowl* first with Gradient Descent to see smooth convergence
- Use *Rosenbrock* or *Saddle* to test optimizer sensitivity and divergence
- In constraint modes, visualize feasible region (red) and Lagrangian surface
- Gradient norm heatmap helps understand how steep or flat each region is
- Use symbolic validator for classroom demonstrations or algorithm debugging

""")

st.title("🧮 Optimizer Visual Playground")

x, y, w = sp.symbols("x y w")

predefined_funcs = {
    "Quadratic Bowl": (x**2 + y**2, [], "Convex bowl, global min at origin."),
    "Saddle": (x**2 - y**2, [], "Saddle point at origin, non-convex."),
    "Rosenbrock": ((1 - x)**2 + 100 * (y - x**2)**2, [], "Banana-shaped curved valley, classic test function."),
    "Constrained Circle": (x * y, [x + y - 1], "Constrained optimization with line x + y = 1."),
    "Double Constraint": (x**2 + y**2, [x + y - 1, x**2 + y**2 - 4], "Circular + linear intersection constraints."),
    "Multi-Objective": (w * ((x - 1)**2 + (y - 2)**2) + (1 - w) * ((x + 2)**2 + (y + 1)**2), [], "Weighted sum of two loss terms.")
}

with st.expander("🚀 Optimizer Visual Playground", expanded=True):
    col_sidebar, col_main = st.columns([1, 3])

    with col_sidebar:
        # 🧭 Guide Button (inside playground block)
        if st.button("📘 Show Guide (Sidebar)", key="show_guide_button"):
            st.session_state.show_guide = True

        st.markdown("## ⚙️ Configuration")

        mode = st.radio("Function Source", ["Predefined", "Custom"])
        func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
        expr_str = st.text_input("Enter function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""
        w_val = st.slider("Weight w (Multi-Objective)", 0.0, 1.0, 0.5) if func_name == "Multi-Objective" else None

        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        steps = st.slider("Steps", 10, 100, 50)
        start_x = st.slider("Initial x", -5.0, 5.0, -3.0)
        start_y = st.slider("Initial y", -5.0, 5.0, 3.0)
        optimizer = st.selectbox("Optimizer", ["GradientDescent", "Adam", "RMSProp"])
        show_animation = st.checkbox("🎞️ Animate Descent Steps")

    if mode == "Predefined":
        f_expr, constraints, description = predefined_funcs[func_name]
        f_expr = f_expr.subs(w, w_val) if func_name == "Multi-Objective" else f_expr
    else:
        try:
            f_expr = sp.sympify(expr_str)
            constraints = []
            description = "Custom function."
        except:
            st.error("Invalid expression.")
            st.stop()

    st.markdown(f"### 📘 Function Description:\n> {description}")

    # --- Symbolic Setup ---
    L_expr = f_expr + sum(sp.Symbol(f"lambda{i+1}") * g for i, g in enumerate(constraints))
    grad_L = [sp.diff(L_expr, v) for v in (x, y)]
    kkt_conditions = grad_L + constraints

    # --- Lambdify ---
    f_func = sp.lambdify((x, y), f_expr, modules=["numpy"])
    g_funcs = [sp.lambdify((x, y), g, modules=["numpy"]) for g in constraints]
    grad_f = lambda x0, y0: np.array([
        (f_func(x0 + 1e-5, y0) - f_func(x0 - 1e-5, y0)) / 2e-5,
        (f_func(x0, y0 + 1e-5) - f_func(x0, y0 - 1e-5)) / 2e-5
    ])

    # --- Optimizer Path ---
    def optimize_path(x0, y0, optimizer="GradientDescent", lr=0.01, steps=50):
        path = [(x0, y0)]
        m, v = np.zeros(2), np.zeros(2)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for t in range(1, steps + 1):
            x_t, y_t = path[-1]
            grad = grad_f(x_t, y_t)
            if optimizer == "Adam":
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                update = lr * m_hat / (np.sqrt(v_hat) + eps)
            elif optimizer == "RMSProp":
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                update = lr * grad / (np.sqrt(v) + eps)
            else:
                update = lr * grad
            path.append((x_t - update[0], y_t - update[1]))
        return path


    path = optimize_path(start_x, start_y, optimizer=optimizer, lr=lr, steps=steps)
    xs, ys = zip(*path)
    Z_path = [f_func(xp, yp) for xp, yp in path]

    # --- Plotting ---
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

    # --- Animation (GIF) ---
    if show_animation:
        frames = []
        fig_anim, ax_anim = plt.subplots(figsize=(5, 4))
        ax_anim.contour(X, Y, Z, levels=30, cmap="viridis")
        for i in range(1, len(path) + 1):
            ax_anim.clear()
            ax_anim.contour(X, Y, Z, levels=30, cmap="viridis")
            ax_anim.plot(*zip(*path[:i]), 'r*-')
            ax_anim.set_title(f"Step {i}/{len(path)-1}")
            buf = BytesIO()
            fig_anim.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img.copy())
            buf.close()
        gif_buf = BytesIO()
        frames[0].save(gif_buf, format="GIF", save_all=True, append_images=frames[1:], duration=300, loop=0)
        gif_buf.seek(0)
        st.image(gif_buf, caption="Animated Descent Path", use_column_width=True)


with st.expander("🧰 Optimizer Diagnostic Tools", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Optimizer Comparison")
        selected_opts = st.multiselect("Optimizers", ["GradientDescent", "Adam", "RMSProp"], default=["GradientDescent", "Adam"], key="compare")
        fig_comp, ax_comp = plt.subplots(figsize=(4, 3))
        for opt in selected_opts:
            path_opt = optimize_path(start_x, start_y, optimizer=opt, lr=lr, steps=steps)
            zs = [f_func(xp, yp) for xp, yp in path_opt]
            ax_comp.plot(zs, label=opt)
        ax_comp.set_title("Convergence")
        ax_comp.set_xlabel("Step")
        ax_comp.set_ylabel("f(x, y)")
        ax_comp.legend()
        st.pyplot(fig_comp)

        st.markdown("#### 🔥 Gradient Norm Heatmap")
        norm_grad = np.sqrt((np.gradient(Z, axis=0))**2 + (np.gradient(Z, axis=1))**2)
        fig3, ax3 = plt.subplots()
        heat = ax3.imshow(norm_grad, extent=[-5, 5, -5, 5], origin='lower', cmap='plasma')
        fig3.colorbar(heat, ax=ax3, label="‖∇f‖")
        ax3.set_title("‖∇f(x, y)‖")
        st.pyplot(fig3)

    with col2:
        st.markdown("#### 🌄 Loss Surface")
        loss_type = st.radio("Loss Type", ["MSE", "Log Loss", "Cross Entropy", "Custom"], key="loss")
        X_loss, Y_loss = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
        if loss_type == "MSE":
            Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2
        elif loss_type == "Log Loss":
            Z_loss = np.log(1 + np.exp(-(X_loss + Y_loss)))
        elif loss_type == "Cross Entropy":
            p = 1 / (1 + np.exp(-(X_loss + Y_loss)))
            Z_loss = -p * np.log(p + 1e-8) - (1 - p) * np.log(1 - p + 1e-8)
        else:
            Z_loss = np.sin(X_loss) * np.cos(Y_loss)

        fig_loss = plt.figure(figsize=(4, 3))
        ax_loss = fig_loss.add_subplot(111, projection='3d')
        ax_loss.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis')
        ax_loss.set_title(f"{loss_type} Surface")
        st.pyplot(fig_loss)

        st.markdown("#### ✅ Constraint Checker")
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



# # === Optimizer Comparison Panel ===
# with st.expander("📊 Optimizer Comparison Panel"):
#     selected_opts = st.multiselect("Compare Optimizers", ["GradientDescent", "Adam", "RMSProp"], default=["GradientDescent", "Adam"])
#     fig_comp, ax_comp = plt.subplots()
#     for opt in selected_opts:
#         path_opt = optimize_path(start_x, start_y, optimizer=opt, lr=lr, steps=steps)
#         zs = [f_func(xp, yp) for xp, yp in path_opt]
#         ax_comp.plot(zs, label=opt)
#     ax_comp.set_title("Convergence Comparison")
#     ax_comp.set_xlabel("Step")
#     ax_comp.set_ylabel("f(x, y)")
#     ax_comp.legend()
#     st.pyplot(fig_comp)

# # === Loss Surface Explorer ===
# with st.expander("🌄 Loss Surface Explorer"):
#     loss_type = st.radio("Select Loss Surface", ["MSE", "Log Loss", "Custom"])
#     X_loss, Y_loss = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
#     if loss_type == "MSE":
#         Z_loss = (X_loss - 2)**2 + (Y_loss + 1)**2
#     elif loss_type == "Log Loss":
#         eps = 1e-5
#         Z_loss = -np.log(1 / (1 + np.exp(-(X_loss + Y_loss))) + eps)
#     else:
#         Z_loss = np.sin(X_loss) * np.cos(Y_loss)

#     fig = plt.figure(figsize=(4, 3))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(X_loss, Y_loss, Z_loss, cmap='viridis')
#     ax.set_title(f"{loss_type} Surface")
#     st.pyplot(fig)


# # === Constraint Satisfaction Checker ===
# with st.expander("✅ Constraint Satisfaction Checker"):
#     path = optimize_path(start_x, start_y, optimizer=optimizer, lr=lr, steps=steps)
#     if constraints:
#         fig, ax = plt.subplots()
#         for i, g_func in enumerate(g_funcs):
#             violations = [g_func(xp, yp) for xp, yp in path]
#             ax.plot(violations, label=f"g{i+1}(x, y)")
#         ax.axhline(0, color="red", linestyle="--")
#         ax.set_ylabel("g(x, y)")
#         ax.set_xlabel("Step")
#         ax.set_title("Constraint Violation Over Time")
#         ax.legend()
#         st.pyplot(fig)
#     else:
#         st.info("No constraints defined for this function.")

# # --- Gradient Norm Heatmap ---


# # === Symbolic Analysis: KKT, Gradient & Hessian ===
with st.expander("📐 Symbolic Analysis: KKT, Gradient & Hessian", expanded=False):
    st.markdown("#### 🎯 Objective & Lagrangian")
    st.latex(r"f(x, y) = " + sp.latex(f_expr))
    st.latex(r"\mathcal{L}(x, y, \lambda) = " + sp.latex(L_expr))

    st.markdown("#### ✅ KKT Conditions")
    for i, cond in enumerate(kkt_conditions):
        st.latex(fr"\text{{KKT}}_{{{i+1}}} = {sp.latex(cond)}")

    st.markdown("#### 🧮 Gradient & Hessian")
    grad = [sp.diff(f_expr, v) for v in (x, y)]
    hessian = sp.hessian(f_expr, (x, y))
    st.latex("Gradient: " + sp.latex(sp.Matrix(grad)))
    st.latex("Hessian: " + sp.latex(hessian))



from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris(as_frame=True)
df = iris.frame
df["target"] = iris.target_names[iris.target]  # for categorical labels


uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.header("Dataset Options")
    st.sidebar.write("Select your target and feature columns below:")
    target = st.sidebar.selectbox("🎯 Target Column", df.columns)
    
    if target:
        if df[target].dtype == 'object' or df[target].dtype.name == 'category':
            df[target] = df[target].astype('category').cat.codes
            original_labels = df[target].astype('category').cat.categories.tolist()
            st.sidebar.write("🔢 Target Mapping:", {i: label for i, label in enumerate(original_labels)})
        feature_candidates = [col for col in df.columns if col != target]
        features = st.sidebar.multiselect("🧩 Feature Columns", feature_candidates, key="feature_columns")

    if target and features:
        X = df[features].select_dtypes(include=[np.number])
        y = pd.to_numeric(df[target], errors="coerce")
        X = X[~y.isna()]
        y = y.dropna()
        y_class = y.round().astype(int)

        # --- Model Selection Redesign ---
        st.sidebar.markdown("## 📌 Select Analysis Type")
        analysis_type = st.sidebar.radio("Choose Analysis", ["📘 Guide", "Regression", "Classification", "Optimization Demo"])

        model_tab = "📘 How to Use"  # default

        if analysis_type == "📘 Guide":
            model_tab = "📘 How to Use"

        elif analysis_type == "Regression":
            regression_type = st.sidebar.radio("Select Model Type", ["Linear Regression", "Logistic Regression"])

            if regression_type == "Linear Regression":
                model_tab = st.sidebar.selectbox("🔍 Linear Regression Tools", [
                    "🔹 Simple Linear Regression",
                    "📐 Polynomial Regression",
                    "📊 Multi-Feature Regression",
                    "🔬 Diagnostics"
                ])
            elif regression_type == "Logistic Regression":
                model_tab = st.sidebar.selectbox("🔍 Logistic Regression Tools", [
                    "🧮 Logistic Regression",
                    "📉 Loss Landscape"
                ])

        elif analysis_type == "Classification":
            model_tab = "🔴 Classification"

        elif analysis_type == "Optimization Demo":
            model_tab = "🌋 Optimization Landscape"

    
        if model_tab == "📘 How to Use":
            st.markdown("""
            ### 👋 Welcome to the Regression Visualizer!
            This tool helps you explore **linear and polynomial regression** with ease. Here's how to use it:

            1. **Upload your CSV** file above.
            2. **Choose a target** (what you want to predict, e.g., Sales).
            3. **Choose one or more features** (e.g., TV, Radio) as predictors.
            4. Go through each tab:
               - 🔹 *Simple Linear Regression* – for 1 feature
               - 📐 *Polynomial Regression* – curve fit up to degree 5
               - 📊 *Multi-Feature Regression* – combine all features
               - 🔬 *Diagnostics* – explore RSS, RSE, confidence intervals, and check residuals

            ### 📊 How to Read the OLS Summary Table

            | Term        | Meaning                                | What to Look For              |
            |-------------|-----------------------------------------|-------------------------------|
            | coef        | Effect size                             | Large absolute values         |
            | std err     | Estimate uncertainty                    | Small = more reliable         |
            | t           | Coefficient / std err                   | Higher magnitude = stronger   |
            | P> abs(t)        | p-value of significance test            | < 0.05 = statistically significant |
            | R-squared   | % variance explained                    | Closer to 1 = better          |
            | Adj. R²     | Adjusted R² for model complexity        | Higher = better               |
            | F-statistic | Model-wide significance test            | Higher = better               |

            ### ℹ️ Glossary of Terms
            - **coef**: Estimated effect of the feature on the target.
            - **std err**: Uncertainty around the coefficient estimate.
            - **t**: t-statistic = coef / std err. Large means stronger evidence the coef ≠ 0.
            - **P>|t|**: p-value. Small means the effect is likely real (statistically significant).
            - **R-squared**: Proportion of variance in the target explained by the model.
            - **Adj. R²**: Adjusted R-squared (accounts for number of predictors).
            - **F-statistic**: Tests whether the overall regression model is statistically significant.

            
            📈 Example: Good Regression Output

            - **R-squared = 0.85** → The model explains 85% of the variation.
            - **P-values for all predictors < 0.01** → All features are statistically significant.
            - **Low standard errors and large t-statistics** → Reliable coefficient estimates.
            - **F-statistic is high with p-value < 0.05** → The overall model is significant.

            ✅ This means the regression model fits the data well and the predictors are meaningful.

            ⚠️ Example: Poor Regression Output

            - **R-squared = 0.25** → The model only explains 25% of the variance.
            - **P-values > 0.05** for most predictors → Features are likely not contributing.
            - **Very high standard errors** → Coefficients are uncertain.
            - **F-statistic is low and not significant** → Model may not be useful overall.

            🚫 Consider revising features, increasing data quality, or using nonlinear models.
            """)


        elif model_tab == "🧮 Logistic Regression":
            st.markdown("### 🧮 Logistic Regression")
            C_val = st.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            max_iter_val = st.slider("Max Iterations", 100, 1000, 300)
            solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
            penalty = st.selectbox("Penalty", ["l2", "l1"])
            y_class = y.round().astype(int)
            model = LogisticRegression(C=C_val, max_iter=max_iter_val, solver=solver, penalty=penalty)
            model.fit(X, y_class)
            y_pred = model.predict(X)

            st.metric("Accuracy", f"{accuracy_score(y_class, y_pred):.4f}")
            st.write("📉 Confusion Matrix")
            fig = plt.figure()
            sns.heatmap(confusion_matrix(y_class, y_pred), annot=True, fmt="d", cmap="Blues")
            st.pyplot(fig)

            st.write("📋 Classification Report")
            st.dataframe(pd.DataFrame(classification_report(y_class, y_pred, output_dict=True)).transpose())

            if X.shape[1] == 2:
                st.write("🌀 2D Decision Boundary")
                x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                    np.linspace(y_min, y_max, 200))
                Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                fig, ax = plt.subplots()
                ax.contourf(xx, yy, Z, alpha=0.4)
                ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, edgecolors='k')
                st.pyplot(fig)

            if len(np.unique(y_class)) == 2:
                y_proba = model.predict_proba(X)[:, 1]
                fpr, tpr, _ = roc_curve(y_class, y_proba)
                roc_auc = roc_auc_score(y_class, y_proba)
                fig = plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], "k--")
                plt.title("ROC Curve")
                st.pyplot(fig)

        elif model_tab == "🔹 Simple Linear Regression":
            st.markdown("### 🔹 Simple Linear Regression")
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
                st.markdown("📋 OLS Summary")
                st.text(model.summary())
            else:
                st.warning("Please select exactly 1 feature.")

        elif model_tab == "📐 Polynomial Regression":
            st.markdown("### 📐 Polynomial Regression")
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

        elif model_tab == "📊 Multi-Feature Regression":
            st.subheader("📊 Multi-Feature Linear Regression")

            with st.expander("📈 Ridge & Lasso Enhancements", expanded=False):
                st.markdown("""
                Advanced regularization tools for regression:

                - **Ridge Regression**: Shrink coefficients with L2 penalty  
                - **Lasso Regression**: Feature selection via L1 penalty  
                - Visualize impact of regularization on coefficients and loss  
                """)

            model_type = st.radio("Select Model", ["Linear", "Ridge", "Lasso"], horizontal=True)
            alpha_val = st.slider("Regularization Strength (α)", 0.0, 10.0, 0.0, help="Used only for Ridge and Lasso")

            # Prepare data
            X_const = sm.add_constant(X)

            # Fit model
            if model_type == "Linear":
                model = sm.OLS(y, X_const).fit()
                y_pred = model.predict(X_const)
            elif model_type == "Ridge":
                model = Ridge(alpha=alpha_val).fit(X, y)
                y_pred = model.predict(X)
            else:
                model = Lasso(alpha=alpha_val).fit(X, y)
                y_pred = model.predict(X)

            # Show summary only for OLS
            if model_type == "Linear":
                st.text(model.summary())

            # Plot predictions
            fig = plt.figure()
            plt.scatter(y, y_pred, alpha=0.6)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title(f"{model_type} Regression Results")
            st.pyplot(fig)


        elif model_tab == "🔬 Diagnostics":
            st.markdown("### 🔬 Residual Diagnostics")
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
            sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws=dict(color="red"))
            st.title("Lowess Residual Plot")
            st.pyplot(fig2)
    
        elif model_tab == "📉 Loss Landscape":
            st.header("📉 Logistic Regression Loss Landscape")

            X_vals = X
            y_vals = y.round().astype(int)

            if len(np.unique(y_vals)) != 2:
                st.warning("Loss Landscape only supports binary classification targets (0 or 1).")
            elif X_vals.shape[1] < 2:
                st.warning("Please select at least two features.")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import log_loss

                C_values = np.logspace(-2, 2, 30)
                max_iter_values = np.linspace(100, 1000, 30, dtype=int)
                C_mesh, Iter_mesh = np.meshgrid(C_values, max_iter_values)
                Z_loss = np.zeros_like(C_mesh)

                X_train, X_val, y_train, y_val = train_test_split(X_vals, y_vals, test_size=0.2, random_state=42)

                for i in range(C_mesh.shape[0]):
                    for j in range(C_mesh.shape[1]):
                        try:
                            model = LogisticRegression(C=C_mesh[i, j], max_iter=Iter_mesh[i, j], solver='lbfgs')
                            model.fit(X_train, y_train)
                            proba = model.predict_proba(X_val)
                            Z_loss[i, j] = log_loss(y_val, proba)
                        except:
                            Z_loss[i, j] = np.nan

                fig = plt.figure(figsize=(10, 6))
                ax = fig.add_subplot(111, projection='3d')
                surf = ax.plot_surface(np.log10(C_mesh), Iter_mesh, Z_loss, cmap='viridis', edgecolor='none')
                ax.set_xlabel("log10(C)")
                ax.set_ylabel("Max Iterations")
                ax.set_zlabel("Log Loss")
                ax.set_title("Logistic Regression Loss Surface")
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
                st.pyplot(fig)
            
            if st.checkbox("🧠 Show Interpretation"):
                st.markdown("### 🧠 How to Interpret This Plot")
                st.markdown("""
                This 3D plot shows how **log loss** changes with different logistic regression hyperparameters:

                - **X-axis (`log₁₀(C)`)**: The inverse of regularization strength. Lower values imply **stronger regularization**.
                - **Y-axis (`Max Iterations`)**: The maximum number of iterations for the solver to converge.
                - **Z-axis (`Log Loss`)**: The model's prediction error — **lower is better**.

                🟢 **Lower regions** (dark blue) indicate better hyperparameter combinations.
                🔴 The **lowest point** (optimal region) helps identify the best values for `C` and `max_iter`.

                This helps guide your **hyperparameter tuning** efforts effectively!
                """)



        elif model_tab == "🔴 Classification":
            st.subheader("🔴 Classification Playground")
            y_class = y.round().astype(int)
            classifier_type = st.radio("Select Classifier", ["Naive Bayes", "Decision Tree", "SVM"], horizontal=True)

            if classifier_type == "Naive Bayes":
                from sklearn.naive_bayes import GaussianNB
                model = GaussianNB()
                model.fit(X, y_class)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1] if len(np.unique(y_class)) == 2 else None

            elif classifier_type == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                max_depth = st.slider("Max Depth", 1, 20, 3)
                criterion = st.selectbox("Criterion", ["gini", "entropy"])
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
                model.fit(X, y_class)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1] if len(np.unique(y_class)) == 2 else None

            elif classifier_type == "SVM":
                from sklearn.svm import SVC
                C_val = st.slider("C", 0.01, 10.0, 1.0)
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
                gamma = st.selectbox("Gamma", ["scale", "auto"])
                model = SVC(C=C_val, kernel=kernel, gamma=gamma, probability=True)
                model.fit(X, y_class)
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1] if len(np.unique(y_class)) == 2 else None

                st.metric("Accuracy", f"{accuracy_score(y_class, y_pred):.4f}")
                st.write("📉 Confusion Matrix")
                fig = plt.figure()
                sns.heatmap(confusion_matrix(y_class, y_pred), annot=True, fmt="d", cmap="Purples")
                st.pyplot(fig)

                st.write("📋 Classification Report")
                st.dataframe(pd.DataFrame(classification_report(y_class, y_pred, output_dict=True)).transpose())

                # --- Optional decision boundary (2D) ---
                if X.shape[1] == 2:
                    st.write("🌀 2D Decision Boundary")
                    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
                    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                                            np.linspace(y_min, y_max, 200))
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    fig, ax = plt.subplots()
                    ax.contourf(xx, yy, Z, alpha=0.4)
                    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_class, edgecolors='k')
                    st.pyplot(fig)

                # --- Optional ROC Curve ---
                if len(np.unique(y_class)) == 2 and y_proba is not None:
                    fpr, tpr, _ = roc_curve(y_class, y_proba)
                    roc_auc = roc_auc_score(y_class, y_proba)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], "k--")
                    plt.title("ROC Curve")
                    st.pyplot(fig)
            

        elif model_tab == "🌋 Optimization Landscape":
            st.header("🌋 Optimization Landscape")

            function_type = st.selectbox("Select Function", ["Himmelblau"])

            x = np.linspace(-6, 6, 400)
            y_ = np.linspace(-6, 6, 400)
            Xg, Yg = np.meshgrid(x, y_)

            # Himmelblau function and gradient

            def himmelblau(x, y):
                return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

            def gradient(x, y):
                dx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
                dy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
                return np.array([dx, dy])

            x0 = st.slider("Initial x", -6.0, 6.0, -4.0)
            y0 = st.slider("Initial y", -6.0, 6.0, 4.0)
            lr = st.slider("Learning Rate", 0.001, 0.05, 0.01)
            steps = st.slider("Steps", 10, 200, 50)
            show_animation = st.checkbox("🎞️ Animate Gradient Descent")

            x = np.linspace(-6, 6, 400)
            y = np.linspace(-6, 6, 400)
            X, Y = np.meshgrid(x, y)
            Z = himmelblau(X, Y)

            path = []
            xi, yi = x0, y0
            for _ in range(steps):
                path.append((xi, yi))
                dx, dy = gradient(xi, yi)
                xi -= lr * dx
                yi -= lr * dy
            path = np.array(path)
            Z_path = himmelblau(path[:, 0], path[:, 1])

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, edgecolor='none')
            ax.contour(X, Y, Z, zdir='z', offset=0, cmap='coolwarm', linestyles="solid")

            # Animate descent if selected
            if show_animation:
                for i in range(1, len(path)):
                    ax.plot(path[:i, 0], path[:i, 1], Z_path[:i], color='red', marker='o')
                    ax.set_zlim(0, np.max(Z))
                    ax.set_title(f"Step {i}/{len(path)}")
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("f(x, y)")
                    st.pyplot(fig)
                    time.sleep(0.1)
                    fig.clf()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, edgecolor='none')
                    ax.contour(X, Y, Z, zdir='z', offset=0, cmap='coolwarm', linestyles="solid")
            else:
                ax.plot(path[:, 0], path[:, 1], Z_path, color='red', marker='o', label='Gradient Descent')
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("f(x, y)")
                ax.set_title("Himmelblau Optimization Landscape")
                ax.legend()
                st.pyplot(fig)

else:
    st.info("👈 Upload a CSV file to begin.")






#################

   with st.expander("🚀 Optimizer Visual Playground", expanded=True):
        col_sidebar, col_main = st.columns([1, 3])



        with col_sidebar:
            # st.markdown("## ⚙️ Configuration")
            st.markdown("<h4>⚙️ Configuration</h4>", unsafe_allow_html=True)

            mode = st.radio("Function Source", ["Predefined", "Custom"])
            func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
            expr_str = st.text_input("Enter function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""
            w_val = st.slider("Weight w (Multi-Objective)", 0.0, 1.0, 0.5) if func_name == "Multi-Objective" else None

            all_optimizers = {
            "Gradient-based 🧮": ["GradientDescent", "Adam", "Newton's Method", "RMSProp"],
            "Heuristic 🔮": ["Simulated Annealing", "Genetic Algorithm"]
        }

            flat_optimizers = sum(all_optimizers.values(), [])
            optimizer = st.selectbox("Optimizer", flat_optimizers, index=flat_optimizers.index("GradientDescent"))


            st.markdown("### 🧪 Optimizer Settings")

            options = {}

            if optimizer == "Simulated Annealing":
                options["T"] = st.slider("Initial Temperature (T)", 0.1, 10.0, 2.0)
                options["cooling"] = st.slider("Cooling Rate", 0.80, 0.99, 0.95)

            elif optimizer == "Genetic Algorithm":
                options["pop_size"] = st.slider("Population Size", 10, 100, 20)
                options["mutation_std"] = st.slider("Mutation Std Dev", 0.1, 1.0, 0.3)

            # auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True)

            # === Full default_config: each function × optimizer
            default_config = {
                (name, opt): {
                    "lr": {
                        "GradientDescent": {
                            "Quadratic Bowl": 0.01, "Saddle": 0.01, "Rosenbrock": 0.0005, "Constrained Circle": 0.01,
                            "Double Constraint": 0.005, "Multi-Objective": 0.005, "Ackley": 0.005, "Rastrigin": 0.005,
                            "Styblinski-Tang": 0.005, "Sphere": 0.01, "Himmelblau": 0.005, "Booth": 0.01, "Beale": 0.005
                        }.get(name, 0.001),
                        "Adam": 0.005,
                        "RMSProp": 0.005,
                        "Newton's Method": 0.01,
                        "Simulated Annealing": 1.0,   # Use as T (temperature)
                        "Genetic Algorithm": 0.1      # Treated as mutation std (not actual LR)
                    }[opt],
                    "steps": {
                        "GradientDescent": {
                            "Quadratic Bowl": 50, "Saddle": 50, "Rosenbrock": 80, "Constrained Circle": 60,
                            "Double Constraint": 60, "Multi-Objective": 60, "Ackley": 70, "Rastrigin": 70,
                            "Styblinski-Tang": 60, "Sphere": 50, "Himmelblau": 70, "Booth": 50, "Beale": 70
                        }.get(name, 50),
                        "Adam": 40,
                        "RMSProp": 40,
                        "Newton's Method": 30,
                        "Simulated Annealing": 80,
                        "Genetic Algorithm": 50
                    }[opt]
                }
                for name in [
                    "Quadratic Bowl", "Saddle", "Rosenbrock", "Constrained Circle", "Double Constraint",
                    "Multi-Objective", "Ackley", "Rastrigin", "Styblinski-Tang", "Sphere",
                    "Himmelblau", "Booth", "Beale"
                ]
                for opt in [
                    "GradientDescent", "Adam", "RMSProp", "Newton's Method",
                    "Simulated Annealing", "Genetic Algorithm"
                ]
            }


            # === Default start positions
            start_xy_defaults = {
                "Quadratic Bowl": (-3.0, 3.0),
                "Saddle": (-2.0, 2.0),
                "Rosenbrock": (-1.5, 1.5),
                "Constrained Circle": (0.5, 0.5),
                "Double Constraint": (-1.5, 1.5),
                "Multi-Objective": (0.0, 0.0),
                "Ackley": (2.0, -2.0),
                "Rastrigin": (3.0, 3.0),
                "Styblinski-Tang": (-2.5, -2.5),
                "Sphere": (-3.0, 3.0),
                "Himmelblau": (0.0, 0.0),
                "Booth": (1.0, 1.0),
                "Beale": (-2.0, 2.0)
            }


            def run_auto_tuning_simulation(
                f_func, optimizer, x0, y0, 
                lr_grid=list(np.logspace(-4, -1, 6)), 
                step_grid=[20, 30, 40, 50, 60, 80], 
                convergence_tol=1e-3, 
                penalty_weight=1e-2
            ):
                best_score = float("inf")
                best_lr = lr_grid[0]
                best_steps = step_grid[0]
                logs = []

                for lr in lr_grid:
                    for steps in step_grid:
                        x_t, y_t = x0, y0
                        m, v = np.zeros(2), np.zeros(2)
                        beta1, beta2, eps = 0.9, 0.999, 1e-8
                        converged = False

                        for t in range(1, steps + 1):
                            dx = (f_func(x_t + 1e-5, y_t) - f_func(x_t - 1e-5, y_t)) / 2e-5
                            dy = (f_func(x_t, y_t + 1e-5) - f_func(x_t, y_t - 1e-5)) / 2e-5
                            grad = np.array([dx, dy])
                            grad_norm = np.linalg.norm(grad)

                            if grad_norm < convergence_tol:
                                converged = True
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

                        final_loss = f_func(x_t, y_t)
                        effective_steps = t if converged else steps
                        score = final_loss + penalty_weight * effective_steps

                        logs.append({
                            "lr": lr,
                            "steps": steps,
                            "effective_steps": effective_steps,
                            "final_loss": final_loss,
                            "score": score,
                            "converged": converged
                        })

                        if score < best_score:
                            best_score = score
                            best_lr = lr
                            best_steps = effective_steps

                df_log = pd.DataFrame(logs)
                st.session_state.df_log = df_log
                return best_lr, best_steps


            col_left, col_right = st.columns([1, 1])
            with col_left:
                if func_name is not None:
                    key = (func_name, optimizer)
                    default_x, default_y = start_xy_defaults.get(func_name, (-3.0, 3.0))
                    default_lr = default_config.get(key, {}).get("lr", 0.001)
                    default_steps = default_config.get(key, {}).get("steps", 50)

                    gradient_optimizers = ["GradientDescent", "Adam", "RMSProp"]
                    if optimizer in ["Simulated Annealing", "Genetic Algorithm"]:
                        auto_tune = False
                        st.info("ℹ️ Auto-tuning not supported for heuristic optimizers.")
                    else:
                        auto_tune = st.checkbox("⚙️ Auto-Tune Learning Rate & Steps", value=True, key="auto_tune_checkbox")

                    if mode == "Predefined" and auto_tune and optimizer in gradient_optimizers:
                        symbolic_func, _, _ = predefined_funcs[func_name]

                        # === FIX: substitute w if needed
                        if func_name == "Multi-Objective":
                            if 'w_val' not in locals():
                                w_val = 0.5
                            symbolic_func = symbolic_func.subs(w, w_val)

                        f_lambdified = sp.lambdify((x, y), symbolic_func, "numpy")
                        best_lr, best_steps = run_auto_tuning_simulation(f_lambdified, optimizer, default_x, default_y)
                        default_lr = best_lr
                        default_steps = best_steps

                        if 'tune_msg_shown' not in st.session_state:
                            st.success(f"✅ Auto-tuned: lr = {default_lr}, steps = {default_steps}, start=({default_x}, {default_y})")
                            st.session_state.tune_msg_shown = True

                    if 'params_set' not in st.session_state or st.button("🔄 Reset to Auto-Tuned"):
                        st.session_state.lr = default_lr
                        st.session_state.steps = default_steps
                        st.session_state.start_x = default_x
                        st.session_state.start_y = default_y
                        st.session_state.params_set = True

                    if 'prev_key' not in st.session_state:
                        st.session_state.prev_key = None

                    current_key = (func_name, optimizer)
                    if auto_tune and current_key != st.session_state.prev_key:
                        st.session_state.lr = default_lr
                        st.session_state.steps = default_steps
                        st.session_state.start_x = default_x
                        st.session_state.start_y = default_y
                        st.session_state.params_set = True
                        st.session_state.tune_msg_shown = False
                        st.session_state.prev_key = current_key

                    lr_options = list(OrderedDict.fromkeys([0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, default_lr]))
                    if st.session_state.lr in lr_options:
                        lr_index = lr_options.index(st.session_state.lr)
                    else:
                        lr_index = 0
                    lr = st.selectbox("Learning Rate", lr_options, index=lr_index, key="lr")

                    steps = st.slider("Steps", 0, 100, st.session_state.steps, key="steps")
                    start_x = st.slider("Initial x", -5.0, 5.0, st.session_state.start_x, key="start_x")
                    start_y = st.slider("Initial y", -5.0, 5.0, st.session_state.start_y, key="start_y")
                    show_animation = st.checkbox("🎮 Animate Descent Steps")

            with col_right:
                st.markdown("### 📊 Auto-Tuning Trial Log")
                if "df_log" in st.session_state:
                    st.dataframe(st.session_state.df_log.sort_values("score").reset_index(drop=True))
                    st.markdown("""
                    **🧠 How to Read Score:**

                    - `score = final_loss + penalty × steps`
                    - ✅ Lower score is better (fast and accurate convergence).
                    - Use this to compare learning rate and step configs.
                    """)
                else:
                    st.info("Run auto-tuning to see results here.")        

        if mode == "Predefined":
            f_expr, constraints, description = predefined_funcs[func_name]
            f_expr = f_expr.subs(w, w_val) if func_name == "Multi-Objective" else f_expr
        else:
            try:
                f_expr = sp.sympify(expr_str)
                constraints = []
                description = "Custom function."
            except:
                st.error("Invalid expression.")
                st.stop()
