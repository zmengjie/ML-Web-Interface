
# === Full Modular Streamlit App ===

import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64
import tempfile

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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter

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

from supervised_learning import supervised_ui

# === Top-level Mode Switch ===
mode = st.sidebar.radio("Choose Mode", ["📈 Supervised Learning", "🌋 Optimization Playground", "🧠 Unsupervised Learning", "🤖 LLM Assistant"])

if mode == "📈 Supervised Learning":
    supervised_ui()

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
    from sidebar_config import get_sidebar_config, get_predefined_funcs




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

    from taylor_expansion import show_univariate_taylor
    
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
        show_univariate_taylor()







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
