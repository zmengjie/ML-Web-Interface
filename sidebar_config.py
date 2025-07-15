# sidebar_config.py
import streamlit as st
import sympy as sp

def get_predefined_funcs():
    x, y, w = sp.symbols("x y w")
    return {
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

def get_sidebar_config(predefined_funcs, optimizers):
    with st.sidebar:
        st.header("⚙️ Optimizer Settings")

        mode = st.radio("Function Source", ["Predefined", "Custom"])
        func_name = st.selectbox("Function", list(predefined_funcs.keys())) if mode == "Predefined" else None
        expr_str = st.text_input("Custom function f(x,y):", "x**2 + y**2") if mode == "Custom" else ""

        w_val = None
        if func_name == "Multi-Objective":
            w_val = st.slider("Weight w", 0.0, 1.0, 0.5)

        optimizer = st.selectbox("Optimizer", optimizers)

        options = {}
        auto_tune = False

        if optimizer in ["GradientDescent", "Adam", "RMSProp"]:
            if optimizer == "GradientDescent":
                use_backtracking = st.checkbox("🔍 Use Backtracking Line Search")
                options["use_backtracking"] = use_backtracking
                if not use_backtracking:
                    auto_tune = st.checkbox("⚙️ Auto-Tune", value=True)
            else:
                auto_tune = st.checkbox("⚙️ Auto-Tune", value=True)
        elif optimizer == "Newton's Method":
            newton_variant = st.selectbox("Newton Variant", ["Classic Newton", "Numerical Newton", "BFGS", "L-BFGS"])
            options["newton_variant"] = newton_variant
        elif optimizer == "Simulated Annealing":
            options["T"] = st.slider("Temperature", 0.1, 10.0, 2.0)
            options["cooling"] = st.slider("Cooling Rate", 0.80, 0.99, 0.95)
        elif optimizer == "Genetic Algorithm":
            options["pop_size"] = st.slider("Population Size", 10, 100, 20)
            options["mutation_std"] = st.slider("Mutation Std Dev", 0.1, 1.0, 0.3)

        start_x = st.slider("Initial x", -5.0, 5.0, -3.0)
        start_y = st.slider("Initial y", -5.0, 5.0, 3.0)

        learning_rate = None
        steps = None

        if optimizer not in ["Newton's Method"] and not options.get("use_backtracking", False):
            learning_rate = st.selectbox("Learning Rate", [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1], index=2)
            steps = st.slider("Steps", 10, 100, 50)

        return {
            "mode": mode,
            "func_name": func_name,
            "expr_str": expr_str,
            "w_val": w_val,
            "optimizer": optimizer,
            "options": options,
            "auto_tune": auto_tune,
            "start_x": start_x,
            "start_y": start_y,
            "learning_rate": learning_rate,
            "steps": steps
        }
