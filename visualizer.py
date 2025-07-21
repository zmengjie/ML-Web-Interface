# import plotly.graph_objects as go
# import streamlit as st
# import numpy as np

# def plot_3d_descent(x_vals, y_vals, Z, path, Z_path, Z_t1=None, Z_t2=None, show_taylor=False, show_2nd=False):
#     xs, ys = zip(*path)

#     fig_3d = go.Figure()

#     fig_3d.add_trace(go.Surface(z=Z, x=x_vals, y=y_vals, colorscale='Viridis', opacity=0.7, name="Function Surface"))

#     fig_3d.add_trace(go.Scatter3d(
#         x=xs, y=ys, z=Z_path,
#         mode='lines+markers',
#         line=dict(color='red', width=4),
#         marker=dict(size=4),
#         name='Descent Path'
#     ))

#     if show_taylor and Z_t1 is not None:
#         fig_3d.add_trace(go.Surface(z=Z_t1, x=x_vals, y=y_vals, colorscale='Reds', opacity=0.5, name="1st-Order Taylor"))
#     if show_taylor and show_2nd and Z_t2 is not None:
#         fig_3d.add_trace(go.Surface(z=Z_t2, x=x_vals, y=y_vals, colorscale='Blues', opacity=0.4, name="2nd-Order Taylor"))

#     fig_3d.update_layout(
#         title="3D Descent Path",
#         scene=dict(
#             xaxis_title="x",
#             yaxis_title="y",
#             zaxis_title="f(x, y)"
#         ),
#         height=600,
#         margin=dict(l=0, r=0, b=0, t=30)
#     )

#     st.plotly_chart(fig_3d, use_container_width=True)



# def plot_2d_contour(x_vals, y_vals, Z, path, g_funcs=None, X=None, Y=None):
#     xs, ys = zip(*path)

#     fig_2d = go.Figure()

#     fig_2d.add_trace(go.Contour(
#         z=Z, x=x_vals, y=y_vals,
#         colorscale='Viridis',
#         contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z) - np.min(Z)) / 20),
#         name="Function Contour"
#     ))

#     fig_2d.add_trace(go.Scatter(
#         x=xs, y=ys,
#         mode='lines+markers',
#         line=dict(color='red', width=3),
#         marker=dict(size=4),
#         name="Descent Path"
#     ))

#     if g_funcs and X is not None and Y is not None:
#         for g_f in g_funcs:
#             G = g_f(X, Y)
#             fig_2d.add_trace(go.Contour(
#                 z=G, x=x_vals, y=y_vals,
#                 contours=dict(showlines=False, coloring='lines'),
#                 line=dict(color='red', width=2),
#                 showscale=False,
#                 name="Constraint"
#             ))

#     fig_2d.update_layout(
#         title="2D Contour + Constraints",
#         xaxis_title="x",
#         yaxis_title="y",
#         height=500,
#         margin=dict(l=0, r=0, b=0, t=30)
#     )

#     st.plotly_chart(fig_2d, use_container_width=True)

import plotly.graph_objects as go
import streamlit as st
import numpy as np


# def plot_3d_descent(x_vals, y_vals, Z, path, Z_path,
#                     Z_t1=None, Z_t2=None,
#                     show_taylor=False, show_2nd=False,
#                     expansion_point=None, f_func=None):
#     xs, ys = zip(*path)

#     fig_3d = go.Figure()

#     # Base surface
#     fig_3d.add_trace(go.Surface(
#         z=Z, x=x_vals, y=y_vals,
#         colorscale='Viridis', opacity=0.7,
#         name="Function Surface"
#     ))

#     # Descent path
#     fig_3d.add_trace(go.Scatter3d(
#         x=xs, y=ys, z=Z_path,
#         mode='lines+markers',
#         line=dict(color='red', width=4),
#         marker=dict(size=4),
#         name='Descent Path',
#         hoverinfo='x+y+z+text',
#         text=['Step {}'.format(i) for i in range(len(path))]
#     ))

#     # Taylor surfaces with fading
#     if show_taylor and Z_t1 is not None:
#         fig_3d.add_trace(go.Surface(
#             z=Z_t1, x=x_vals, y=y_vals,
#             colorscale='Reds', opacity=0.6,
#             line=dict(width=2, color='black'),
#             name="1st-Order Taylor"
#         ))
#     # if show_taylor and show_2nd and Z_t2 is not None and expansion_point is not None:
#     #     a, b = expansion_point
#     #     distance = np.sqrt((x_vals[:, None] - a)**2 + (y_vals[None, :] - b)**2)
#     #     fade_opacity = 0.6 * np.exp(-0.1 * distance)
#     #     fig_3d.add_trace(go.Surface(
#     #         z=Z_t2, x=x_vals, y=y_vals,
#     #         surfacecolor=fade_opacity,
#     #         colorscale='Blues',
#     #         opacity=0.4,
#     #         name="2nd-Order Taylor"
#     #     ))

# # --- Taylor surfaces with fading ---

#     if show_taylor and show_2nd:
#         if Z_t2 is not None:
#             try:
#                 if isinstance(Z_t2, (int, float)):
#                     raise ValueError(f"Z_t2 is a scalar ({Z_t2}); expected 2D array.")
#                 Z_t2 = np.array(Z_t2)
#                 if Z_t2.ndim != 2:
#                     raise ValueError(f"Z_t2 is not 2D (shape: {Z_t2.shape})")
#                 if np.isnan(Z_t2).any():
#                     raise ValueError("Z_t2 contains NaNs.")
#                 if Z_t2.shape != (len(y_vals), len(x_vals)):
#                     if Z_t2.shape == (len(x_vals), len(y_vals)):
#                         Z_t2 = Z_t2.T
#                     else:
#                         raise ValueError(f"Z_t2 shape mismatch: {Z_t2.shape} vs expected {(len(y_vals), len(x_vals))}")
#                 fig_3d.add_trace(go.Surface(
#                     z=Z_t2, x=x_vals, y=y_vals,
#                     colorscale='RdBu',
#                     opacity=0.4,
#                     cmin=np.min(Z_t2),
#                     cmax=np.max(Z_t2),
#                     line=dict(width=2, color='black'),
#                     name="2nd-Order Taylor"
#                 ))
#             except Exception as e:
#                 st.warning(f"⚠️ Skipped 2nd-order Taylor surface: {e}")




#     # Marker and dashed line from (a,b) to step 1
#     if expansion_point is not None and f_func is not None:
#         a, b = expansion_point
#         z_ab = f_func(a, b)

#         fig_3d.add_trace(go.Scatter3d(
#             x=[a], y=[b], z=[z_ab],
#             mode='markers+text',
#             marker=dict(size=7, color='black', symbol='circle'),
#             text=["(a, b)"],
#             textposition="bottom right",
#             textfont=dict(size=12),
#             name="Expansion Point".
#             hoverinfo='text+x+y+z'
#         ))

#         if len(path) > 1:
#             x1, y1 = path[1]
#             z1 = f_func(x1, y1)
#             fig_3d.add_trace(go.Scatter3d(
#                 x=[a, x1], y=[b, y1], z=[z_ab, z1],
#                 mode='lines',
#                 line=dict(color='black', width=3, dash='dash'),
#                 name="Expansion → 1st Step"
#             ))

#     fig_3d.update_layout(
#         title="3D Descent Path + Taylor Approximation",
#         scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)"),
#         height=600,
#         margin=dict(l=60, r=40, b=40, t=50),
#         legend=dict(x=0.7, y=0.9),
#         hovermode='closest'
#     )

#     st.markdown("""
#     ### 🧠 Teaching Tip
#     The dashed vector from (a, b) shows how the Taylor approximation predicts the direction of descent.
#     The 2nd-order surface illustrates curvature guidance for Newton's Method.
#     """)
#     st.plotly_chart(fig_3d, use_container_width=True)

# def plot_2d_contour(x_vals, y_vals, Z, path,
#                     g_funcs=None, X=None, Y=None,
#                     Z_t2=None, show_2nd=False,
#                     expansion_point=None):
#     xs, ys = zip(*path)

#     fig_2d = go.Figure()

#     # Taylor contour first to draw under others
#     if show_2nd and Z_t2 is not None:
#         fig_2d.add_trace(go.Contour(
#             z=Z_t2, x=x_vals, y=y_vals,
#             showscale=False,
#             colorscale='Blues',
#             opacity=0.4,
#             name="2nd-Order Taylor"
#         ))

#     # Base contour
#     fig_2d.add_trace(go.Contour(
#         z=Z, x=x_vals, y=y_vals,
#         colorscale='Viridis',
#         contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z) - np.min(Z)) / 20),
#         name="Function Contour",
#         showscale=True
#     ))

#     # Descent path
#     fig_2d.add_trace(go.Scatter(
#         x=xs, y=ys,
#         mode='lines+markers',
#         line=dict(color='red', width=3),
#         marker=dict(size=4),
#         name="Descent Path"
#     ))

#     # Constraints
#     if g_funcs and X is not None and Y is not None:
#         for g_f in g_funcs:
#             G = g_f(X, Y)
#             fig_2d.add_trace(go.Contour(
#                 z=G, x=x_vals, y=y_vals,
#                 contours=dict(showlines=False, coloring='lines'),
#                 line=dict(color='red', width=2),
#                 showscale=False,
#                 name="Constraint"
#             ))

#     if expansion_point is not None:
#         a, b = expansion_point
#         fig_2d.add_trace(go.Scatter(
#             x=[a], y=[b],
#             mode='markers+text',
#             marker=dict(color='black', size=9, symbol="circle"),
#             text=["(a, b)"],
#             textposition="bottom center",
#             textfont=dict(size=12),
#             name='Expansion Point'
#         ))

#         if len(path) > 1:
#             x1, y1 = path[1]
#             fig_2d.add_trace(go.Scatter(
#                 x=[a, x1], y=[b, y1],
#                 mode='lines',
#                 line=dict(color='black', width=2, dash='dash'),
#                 name="Expansion → 1st Step"
#             ))

#     fig_2d.update_layout(
#         title="2D Contour + Constraints + Taylor Overlay",
#         xaxis_title="x", yaxis_title="y",
#         height=500,
#         margin=dict(l=60, r=40, b=40, t=50),
#         legend=dict(x=1.05),
#         dragmode='zoom',
#         hovermode='closest',
#         updatemenus=[dict(
#             type="buttons", showactive=False,
#             buttons=[dict(label="Reset Zoom", method="relayout",
#                          args=[{"xaxis.autorange": True, "yaxis.autorange": True}])],
#             x=0, y=0.92, xanchor='left', yanchor='top')]
#     )

#     st.markdown("""
#     ### 🧠 Teaching Tip
#     The dashed vector from (a, b) shows how the Taylor approximation predicts the direction of descent.
#     The 2nd-order contour shows how curvature affects local optimization.
#     """)
#     st.plotly_chart(fig_2d, use_container_width=True)

import plotly.graph_objects as go
import streamlit as st
import numpy as np

def compute_gradient_field(f_func, x_vals, y_vals, step=5):
    X, Y = np.meshgrid(x_vals, y_vals)
    X_sample = X[::step, ::step]
    Y_sample = Y[::step, ::step]
    h = 1e-4

    dfx = (f_func(X_sample + h, Y_sample) - f_func(X_sample - h, Y_sample)) / (2 * h)
    dfy = (f_func(X_sample, Y_sample + h) - f_func(X_sample, Y_sample - h)) / (2 * h)
    Z_sample = f_func(X_sample, Y_sample)

    return X_sample, Y_sample, Z_sample, dfx, dfy

def plot_3d_descent(x_vals, y_vals, Z, path, Z_path,
                    Z_t1=None, Z_t2=None,
                    show_taylor=False, show_2nd=False,
                    expansion_point=None,
                    f_func=None,
                    grad_func=None,
                    hess_func=None):
    xs, ys = zip(*path)
    fig_3d = go.Figure()

    # Surface
    fig_3d.add_trace(go.Surface(
        z=Z, x=x_vals, y=y_vals,
        colorscale='Viridis', opacity=0.7,
        name="Function Surface"
    ))

    # Descent Path
    fig_3d.add_trace(go.Scatter3d(
        x=xs, y=ys, z=Z_path,
        mode='lines+markers+text',
        line=dict(color='red', width=4),
        marker=dict(size=4),
        text=[f"Step {i}" for i in range(len(path))],
        textposition="top center",
        name="Descent Path",
        hoverinfo='text+x+y+z'
    ))

    # Final Point Marker
    fig_3d.add_trace(go.Scatter3d(
        x=[xs[-1]], y=[ys[-1]], z=[Z_path[-1]],
        mode='markers+text',
        marker=dict(size=8, color='green', symbol='diamond'),
        text=["Final Point"],
        textposition="bottom center",
        name="Local Min"
    ))

    # 1st-order Taylor Surface
    if show_taylor and Z_t1 is not None:
        fig_3d.add_trace(go.Surface(
            z=Z_t1, x=x_vals, y=y_vals,
            colorscale='Reds', opacity=0.6,
            name="1st-Order Taylor",
            line=dict(width=1)
        ))

    # 2nd-order Taylor Surface
    if show_taylor and show_2nd and Z_t2 is not None:
        try:
            Z_t2 = np.array(Z_t2)
            if Z_t2.shape != (len(y_vals), len(x_vals)):
                if Z_t2.shape == (len(x_vals), len(y_vals)):
                    Z_t2 = Z_t2.T
                else:
                    raise ValueError(f"Z_t2 shape mismatch: {Z_t2.shape}")
            fig_3d.add_trace(go.Surface(
                z=Z_t2, x=x_vals, y=y_vals,
                colorscale='RdBu', opacity=0.4,
                name="2nd-Order Taylor",
                cmin=np.min(Z_t2), cmax=np.max(Z_t2)
            ))
        except Exception as e:
            st.warning(f"⚠️ Skipped 2nd-order Taylor surface: {e}")

    # Expansion Point & Arrows
    if expansion_point is not None and f_func is not None:
        a, b = expansion_point
        z_ab = f_func(a, b)

        # Marker
        fig_3d.add_trace(go.Scatter3d(
            x=[a], y=[b], z=[z_ab],
            mode='markers+text',
            marker=dict(size=7, color='black', symbol='circle'),
            text=["Expansion (a, b)"],
            textposition="bottom right",
            name="Expansion Point"
        ))

        # 1st Step Arrow
        if len(path) > 1:
            x1, y1 = path[1]
            z1 = f_func(x1, y1)
            fig_3d.add_trace(go.Scatter3d(
                x=[a, x1], y=[b, y1], z=[z_ab, z1],
                mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                name="1st Step Vector"
            ))

        # Add 2nd-order Newton step (Hessian-guided)
        if grad_func is not None and hess_func is not None:
            grad = np.array(grad_func(a, b)).reshape(-1)
            hess = np.array(hess_func(a, b))
            try:
                delta = -np.linalg.solve(hess, grad)
                dx, dy = delta[0], delta[1]
                x_new, y_new = a + dx, b + dy
                z_new = f_func(x_new, y_new)
                fig_3d.add_trace(go.Scatter3d(
                    x=[a, x_new], y=[b, y_new], z=[z_ab, z_new],
                    mode='lines',
                    line=dict(color='blue', width=4, dash='dot'),
                    name="2nd-Order Newton Step"
                ))
            except np.linalg.LinAlgError:
                st.warning("⚠️ Hessian not invertible — Newton step skipped.")

    # ➕ Gradient Field
    if f_func is not None:
        try:
            Xg, Yg, Zg, dfx, dfy = compute_gradient_field(f_func, x_vals, y_vals, step=5)
            fig_3d.add_trace(go.Cone(
                x=Xg.flatten(), y=Yg.flatten(), z=Zg.flatten(),
                u=dfx.flatten(), v=dfy.flatten(), w=np.zeros_like(dfx).flatten(),
                sizemode="scaled", sizeref=0.3 * np.max(np.abs(Z)),
                anchor="tail", showscale=False,
                colorscale="Greys", opacity=0.5,
                name="∇f (Gradient Arrows)"
            ))
        except Exception as e:
            st.warning(f"⚠️ Gradient field skipped: {e}")

    # Final Layout
    fig_3d.update_layout(
        title="3D Descent Path + Taylor + Gradient + Newton",
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)",
            camera=dict(eye=dict(x=1.3, y=1.2, z=0.8))
        ),
        height=700,
        margin=dict(l=60, r=40, b=40, t=50),
        legend=dict(x=0.7, y=0.85)
    )


    # Teaching block
    st.markdown("""
    ### 🧠 Optimization Insights
    - 🟢 Final green marker = convergence point.
    - ⚫ Dotted line from expansion = 1st-order Taylor prediction.
    - 🔵 Dashed blue arrow = 2nd-order Newton update (if Hessian is valid).
    - ➕ Gray cones show ∇f direction at sampled points (pointing uphill).
    """)
    st.info("🎥 Drag to rotate, scroll to zoom, right-click to pan. Arrows show gradient flow.")
    st.plotly_chart(fig_3d, use_container_width=True)


def plot_2d_contour(x_vals, y_vals, Z, path,
                    g_funcs=None, X=None, Y=None,
                    Z_t2=None, show_2nd=False,
                    expansion_point=None,
                    f_func=None,
                    grad_func=None,
                    hess_func=None):
    xs, ys = zip(*path)
    fig_2d = go.Figure()

    # --- 2nd-order Taylor surface as contour ---
    if show_2nd and Z_t2 is not None:
        fig_2d.add_trace(go.Contour(
            z=Z_t2, x=x_vals, y=y_vals,
            showscale=False,
            colorscale='Blues',
            opacity=0.4,
            name="2nd-Order Taylor"
        ))

    # --- Function contour ---
    fig_2d.add_trace(go.Contour(
        z=Z, x=x_vals, y=y_vals,
        colorscale='Viridis',
        contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z) - np.min(Z)) / 20),
        name="Function Contour",
        showscale=True
    ))

    # --- Descent path ---
    fig_2d.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=4),
        name="Descent Path"
    ))

    # --- Constraints ---
    if g_funcs and X is not None and Y is not None:
        for g_f in g_funcs:
            G = g_f(X, Y)
            fig_2d.add_trace(go.Contour(
                z=G, x=x_vals, y=y_vals,
                contours=dict(showlines=False, coloring='lines'),
                line=dict(color='red', width=2),
                showscale=False,
                name="Constraint"
            ))

    # --- Expansion point marker ---
    if expansion_point is not None:
        a, b = expansion_point
        fig_2d.add_trace(go.Scatter(
            x=[a], y=[b],
            mode='markers+text',
            marker=dict(color='black', size=9, symbol="circle"),
            text=["(a, b)"],
            textposition="bottom center",
            textfont=dict(size=12),
            name='Expansion Point'
        ))

        if len(path) > 1:
            x1, y1 = path[1]
            fig_2d.add_trace(go.Scatter(
                x=[a, x1], y=[b, y1],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name="1st Step Vector"
            ))

        # --- Newton (2nd-order) update arrow ---
        if grad_func is not None and hess_func is not None:
            try:
                grad = np.array(grad_func(a, b)).reshape(-1)
                hess = np.array(hess_func(a, b))
                delta = -np.linalg.solve(hess, grad)
                dx, dy = delta[0], delta[1]
                fig_2d.add_trace(go.Scatter(
                    x=[a, a + dx], y=[b, b + dy],
                    mode='lines',
                    line=dict(color='blue', width=3, dash='dot'),
                    name="2nd-Order Newton Step"
                ))
            except np.linalg.LinAlgError:
                st.warning("⚠️ Hessian not invertible — Newton arrow skipped.")

    # --- Final layout ---
    fig_2d.update_layout(
        title="2D Contour + Constraints + Taylor + Newton Step",
        xaxis_title="x", yaxis_title="y",
        height=500,
        margin=dict(l=60, r=40, b=40, t=50),
        legend=dict(x=1.05),
        dragmode='zoom',
        hovermode='closest',
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[dict(label="Reset Zoom", method="relayout",
                         args=[{"xaxis.autorange": True, "yaxis.autorange": True}])],
            x=0, y=0.92, xanchor='left', yanchor='top')]
    )

    st.markdown("""
    ### 🧠 Newton's Method Insight
    - ⚫ Dotted arrow from (a, b) = 1st-order prediction direction (gradient).
    - 🔵 Dashed blue arrow = 2nd-order Newton direction, guided by Hessian.
    - The Newton step typically points more directly toward the minimum when curvature is well-behaved.
    """)
    st.plotly_chart(fig_2d, use_container_width=True)
