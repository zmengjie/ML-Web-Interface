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

def plot_3d_descent(x_vals, y_vals, Z, path, Z_path,
                    Z_t1=None, Z_t2=None,
                    show_taylor=False, show_2nd=False,
                    expansion_point=None, f_func=None):
    xs, ys = zip(*path)

    fig_3d = go.Figure()

    # Base surface
    fig_3d.add_trace(go.Surface(
        z=Z, x=x_vals, y=y_vals,
        colorscale='Viridis', opacity=0.7,
        name="Function Surface"
    ))

    # Descent path
    fig_3d.add_trace(go.Scatter3d(
        x=xs, y=ys, z=Z_path,
        mode='lines+markers',
        line=dict(color='red', width=4),
        marker=dict(size=4),
        name='Descent Path'
    ))

    # Taylor surfaces
    if show_taylor and Z_t1 is not None:
        fig_3d.add_trace(go.Surface(
            z=Z_t1, x=x_vals, y=y_vals,
            colorscale='Reds', opacity=0.5,
            name="1st-Order Taylor"
        ))
    if show_taylor and show_2nd and Z_t2 is not None:
        fig_3d.add_trace(go.Surface(
            z=Z_t2, x=x_vals, y=y_vals,
            colorscale='Blues', opacity=0.4,
            name="2nd-Order Taylor"
        ))

    # Marker and dashed line from (a,b) to step 1
    if expansion_point is not None and f_func is not None:
        a, b = expansion_point
        z_ab = f_func(a, b)

        fig_3d.add_trace(go.Scatter3d(
            x=[a], y=[b], z=[z_ab],
            mode='markers+text',
            marker=dict(size=7, color='black', symbol='circle'),
            text=["(a, b)"],
            textposition="top center",
            textfont=dict(size=11),
            name="Expansion Point"
        ))

        # Dashed line to step 1
        if len(path) > 1:
            x1, y1 = path[1]
            z1 = f_func(x1, y1)
            fig_3d.add_trace(go.Scatter3d(
                x=[a, x1], y=[b, y1], z=[z_ab, z1],
                mode='lines',
                line=dict(color='black', width=3, dash='dash'),
                name="Expansion → 1st Step"
            ))

    fig_3d.update_layout(
        title="3D Descent Path + Taylor Approximation",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x, y)"
        ),
        height=600,
        margin=dict(l=40, r=20, b=20, t=50)
    )

    st.plotly_chart(fig_3d, use_container_width=True)


def plot_2d_contour(x_vals, y_vals, Z, path,
                    g_funcs=None, X=None, Y=None,
                    Z_t2=None, show_2nd=False,
                    expansion_point=None):
    xs, ys = zip(*path)

    fig_2d = go.Figure()

    # Base contour
    fig_2d.add_trace(go.Contour(
        z=Z, x=x_vals, y=y_vals,
        colorscale='Viridis',
        contours=dict(start=np.min(Z), end=np.max(Z), size=(np.max(Z) - np.min(Z)) / 20),
        name="Function Contour",
        showscale=True
    ))

    # Descent path
    fig_2d.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=4),
        name="Descent Path"
    ))

    # Constraint contours
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

    # Marker and dashed arrow
    if expansion_point is not None:
        a, b = expansion_point
        fig_2d.add_trace(go.Scatter(
            x=[a], y=[b],
            mode='markers+text',
            marker=dict(color='black', size=9, symbol="circle"),
            text=["(a, b)"],
            textposition="bottom center",
            textfont=dict(size=11),
            name='Expansion Point'
        ))

        if len(path) > 1:
            x1, y1 = path[1]
            fig_2d.add_trace(go.Scatter(
                x=[a, x1], y=[b, y1],
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                name="Expansion → 1st Step"
            ))

    # 2nd-order Taylor contour
    if show_2nd and Z_t2 is not None:
        fig_2d.add_trace(go.Contour(
            z=Z_t2, x=x_vals, y=y_vals,
            showscale=False,
            colorscale='Blues',
            opacity=0.4,
            name="2nd-Order Taylor"
        ))

    fig_2d.update_layout(
        title="2D Contour + Constraints + Taylor Overlay",
        xaxis_title="x",
        yaxis_title="y",
        height=500,
        margin=dict(l=40, r=20, b=20, t=50),
        dragmode='zoom',
        hovermode='closest',
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(
                label="Reset Zoom",
                method="relayout",
                args=[{"xaxis.autorange": True, "yaxis.autorange": True}]
            )],
            x=0, y=0.92, xanchor='left', yanchor='top'
        )]
    )

    fig_2d.update_xaxes(scaleanchor=None)
    fig_2d.update_yaxes(scaleanchor=None)

    st.plotly_chart(fig_2d, use_container_width=True)
