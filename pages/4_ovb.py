import os
import random
import sys

import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from matplotlib import pyplot as plt

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

utl.local_css("src/styles/styles_pages.css")

random_seed = 0

if "random_seed" not in st.session_state:
    st.session_state.random_seed = 0

# DASHBOARD COLUMNS
# Initial text column
s1, c01, s2 = utl.wide_col()
# Input beta 0, sigma, n, Choose X
input_col_b1, input_col_b2, input_col_cov = st.columns(3)
# Input beta 1 and 2
# input_col_1, input_col_2 = st.columns((0.5, 1))

s1, c02, s2 = utl.wide_col()

# Chart columns
chart1_col, chart2_col = st.columns(2)
chart3_col, chart4_col = st.columns(2)

# Resample data col
resample_col, _ = st.columns(2)
# Table and chart columns
_, table1_col, _, table2_col, _ = st.columns((0.25, 1, 0.01, 0.5, 0.25))

### Data viz part

### Theory part


### PAGE START ###
# Dashboard header

with c01:
    st.title("Omitted Variable Bias")
    st.divider()
    st.header("Visualizing OVB")

    st.markdown(
        r"""Suppose $y$ is causally determined by two variables, $x_1$ and $x_2$. However, you do not think $x_2$ is related to $y$ (or you cannot observe it) and thus do not include it in the regression.
        If $x_1$ and $x_2$ are correlated, omitting $x_2$ from the regression will lead to a biased estimate of the coefficient on $x_1$.
        The difference between the true $\beta_1$ and expected $\hat{\alpha}_1$ when $x_2$ is omitted is called the *omitted variable bias*.
        """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
    \begin{array}{l r}
        \text{True $y$ DGP:} & y_i = \beta_0 + \beta_1 x_{1, i} + \beta_2 x_{2, i} + \varepsilon_i \\
        \text{True $x_2$ DGP: } & x_{2, i} = \gamma_1 x_{1, i} + \nu_i \\
        \text{Estimated $y$ DGP:} & y_i = \alpha_0 + \alpha_1 x_{1, i} + u_i \\  
        
        \text{OVB:} & E[\hat{\alpha}_1] - \beta_1 = \beta_2 \frac{cov(x_1, x_2)}{var(x_1)} = \beta_2 \gamma_1 \\
                
    \end{array}
    """
    )
    st.write("")

    st.markdown(
        r"""We can visualize this bias by comparing the fitted lines from the true and assumed models, and also look at how the OVB depends on the relationship between $x_1$ and $x_2$.
        Residuals in OVB model are $u_i = \beta_2 x_{2, i} + \varepsilon_i$, so $E[u_i|x_1] \neq 0$ if $cov(x_1, x_2) \neq 0$, which is equivalent of saying that $x_1$ is endogenous.<br>
        Note that $\varepsilon \sim N(0, \sigma^2)$, where $\sigma=20$. Sample size $n=500$.
        """,
        unsafe_allow_html=True,
    )


def gen_reg_data(b0, b1, b2, sd, N, cov_x1_x2, rseed):
    np.random.seed(rseed)

    beta = np.array([b0, b1, b2])

    X1 = np.random.uniform(-5, 5, 1000)
    alpha = cov_x1_x2
    sigma_eta = np.sqrt(1 - cov_x1_x2**2)
    eta = np.random.normal(0, sigma_eta, size=len(X1))
    X2 = alpha * X1 + eta

    # Generate random noise
    eps = np.random.normal(0, sd, 1000)

    # Take first N samples
    X1 = X1[:N]
    X2 = X2[:N]
    eps = eps[:N]

    X = np.column_stack((X1, X2))
    X = sm.add_constant(X)

    X_omit_X2 = X[:, [0, 1]]

    # Generate y = b0 + b1*X1 + b2*X2 + eps
    y = np.dot(X, beta) + eps

    # OLS regression with both X1 and X2
    model = sm.OLS(y, X).fit()
    y_hat = model.predict(X)
    # Confidence intervals
    ci = model.get_prediction(X).conf_int(alpha=0.05)
    # True standard error
    s = model.mse_resid**0.5

    # OLS regression with just X1 to show OVB
    model_ovb = sm.OLS(y, X_omit_X2).fit()
    y_hat_ovb = model_ovb.predict(X_omit_X2)

    # OLS regression of X2 on X1
    model_x2_x1 = sm.OLS(X2, X_omit_X2).fit()
    x2_hat = model_x2_x1.predict(X_omit_X2)

    return {
        "y": y,
        "y_hat": y_hat,
        "y_hat_ovb": y_hat_ovb,
        "x2_hat": x2_hat,
        "beta": beta,
        "ci": ci,
        "s": s,
        "X": X,
        "model": model,
        "model_ovb": model_ovb,
        "model_x2_x1": model_x2_x1,
    }


if "reg_data" not in st.session_state:
    st.session_state.reg_data = gen_reg_data(
        0.0, -3.0, 3.0, 15.0, 200, 0.5, st.session_state.random_seed
    )

# with input_col_x:
#     plot_x = st.radio(
#         "**X perspective to plot**",
#         ("*x1*", "*x2*"),
#         horizontal=True,
#     )
#     plot_x = plot_x[1:-1]
plot_x = "x1"

# Intercept, β0
# b0_cust = input_col_0.number_input(
#     r"$\beta_0$",
#     min_value=-10.0,
#     max_value=10.0,
#     value=0.0,
#     step=1.0,
# )
b0_cust = 0

# Error SD, √var(ϵ)=σ
var_cust = 20

# Sample size, n
n_cust = 500


# Slope for x1, β1
b1_cust = input_col_b1.number_input(
    r"$\beta_1$",
    min_value=-10.0,
    max_value=10.0,
    value=-5.0,
    step=1.0,
)

# Slope for x2, β2
b2_cust = input_col_b2.number_input(
    r"$\beta_2$",
    min_value=-10.0,
    max_value=10.0,
    value=5.0,
    step=1.0,
)

# Slope for x2, β2
cov_cust = input_col_cov.number_input(
    r"$\gamma_1$",
    min_value=-0.9,
    max_value=0.9,
    value=0.5,
    step=0.1,
)

st.session_state.reg_data = gen_reg_data(
    b0_cust,
    b1_cust,
    b2_cust,
    var_cust,
    n_cust,
    cov_cust,
    st.session_state.random_seed,
)

# 3D OVB PLOT
pio.templates.default = "my_streamlit"


def create_3d_plot(data, view_var="x1"):
    fig = go.Figure()

    # hover text for all scatters
    indices = list(range(len(data["y"])))
    hover_text_true = [
        f"x<sub>1</sub>: {x:.2f}, x<sub>2</sub>: {y:.2f}, y: {z:.2f}<br>i: {i}"
        for x, y, z, i in zip(data["X"][:, 1], data["X"][:, 2], data["y"], indices)
    ]

    # Scatter trace for the data
    fig.add_trace(
        go.Scatter3d(
            x=data["X"][:, 1],
            y=data["X"][:, 2],
            z=data["y"],
            mode="markers",
            marker=dict(size=4.5, opacity=0.5, color=thm.cols_set1_px["blue"]),
            name="y<sub>true</sub>",
            hovertext=hover_text_true,
            hoverinfo="text",
            # hovertemplate="x1:%{x:.2f}<br>%x2: {y:.2f}<br>%y: {z:.2f}<extra></extra>",
            # hovertemplate="x<sub>1</sub>: %{x:.2f}<br>x<sub>2</sub>: %{y:.2f}<br>y: %{z:.2f}<extra></extra>",
            showlegend=False,
        )
    )

    # Generate x-ranges for predicted plane/lines
    x1_range = np.linspace(-5, 5, 10)
    x2_range = np.linspace(-5, 5, 10)

    # Generate a meshgrid for predicted plane
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)

    # Calculate z values for the plane using the coefficients from the model
    b0, b1, b2 = data["model"].params
    y_grid = b0 + b1 * x1_grid + b2 * x2_grid

    ### FITTED PLOTS ###
    # Scatter trace for the fitted y-values (y_hat)
    hover_text_fit = [
        f"x<sub>1</sub>: {x:.2f}, x<sub>2</sub>: {y:.2f}, ŷ: {z:.2f}<br>i: {i}"
        for x, y, z, i in zip(data["X"][:, 1], data["X"][:, 2], data["y"], indices)
    ]

    fig.add_trace(
        go.Scatter3d(
            x=data["X"][:, 1],
            y=data["X"][:, 2],
            z=data["y_hat"],
            mode="markers",
            marker=dict(size=4.5, opacity=0.7, color=thm.cols_set1_px["green"]),
            name="y<sub>hat</sub>",
            hovertext=hover_text_fit,
            hoverinfo="text",
            showlegend=False,
        )
    )

    # Surface trace for the fitted plane
    fig.add_trace(
        go.Surface(
            x=x1_range,
            y=x2_range,
            z=y_grid,
            colorscale=[
                [0, thm.cols_set1_px["green"]],
                [1, thm.cols_set1_px["green"]],
            ],
            showscale=False,
            opacity=0.2,
            name="Fitted Plane",
            hoverinfo="none",
            # hoveron="points",
            # hidesurface=True,
            contours=dict(
                x=dict(highlight=False),
                y=dict(highlight=False),
                z=dict(highlight=False),
            ),
        )
    )

    # Add fitted line for X1 (with X2 = 0)
    x2_vals_0 = np.zeros_like(x1_range)
    y_hat_x1 = b0 + b1 * x1_range
    fig.add_trace(
        go.Scatter3d(
            x=x1_range,
            y=x2_vals_0,
            z=y_hat_x1,
            mode="lines",
            line=dict(color=thm.cols_set1_px["orange"], width=10),
            name="y<sub>hat</sub> = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub>",
            hovertemplate="x<sub>1</sub>: %{x:.2f}<br>x<sub>2</sub>: %{y:.2f}<br>y<sub>hat</sub>: %{z:.2f}<extra></extra>",
        )
    )

    # Add fitted line for X2 (with X1 = 0)
    x1_vals_0 = np.zeros_like(x2_range)
    y_hat_x2 = b0 + b2 * x2_range
    fig.add_trace(
        go.Scatter3d(
            x=x1_vals_0,
            y=x2_range,
            z=y_hat_x2,
            mode="lines",
            line=dict(color=thm.cols_set1_px["purple"], width=10),
            name="y<sub>hat</sub> = b<sub>0</sub> + b<sub>2</sub>x<sub>2</sub>",
            hovertemplate="x<sub>1</sub>: %{x:.2f}<br>x<sub>2</sub>: %{y:.2f}<br>y<sub>hat</sub>: %{z:.2f}<extra></extra>",
        )
    )

    # Axis lines with text
    axis_line_props = dict(color="black", width=3)  # Set color and width

    # change camera view based on x selected
    # x, y, and z determine the position of the camera's 'eye'.
    # A larger value of x moves the 'eye' rightwards (and vice versa).
    # A larger value of y moves the 'eye' upwards (and vice versa).
    # A larger value of z moves the 'eye' closer to the plot (and vice versa).

    # changing axes positioning based on x selected - don't think it's more intuitive
    axis_perspective_change = False
    xaxis_x_pos = [-5.5, 5.2]
    xaxis_y_pos = [-5.5, -5.5]
    yaxis_x_pos = [-5.5, -5.5]
    yaxis_y_pos = [-5.5, 5.2]
    zaxis_x_pos = [-5.5, -5.5]
    zaxis_y_pos = [-5.5, -5.5]

    if view_var == "x1":
        camera_view = dict(eye=dict(x=0, y=-1.92, z=0))
    elif view_var == "x2":
        camera_view = dict(eye=dict(x=1.75, y=0, z=0))
        if axis_perspective_change:
            xaxis_x_pos = [-5.5, 5.2]
            xaxis_y_pos = [-5.5, -5.5]
            yaxis_x_pos = [5.5, 5.5]
            yaxis_y_pos = [-5.5, 5.2]
            zaxis_x_pos = [5.5, 5.5]
            zaxis_y_pos = [-5.5, -5.5]

    # x-axis line with text
    fig.add_trace(
        go.Scatter3d(
            x=xaxis_x_pos,
            y=xaxis_y_pos,
            z=[-100, -100],
            text=["", "x<sub>1</sub>"],
            mode="lines+text",
            line=axis_line_props,
            textposition="middle right",
            textfont=dict(family="Sans-Serif", size=18, color="black"),
            showlegend=False,
        )
    )

    # y-axis line with text
    fig.add_trace(
        go.Scatter3d(
            x=yaxis_x_pos,
            y=yaxis_y_pos,
            z=[-100, -100],
            text=["", "x<sub>2</sub>"],
            mode="lines+text",
            line=axis_line_props,
            textposition="middle right",
            textfont=dict(family="Sans-Serif", size=18, color="black"),
            showlegend=False,
        )
    )

    # z-axis line with text
    fig.add_trace(
        go.Scatter3d(
            x=zaxis_x_pos,
            y=zaxis_y_pos,
            z=[-100, 98],  # placed text within range
            text=["", "y"],
            mode="lines+text",
            line=axis_line_props,
            textposition="middle right",
            textfont=dict(family="Sans-Serif", size=18, color="black"),
            showlegend=False,
        )
    )

    # Marking 0 on x1 axis
    fig.add_trace(
        go.Scatter3d(
            x=[0],
            y=[-5.5],
            z=[-93],
            mode="text",
            text=["0"],
            textposition="bottom center",
            textfont=dict(color="black", size=16),
            showlegend=False,
        )
    )

    # Marking 0 on x2 axis
    fig.add_trace(
        go.Scatter3d(
            x=[-5.5],
            y=[0],
            z=[-93],
            mode="text",
            text=["0"],
            textposition="bottom center",
            textfont=dict(color="black", size=16),
            showlegend=False,
        )
    )

    # Function to add a tickmark
    def add_tickmark(fig, x_start, y_start, z_start, x_end, y_end, z_end):
        fig.add_trace(
            go.Scatter3d(
                x=[x_start, x_end],
                y=[y_start, y_end],
                z=[z_start, z_end],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            )
        )

    # Add tickmark for x1 axis at origin
    add_tickmark(fig, 0, -5.5, -100, 0, -5.5, -95)

    # Add tickmark for x2 axis at origin
    add_tickmark(fig, -5.5, 0, -100, -5.5, 0, -95)

    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        # margin=dict(autoexpand=True, l=30, r=40, t=25, b=40, pad=0),
        # hovermode=False,
        legend=dict(
            x=0,  # x-position of the legend
            y=1,  # y-position of the legend
            xanchor="left",  # 'left' ensures the legend starts at the specified x-position
            yanchor="top",  # 'top' ensures the legend is positioned with its top at the specified y-position
            bordercolor="Black",
            borderwidth=1,
            bgcolor="white",
        ),
        showlegend=False,
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=1.1, y=1.1, z=1),
            camera=camera_view,
            xaxis=dict(
                title_text="",  # Remove title
                range=[-5.5, 5.5],
                showgrid=False,
                zeroline=False,  # Remove zeroline
                showticklabels=False,  # Hide tick labels
                showspikes=False,  # Remove spikes
            ),
            yaxis=dict(
                title_text="",  # Remove title
                range=[-5.5, 5.5],
                showgrid=False,
                zeroline=False,  # Remove zeroline
                showticklabels=False,  # Hide tick labels
                showspikes=False,  # Remove spikes
            ),
            zaxis=dict(
                title_text="",  # Remove title
                range=[-105, 100],
                showgrid=False,
                zeroline=False,  # Remove zeroline
                showticklabels=False,  # Hide tick labels
                showspikes=False,  # Remove spikes
            ),
            bgcolor="white",  # Set background color
        ),
        width=400,
        height=400,
    )

    return fig


def color_and_margin(fig):
    """
    Update plot and paper background color because streamlit overwrites Plotly template.

    """
    fig.update_layout(
        plot_bgcolor="white",
        # paper_bgcolor="rgb(235, 255, 250)",
        paper_bgcolor="white",
        margin=dict(t=30, autoexpand=True),
    )
    return None


#### 2D OVB PLOT


def plot_ols_plotly(data_custom, beta_true, show="x1"):
    # Prepare data
    show_X = data_custom["X"][:, 1] if show == "x1" else data_custom["X"][:, 2]

    b0, b1, b2 = beta_true

    b0_ols = data_custom["model"].params[0]
    b1_ols = data_custom["model"].params[1]
    b2_ols = data_custom["model"].params[2]

    b0_ovb = data_custom["model_ovb"].params[0]
    b1_ovb = data_custom["model_ovb"].params[1]

    indices = list(range(len(data_custom["y"])))
    # Fitted line for one x
    x_range = np.linspace(-5, 5, 10)
    if show == "x1":
        y_from_one_x = b0_ols + b1_ols * x_range
        color_one_x = thm.cols_set1_px["orange"]

    elif show == "x2":
        y_from_one_x = b0_ols + b2_ols * x_range
        color_one_x = thm.cols_set1_px["purple"]

    y_ovb = b0_ovb + b1_ovb * x_range

    fig = go.Figure()

    # Scatter for Sample data
    if show == "x1":
        hover_text_fit = [
            f"x<sub>1</sub>: {x:.2f}, x<sub>2</sub>: {y:.2f}, ŷ: {z:.2f}<br>i: {i}"
            for x, y, z, i in zip(
                data_custom["X"][:, 1],
                data_custom["X"][:, 2],
                data_custom["y"],
                indices,
            )
        ]
    elif show == "x2":
        hover_text_fit = [
            f"x<sub>1</sub>: {x:.2f}, x<sub>2</sub>: {y:.2f}, ŷ: {z:.2f}<br>i: {i}"
            for x, y, z, i in zip(
                data_custom["X"][:, 1],
                data_custom["X"][:, 2],
                data_custom["y"],
                indices,
            )
        ]

    fig.add_trace(
        go.Scatter(
            x=show_X,
            y=data_custom["y"],
            mode="markers",
            marker=dict(color=thm.cols_set1_px["blue"], opacity=0.5),
            hoverinfo="text",
            hovertext=hover_text_fit,
            name=f"y = {b0:.2f} + {b1:.2f}x<sub>1</sub> + {b2:.2f}x<sub>2</sub> + ε",
        )
    )

    # Scatter for Fitted markers
    fig.add_trace(
        go.Scatter(
            x=show_X,
            y=data_custom["y_hat"],
            mode="markers",
            marker=dict(color=thm.cols_set1_px["green"], opacity=0.5),
            hoverinfo="text",
            hovertext=hover_text_fit,
            name=f"ŷ = {b0_ols:.2f} + {b1_ols:.2f}x<sub>1</sub> + {b2_ols:.2f}x<sub>2</sub>",
        )
    )

    # Line for Fitted line for one x
    name_fitted_2d = "x<sub>1</sub>" if show == "x1" else "x<sub>2</sub>"
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_from_one_x,
            mode="lines",
            line=dict(color=color_one_x),
            name=f"True fit: ŷ = {b0_ols:.2f} + {b1_ols if show=='x1' else b2_ols:.2f}{name_fitted_2d}",
            hoverinfo="none",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_ovb,
            mode="lines",
            line=dict(color="red"),
            name=f"OVB fit: ŷ = {b0_ovb:.2f} + {b1_ovb if show=='x1' else b1_ovb:.2f}{name_fitted_2d}",
            hoverinfo="none",
        )
    )

    # R-squared
    r_squared = data_custom["model"].rsquared
    x_label = "x<sub>1</sub>" if show == "x1" else "x<sub>2</sub>"
    # Update layout
    fig.update_layout(
        # Paper options
        margin=dict(autoexpand=True, l=30, r=40, t=25, b=40, pad=0),
        # Legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        # Hover
        hovermode="closest",
        # Axes
        xaxis=dict(
            zeroline=False,
            showgrid=False,
            range=[-5.5, 5.5],
            autorange=False,
            title="",  # Empty the title here as we'll add it as an annotation
            tickvals=[-3, 0, 3],
            tickfont=dict(size=14),  # font size for tick marks
            showspikes=False,
        ),
        yaxis=dict(
            zeroline=False,
            showgrid=False,
            range=[-70, 70],
            autorange=False,
            title="y",
            title_font=dict(size=18),
            # tickvals=list(range(-100, 101, 30)),
            tickvals=[-50, 0, 50],
            tickfont=dict(size=14),  # font size for tick marks
            tickformat="d",  # format as integers
            showticklabels=True,
            showspikes=False,
        ),
        # Adding annotations for the axis labels
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=1,
                y=-0.1,
                text=x_label,
                font=dict(size=20),
                showarrow=False,
            ),
            # The R-squared annotation remains unchanged
            dict(
                text=f"R<sup>2</sup> = {r_squared:.2f}",
                font=dict(size=20),
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.99,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.7)",
            ),
        ],
    )

    color_and_margin(fig)

    return fig


def plot_x2_x1_plotly(data_custom, beta_true, show="x1"):
    # Prepare data
    b0, b1, b2 = beta_true

    x1 = data_custom["X"][:, 1]
    x2 = data_custom["X"][:, 2]

    gamma_0 = data_custom["model_x2_x1"].params[0]
    gamma_1 = data_custom["model_x2_x1"].params[1]

    indices = list(range(len(data_custom["y"])))
    # Fitted line for one x
    x_range = np.linspace(-5, 5, 10)

    x2_line = gamma_0 + gamma_1 * x_range
    x2_hat = data_custom["x2_hat"]

    fig = go.Figure()

    # Scatter for Sample data
    hover_text_fit = [
        f"x<sub>1</sub>: {x:.2f}, x<sub>2</sub>: {y:.2f}, ŷ: {z:.2f}<br>i: {i}"
        for x, y, z, i in zip(
            data_custom["X"][:, 1],
            data_custom["X"][:, 2],
            data_custom["y"],
            indices,
        )
    ]

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=x2,
            mode="markers",
            marker=dict(color=thm.cols_set1_px["brown"], opacity=0.5),
            hoverinfo="text",
            hovertext=hover_text_fit,
            name=f"x<sub>2</sub> = {gamma_0:.2f} + {gamma_1:.2f}x<sub>1</sub> + ν",
        )
    )

    # Line for Fitted line for one x
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=x2_line,
            mode="lines",
            line=dict(color=thm.cols_set1_px["brown"]),
            name=f"x̂<sub>2</sub> = {gamma_0:.2f} + {gamma_1:.2f}x<sub>1</sub>",
            hoverinfo="none",
        )
    )

    # R-squared
    r_squared = data_custom["model_x2_x1"].rsquared

    # Update layout
    fig.update_layout(
        # Paper options
        margin=dict(autoexpand=True, l=30, r=40, t=25, b=40, pad=0),
        # Legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        # Hover
        hovermode="closest",
        # Axes
        xaxis=dict(
            zeroline=False,
            showgrid=False,
            range=[-5.5, 5.5],
            autorange=False,
            title="",  # Empty the title here as we'll add it as an annotation
            tickvals=[-3, -1, 0, 1, 3],
            tickfont=dict(size=14),  # font size for tick marks
            showspikes=False,
        ),
        yaxis=dict(
            zeroline=False,
            showgrid=False,
            range=[-5.5, 5.5],
            autorange=False,
            title="x<sub>2</sub>",
            title_font=dict(size=18),
            # tickvals=list(range(-100, 101, 30)),
            tickvals=[-3, -1, 0, 1, 3],
            tickfont=dict(size=14),  # font size for tick marks
            tickformat="d",  # format as integers
            showticklabels=True,
            showspikes=False,
        ),
        # Adding annotations for the axis labels
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=1,
                y=-0.1,
                text=r"x<sub>1</sub>",
                font=dict(size=20),
                showarrow=False,
            ),
            # The R-squared annotation remains unchanged
            dict(
                text=f"R<sup>2</sup> = {r_squared:.2f}",
                font=dict(size=20),
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.99,
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.7)",
            ),
        ],
    )

    color_and_margin(fig)

    return fig


def plot_eps_x1_plotly(data_custom, beta_true, show="x1"):

    x1 = data_custom["X"][:, 1]

    residuals = data_custom["y"] - (
        data_custom["model"].params[0] + data_custom["model"].params[1] * x1
    )

    indices = list(range(len(data_custom["y"])))

    # Fitted line for one x
    x1_with_const = sm.add_constant(x1)
    model_residuals_on_x1 = sm.OLS(residuals, x1_with_const).fit()

    x_range = np.linspace(-5, 5, 10)

    fitted_line = model_residuals_on_x1.predict(sm.add_constant(x_range))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x1,
            y=residuals,
            mode="markers",
            marker=dict(color=thm.cols_set1_px["pink"], opacity=0.5),
            hoverinfo="none",
            name=f"True residuals from OVB model",
        )
    )

    # Line for Fitted line for one x
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=fitted_line,
            mode="lines",
            line=dict(color=thm.cols_set1_px["pink"]),
            name=f"E[u|x<sub>1</sub>]",
            hoverinfo="none",
        )
    )

    # Update layout
    fig.update_layout(
        # Paper options
        margin=dict(autoexpand=True, l=30, r=40, t=25, b=40, pad=0),
        # Legend
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255, 255, 255, 0.7)",
        ),
        # Hover
        hovermode="closest",
        # Axes
        xaxis=dict(
            zeroline=False,
            showgrid=False,
            range=[-5, 5],
            autorange=False,
            title="",  # Empty the title here as we'll add it as an annotation
            tickvals=[-3, -1, 0, 1, 3],
            tickfont=dict(size=14),  # font size for tick marks
            showspikes=False,
        ),
        yaxis=dict(
            zeroline=False,
            showgrid=False,
            range=[-60, 60],
            autorange=False,
            title="u",
            title_font=dict(size=18),
            # tickvals=list(range(-100, 101, 30)),
            tickvals=[-40, -20, 0, 20, 40],
            tickfont=dict(size=14),  # font size for tick marks
            tickformat="d",  # format as integers
            showticklabels=True,
            showspikes=False,
        ),
        # Adding annotations for the axis labels
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=1,
                y=-0.1,
                text=r"x<sub>1</sub>",
                font=dict(size=20),
                showarrow=False,
            ),
        ],
    )

    color_and_margin(fig)

    return fig


with resample_col:
    if st.button("Resample data", type="primary"):
        st.session_state.random_seed = random.randint(0, 10000)

        st.session_state.reg_data = gen_reg_data(
            b0_cust,
            b1_cust,
            b2_cust,
            var_cust,
            n_cust,
            cov_cust,
            st.session_state.random_seed,
        )

with c02:
    b1_ols = st.session_state.reg_data["model"].params[1]
    b2_ols = st.session_state.reg_data["model"].params[2]
    b1_ovb = st.session_state.reg_data["model_ovb"].params[1]

    st.markdown(
        r"Regression coefficients: full model $\hat{\beta}_1$ = "
        + f"{b1_ols:.2f}; "
        + r"$\hat{\beta}_2$ = "
        + f"{b2_ols:.2f}; "
        + r"OVB model $\hat{\alpha}_1 = $"
        + f"{b1_ovb:.2f}."
        + r"<br>Sample OVB bias: $\hat{\alpha_1} - \hat{\beta}_1$ = "
        + f"{b1_ovb - b1_ols:.2f}",
        unsafe_allow_html=True,
    )


with chart1_col:
    fig_3d = create_3d_plot(st.session_state.reg_data, view_var=plot_x)

    st.write("#### True DGP and model fit")

    st.plotly_chart(fig_3d, theme=None, use_container_width=True)


with chart2_col:
    fig_2d = plot_ols_plotly(
        st.session_state.reg_data,
        [b0_cust, b1_cust, b2_cust],
        show=plot_x,
    )

    st.write("#### True vs OVB model fit")

    st.plotly_chart(
        fig_2d,
        theme=None,
        use_container_width=True,
    )

with chart3_col:
    fig_x2_x1 = plot_x2_x1_plotly(
        st.session_state.reg_data,
        [b0_cust, b1_cust, b2_cust],
        show=plot_x,
    )

    st.markdown(r"#### Covariates relationship")

    st.plotly_chart(
        fig_x2_x1,
        theme=None,
        use_container_width=True,
    )


with chart4_col:
    fig_ovb_resid = plot_eps_x1_plotly(
        st.session_state.reg_data,
        [b0_cust, b1_cust, b2_cust],
        show=plot_x,
    )

    st.markdown(
        r"#### OVB residuals $u_i = \beta_2 x_{2, i} + \varepsilon_i$ plot against $x_1$"
    )

    st.plotly_chart(
        fig_ovb_resid,
        theme=None,
        use_container_width=True,
    )


def gen_stats_table(model):
    # Extract statistics
    r2 = model.rsquared
    r2_adj = model.rsquared_adj
    N = model.nobs

    data = [
        ("R<sup>2</sup>", r2),
        ("Adj. R<sup>2</sup>", r2_adj),
        ("\u03C3\u0302", st.session_state.reg_data["s"]),
        ("N", N),
    ]

    html_string = """<table class="table" border="1">
                    <thead>
                        <tr>
                            <th>Stat</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>"""

    for name, value in data:
        if name != "N":
            html_string += f"<tr><td>{name}</td><td>{value:.2f}</td></tr>"
        else:
            html_string += f"<tr><td>{name}</td><td>{value:.0f}</td></tr>"

    html_string += "</tbody></table>"

    return html_string


def gen_coef_table(true_betas, model):
    """
    Generate an HTML table for the true population parameters and
    estimated sample OLS parameters with standard errors and 95% confidence intervals.
    """
    # Extract coefficients, standard errors, and confidence intervals from the model
    estimated_betas = model.params
    standard_errors = model.bse
    confidence_intervals = model.conf_int(alpha=0.05)

    data = [
        (
            "β0",
            true_betas[0],
            estimated_betas[0],
            standard_errors[0],
        ),
        (
            "β1",
            true_betas[1],
            estimated_betas[1],
            standard_errors[1],
        ),
        (
            "β2",
            true_betas[2],
            estimated_betas[2],
            standard_errors[2],
        ),
    ]

    html_string = """
        <table class="table" border="1">
        <thead>
            <tr>
                <th style="width: 20%;">Coefficient</th>
                <th style="width: 15%;">Pop. Value</th>
                <th style="width: 28%;">Sample Est.<br/>(SE)</th>
                <th style="width: 37%;">95% CI</th>
            </tr>
        </thead>
        <tbody>"""

    i = 0
    for row in data:
        ci = confidence_intervals[i]
        html_string += f"<tr><td>{row[0]}</td><td>{row[1]:.2f}</td><td>{row[2]:.2f}<br/><span style='font-size: 0.8em'>({row[3]:.2f})</span></td><td>[{ci[0]:.2f}, {ci[1]:.2f}]</td></tr>"
        i += 1

    html_string += "</tbody></table>"

    return html_string


# with table1_col:
#     coef_table = gen_coef_table(
#         [b0_cust, b1_cust, b2_cust], st.session_state.reg_data["model"]
#     )

#     st.markdown(coef_table, unsafe_allow_html=True)

# with table2_col:
#     stats_table = gen_stats_table(st.session_state.reg_data["model"])
#     st.markdown(stats_table, unsafe_allow_html=True)

s0, c03, s1 = utl.wide_col()

with c03:
    st.markdown("### Interesting takeaways")

    with st.expander("Click to expand.", expanded=True):
        st.markdown(
            r"""
    
1. Varying $\beta_2$ shows the sneaky effect of OVB - looking at the 2D chart, one would think that the red line fits data much better.
    However, the effect of $x_1$ is being falsely attributed to $x_2$.
    
2. Varying $\beta_2$ and $cov(x_1, x_2)$ shows how the product of these two statistics determines the size of the bias.

3. True $\beta_1$ has no effect on the size of the bias.

4. $cov(x_1, x_2)$ has no effect on the true model fit (fitted plane in 3D does not change),
    """,
            unsafe_allow_html=True,
        )

s0, c04, s1 = utl.wide_col()

with c04:
    st.markdown("## OVB and Simpson's Paradox")

    st.markdown(
        r"""You are interested in the relationship between anual $salary$, average $grade$ in college, and studying $economics$. You think that higher grades should on average increase
            your salary, however, studying economics might lead to lower grades. When choosing what to study and how much effort to put in for getting good grades,
            you want to know what are the effects of each variable on the expected salary, so you collect individual data on alumni salaries, grades, and area of studies,
            and you will estimate these effects with linear regressions.
            """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""Suppose the true DGP (data generating process) is defined through equations below. We are interested in seeing what happens when you omit $econ$ variable
        when estimating the effect of $grade$.
            """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \begin{array}{l l}
            \text{True Salary DGP:} & salary_i = 50000 + \beta_{grade} grade_i + \beta_{econ} econ_i + \varepsilon_i \\
            \text{True Grade DGP:} & grade_i = 80 + \gamma_{econ} econ_i + \nu_i \\
            \text{Estimated DGP:} & salary_i = \alpha_0 + \alpha_{grade} grade_i + u_i \\
            \text{OVB:} & E[\hat{\alpha}_{grade}] - \beta_{grade} = \beta_{econ} \frac{cov(grade, econ)}{var(grade)} \\        
        \end{array}
        """
    )

    # st.latex(
    #     r"""
    #     \begin{array}{l l}
    #         \text{True Salary DGP:} & salary_i = 50000 + \beta_{grade} grade_i + \beta_{econ} econ_i + \beta_{econ\_grade} grade_i \times econ_i + \varepsilon_i \\
    #         \text{True grade DGP:} & grade_i = 70 + \gamma_{econ} econ_i + \nu_i \\
    #     \end{array}
    #     """
    # )

    st.write("True DGP parameters are defined below:")
# Input beta 0, sigma, n, Choose X
input_col_b_gr, input_col_b_econ, input_gamma, input_n = st.columns(4)

beta_grade = input_col_b_gr.number_input(
    r"$\beta_{grade}$",
    min_value=0,
    max_value=4000,
    value=500,
    step=250,
)

beta_econ = input_col_b_econ.number_input(
    r"$\beta_{econ}$",
    min_value=-10000,
    max_value=70000,
    value=50000,
    step=10000,
)

gamma_econ = input_gamma.number_input(
    r"$\gamma_{econ}$",
    min_value=-50,
    max_value=-10,
    value=-30,
    step=5,
)

N = input_n.number_input(
    r"Sample size, $N$",
    min_value=50,
    max_value=20000,
    value=10000,
    step=1000,
)


# Generate data
def gen_data(
    N,
    b_maj_salary=10000,
    b_gpa_salary=2000,
    b_maj_gpa=-10,
    incl_leis=False,
    b_leis_salary=-1000,
    b_maj_leis=-1000,
    b_leis_gpa=-3,
    rseed=12345,
):

    np.random.seed(rseed)

    major = np.random.choice([0, 1], size=N)

    if incl_leis:
        e_maj_leis = 1
        leisure = (
            5 + b_maj_leis * major + np.random.normal(loc=0, scale=e_maj_leis, size=N)
        )
    else:
        leisure = 0

    e_gpa = 10
    gpa = (
        80
        + b_maj_gpa * major
        + b_leis_gpa * leisure
        + np.random.normal(loc=0, scale=e_gpa, size=N)
    )

    e_salary = 10000
    salary = (
        50000
        + b_maj_salary * major
        + b_gpa_salary * gpa
        + b_leis_salary * leisure
        + np.random.normal(loc=0, scale=e_salary, size=N)
    )

    return pd.DataFrame(
        {"salary": salary, "major": major, "gpa": gpa, "leisure": leisure}
    )


reg_data = gen_data(
    N,
    b_maj_salary=beta_econ,
    b_gpa_salary=beta_grade,
    b_maj_gpa=gamma_econ,
    incl_leis=False,
)


def plot_salary_gpa(data, formula):
    # Extract regression coefficients
    model = smf.ols(formula=formula, data=data).fit()
    beta_gpa_est = model.params["gpa"]
    beta_const = model.params["Intercept"]

    # Generate regression line
    gpa_range = np.linspace(data["gpa"].min(), data["gpa"].max(), 100)
    salary_pred = beta_const + beta_gpa_est * gpa_range

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        data["gpa"], data["salary"], alpha=0.6, label="Observed Data (all majors)"
    )
    ax.plot(
        gpa_range,
        salary_pred,
        color="red",
        linewidth=2,
        label=f"Regression Line: Salary = {beta_const:.0f} + {beta_gpa_est:.0f}×grade",
    )

    # Add labels, legend, and title
    ax.set_xlabel("Grade", fontweight="bold")
    ax.set_ylabel("Salary (Thousands USD)", fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ax.set_title("Salary as a Function of GPA", fontweight="bold")
    ax.set_xlim([0, 115])
    ax.set_ylim([0, 300000])
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    return fig


def plot_salary_gpa_major(data, formula="salary ~ gpa + major"):
    # Fit the regression model
    model = smf.ols(formula=formula, data=data).fit()
    beta_const = model.params["Intercept"]
    beta_gpa = model.params["gpa"]
    beta_major = model.params["major"]

    # Generate GPA range
    gpa_range = np.linspace(data["gpa"].min(), data["gpa"].max(), 100)

    # Predict salary for each major group
    salary_major_0 = beta_const + beta_gpa * gpa_range  # For major = 0
    salary_major_1 = beta_const + beta_gpa * gpa_range + beta_major  # For major = 1

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color coding for major
    colors = data["major"].map({0: thm.set1_purple, 1: thm.set1_orange})

    ax.scatter(data["gpa"], data["salary"], c=colors, alpha=0.6)

    # Plot regression lines
    ax.plot(
        gpa_range,
        salary_major_0,
        color=thm.set1_purple,
        linewidth=2,
        label="Econ=0: Salary = " f"{beta_const:.0f} + {beta_gpa:.0f}xgrade",
    )
    ax.plot(
        gpa_range,
        salary_major_1,
        color=thm.set1_orange,
        linewidth=2,
        label="Econ=1: Salary = " f"{beta_const+beta_major:.0f} + {beta_gpa:.0f}xgrade",
    )

    # Add labels, legend, and title
    ax.set_xlabel("Grade", fontweight="bold")
    ax.set_ylabel("Salary (Thousands USD)", fontweight="bold")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

    # ax.set_title("Salary as a Function of GPA and Major", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xlim([0, 115])
    ax.set_ylim([0, 300000])
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    return fig


chart1_col, chart2_col = st.columns(2)

with chart1_col:
    st.markdown("#### Salary vs GPA")
    st.pyplot(plot_salary_gpa(reg_data, "salary ~ gpa"))


with chart2_col:
    st.markdown("#### Salary vs GPA, by Major")
    st.pyplot(plot_salary_gpa_major(reg_data, "salary ~ gpa + major"))

s1, c03, s2 = utl.wide_col()

with c03:
    ovb_model = smf.ols(formula="salary ~ gpa", data=reg_data).fit()
    alpha_gpa_ovb = ovb_model.params["gpa"]

    true_model = smf.ols(formula="salary ~ gpa + major", data=reg_data).fit()
    beta_gpa_est = true_model.params["gpa"]
    beta_econ_est = true_model.params["major"]

    model_gpa_major = smf.ols(formula="major ~ gpa", data=reg_data).fit()
    gamma_major_gpa = model_gpa_major.params["gpa"]

    st.markdown(
        r"Parameter estimates:"
        + r" $\hat{\alpha}_{grade}$"
        + f" = {alpha_gpa_ovb:.1f};      "
        + r"$\hat{\beta}_{grade}$"
        + f" = {beta_gpa_est:.1f};     "
        + r" $\hat{\beta}_{econ}$"
        + f" = {beta_econ_est:.1f};     "
        + r"$\hat{\gamma}_{econ\_on\_grade}$"
        + f" = {gamma_major_gpa:.3f}.",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"OVB: $\hat{\alpha}_{grade} - \hat{\beta}_{grade}$"
        + f" = {alpha_gpa_ovb-beta_gpa_est:.1f}. Using OVB formula: "
        + r" $\hat{\beta}_{econ} \frac{cov(grade, econ)}{var(grade)} = \hat{\beta}_{econ} \hat{\gamma}_{econ\_on\_grade}$"
        + f" = {beta_econ_est * gamma_major_gpa:.1f}.",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"OVB relative to true population parameter: $\hat{\alpha}_{grade} - \beta_{grade}$"
        + f" = {alpha_gpa_ovb-beta_grade:.1f}.",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""Simpson's Paradox refers to a situation where the aggregate effect has the opposite sign than the within-group effect.
                This can be viewed as omitting the group variable (and potentially the interaction between group and the variable of interest),
                 which is a relevant factor in the true DGP. In our case, we saw that if $econ$ has a relatively strong effect on $salary$ and $grade$,
                 while $grade$ has a relatively weak effect on salary, then omitting $econ$ from the regression will give a false sign on the $grade$ effect.""",
        unsafe_allow_html=True,
    )

s1, c04, s2 = utl.wide_col()

with c04:
    st.markdown("## OVB with 3 variables")

    st.markdown(
        r"""You are interested in the same question as above, but you hypothesize that leisure can also be an important variable, which affects both the expected salary and the grades,
        and also might be influenced by your area of study. We will look what happens when you omit both $econ$ and $leisure$ or just $econ$ variable
        when estimating the effect of $grade$.
            """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""Suppose the true DGP (data generating process) is defined through equations below. 
            """,
        unsafe_allow_html=True,
    )

    st.latex(
        r"""
        \begin{array}{l l}
            \text{True Salary DGP:} & salary_i = 50000 + \beta_{grade} grade_i + \beta_{econ} econ_i + \beta_{leis}leis_i + \varepsilon_i \\
            \text{True Grade DGP:} & grade_i = 80 + \gamma_{econ} econ_i + \gamma_{leis} leis_i + \nu_i \\
            \text{True Leisure DGP:} & leis_i = 5 + \delta_{econ} econ_i + \eta_i \\     
            \text{} & \\        
            \text{Estimated DGP:} & salary_i = \alpha_0 + \alpha_{grade} grade_i + u_i \\
            \text{OVB:} & E[\hat{\alpha}_{grade}] - \beta_{grade} = \beta_{econ} \frac{cov(grade, econ)}{var(grade)} + \beta_{leis} \frac{cov(grade, leis)}{var(grade)} \\        
        \end{array}
        """
    )

    st.write("True DGP parameters are defined below:")

input_col_b_gr, input_col_b_econ, input_col_b_leis = st.columns(3)
input_col_g_econ, input_col_g_leis, input_col_d_econ = st.columns(3)


beta_grade = input_col_b_gr.number_input(
    r"$\beta_{grade}$", min_value=0, max_value=4000, value=500, step=250, key="123"
)

beta_econ = input_col_b_econ.number_input(
    r"$\beta_{econ}$",
    min_value=-10000,
    max_value=70000,
    value=50000,
    step=10000,
    key="124",
)

beta_leis = input_col_b_leis.number_input(
    r"$\beta_{leis}$",
    min_value=-15000,
    max_value=10000,
    value=-5000,
    step=1000,
)

gamma_econ = input_col_g_econ.number_input(
    r"$\gamma_{econ}$", min_value=-50, max_value=-10, value=-30, step=5, key="125"
)

gamma_leis = input_col_g_leis.number_input(
    r"$\gamma_{leis}$",
    min_value=-15,
    max_value=5,
    value=-5,
    step=1,
)

delta_econ = input_col_d_econ.number_input(
    r"$\delta_{econ}$",
    min_value=-5,
    max_value=0,
    value=-3,
    step=1,
)

reg_data_leisure = gen_data(
    10000,
    b_maj_salary=beta_econ,
    b_gpa_salary=beta_grade,
    b_maj_gpa=gamma_econ,
    incl_leis=True,
    b_leis_salary=beta_leis,
    b_maj_leis=delta_econ,
    b_leis_gpa=gamma_leis,
)

s1, c05, s2 = utl.wide_col()


model_true = smf.ols(
    formula="salary ~ gpa + major + leisure", data=reg_data_leisure
).fit()
model_summary = model_true.summary().as_text()

with c05:
    with st.expander(
        "Model summary when including all relevant variables:", expanded=False
    ):
        st.text("Sample estimates when including all relevant variables:")
        st.code(model_summary, language="plaintext")


# def plot_salary_gpa_major_leisure(data, formula="salary ~ gpa + major + leisure"):
#     # Fit the regression model
#     model = smf.ols(formula=formula, data=data).fit()
#     beta_const = model.params["Intercept"]
#     beta_gpa = model.params["gpa"]
#     beta_major = model.params["major"]

#     # Generate GPA range
#     gpa_range = np.linspace(data["gpa"].min(), data["gpa"].max(), 100)

#     # Predict salary for each major group
#     salary_major_0 = beta_const + beta_gpa * gpa_range  # For major = 0
#     salary_major_1 = beta_const + beta_gpa * gpa_range + beta_major  # For major = 1

#     # Compute average leisure
#     avg_leisure = data["leisure"].mean()

#     # Create leisure-based categories
#     data["leisure_category"] = np.where(
#         data["leisure"] > avg_leisure, "Above Avg", "Below Avg"
#     )

#     # Define colors for each combination of major and leisure category
#     color_map = {
#         (0, "Below Avg"): "blue",
#         (0, "Above Avg"): "purple",
#         (1, "Below Avg"): "orange",
#         (1, "Above Avg"): "red",
#     }

#     data["color"] = data.apply(
#         lambda row: color_map[(row["major"], row["leisure_category"])], axis=1
#     )

#     # Create figure and axes
#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Scatter plot with color coding for major and leisure
#     for (major, leisure_category), group in data.groupby(["major", "leisure_category"]):
#         label = f"Econ={major}, Leisure={leisure_category}"
#         ax.scatter(
#             group["gpa"],
#             group["salary"],
#             alpha=0.6,
#             label=label,
#             color=color_map[(major, leisure_category)],
#         )

#     # Plot regression lines
#     ax.plot(
#         gpa_range,
#         salary_major_0,
#         color="blue",
#         linewidth=2,
#         label="Econ=0 Regression Line",
#     )
#     ax.plot(
#         gpa_range,
#         salary_major_1,
#         color="orange",
#         linewidth=2,
#         label="Econ=1 Regression Line",
#     )

#     # Add labels, legend, and title
#     ax.set_xlabel("GPA", fontweight="bold")
#     ax.set_ylabel("Salary (Thousands USD)", fontweight="bold")
#     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
#     ax.set_title("Salary as a Function of GPA, Major, and Leisure", fontweight="bold")
#     ax.legend(loc="upper left", fontsize="small")
#     ax.grid(alpha=0.3)

#     return fig


# with c05:
#     st.pyplot(plot_salary_gpa_major_leisure(reg_data_leisure))


with c05:
    beta_est_const = model_true.params["Intercept"]
    beta_est_gpa = model_true.params["gpa"]
    beta_est_major = model_true.params["major"]
    beta_est_leis = model_true.params["leisure"]

    st.markdown(
        "Sample estimates:"
        + r"$\hat{salary}$"
        + rf"""= {beta_est_const:0f}+ {beta_est_gpa:.0f} x grade + {beta_est_major:.0f} x econ + {beta_est_leis:.0f} x leis"""
    )

    st.markdown(r"#### Case 1 - if only $grade$ is included")
    st.latex(
        r"""
        \begin{array}{l l}
            \text{Estimated DGP:} & salary_i = \alpha_0 + \alpha_{grade} grade_i + u_i \\
            \text{OVB:} & E[\hat{\alpha}_{grade}] - \beta_{grade} = \beta_{econ} \frac{cov(grade, econ)}{var(grade)} + \beta_{leis} \frac{cov(grade, leis)}{var(grade)} \\
        \end{array}
        """
    )

    model_ovb_1 = smf.ols(formula="salary ~ gpa", data=reg_data_leisure).fit()

    alpha_est_gpa = model_ovb_1.params["gpa"]

    st.markdown(
        "Sample estimate: " + r"$\hat{\alpha}_{grade}$" + f"""= {alpha_est_gpa:.0f}"""
    )

    model_gpa_major = smf.ols(formula="major ~ gpa", data=reg_data_leisure).fit()
    gamma_gpa_major = model_gpa_major.params["gpa"]
    model_gpa_leisure = smf.ols(formula="leisure ~ gpa", data=reg_data_leisure).fit()
    gamma_gpa_leis = model_gpa_leisure.params["gpa"]

    st.markdown(
        r"OVB: $\hat{\alpha}_{grade} - \hat{\beta}_{grade}$"
        + f" = {alpha_est_gpa-beta_est_gpa:.0f}"
    )
    st.markdown(
        r"Using OVB formula: "
        + r""" $\hat{\beta}_{econ} \frac{cov(grade, econ)}{var(grade)} + \hat{\beta}_{leis} \frac{cov(grade, leis)}{var(grade)} 
        = \hat{\beta}_{econ} \hat{\gamma}_{econ\_on\_grade} + \hat{\beta}_{leis} \hat{\gamma}_{leis\_on\_grade}$"""
        + f" = {beta_est_major * gamma_gpa_major:.0f} + {beta_est_leis * gamma_gpa_leis:.0f} = {beta_est_major * gamma_gpa_major + beta_est_leis * gamma_gpa_leis:.0f}.",
        unsafe_allow_html=True,
    )

    st.markdown(r"#### Case 2 - if $grade$ and $leisure$ are included")

    st.latex(
        r"""
        \begin{array}{l l}
            \text{Estimated DGP:} & salary_i = \alpha_0 + \alpha_{grade} grade_i + \alpha_{leis} leis_i + u_i \\
            \text{OVB on $grade$:} & E[\hat{\alpha}_{grade}] - \beta_{grade} = \beta_{econ} \frac{cov(grade, econ|leisure)}{var(grade|leisure)}\\
            \text{OVB on $leis$:} & E[\hat{\alpha}_{leis}] - \beta_{leis} = \beta_{econ} \frac{cov(leisure, econ|grade)}{var(leisure|grade)}\\
        \end{array}
        """
    )

    model_ovb_2 = smf.ols(formula="salary ~ gpa + leisure", data=reg_data_leisure).fit()

    alpha_est_gpa_2 = model_ovb_2.params["gpa"]
    alpha_est_leisure_2 = model_ovb_2.params["leisure"]

    st.markdown(
        "Sample estimates: "
        + r"$\hat{\alpha}_{grade}$"
        + f"= {alpha_est_gpa_2:.0f}; "
        + r" $\hat{\alpha}_{leis}$"
        + f"""= {alpha_est_leisure_2:.0f}
"""
    )

    st.markdown(
        r"OVB on grade: $\hat{\alpha}_{grade} - \hat{\beta}_{grade}$"
        + f" = {alpha_est_gpa_2-beta_est_gpa:.0f}."
    )

    st.markdown(
        r"OVB on leisure: $\hat{\alpha}_{leis} - \hat{\beta}_{leis}$"
        + f" = {alpha_est_leisure_2-beta_est_leis:.0f}."
    )

    st.markdown(r"**Understanding OVB formula with matrix algebra**")

    st.markdown(
        "Confirm the estimated bias by decomposing the OVB formula with matrix algebra:"
        + r"$E[\hat{\alpha}]-\beta=\beta_q (X'X)^{-1}X'q$, where $q$ is the omitted variable. In our example, $\beta_q$ is $\beta_{econ}$ and $X = [1 \; grade \; leisure]$"
    )

form_col1, form_col2, form_col3, form_col4, form_col5 = st.columns([1.5, 1, 1, 1, 1])

with form_col1:
    st.markdown(r"$[X'X]$")
    X = reg_data_leisure[["gpa", "leisure"]].to_numpy()
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    st.write(XtX)

with form_col2:
    st.markdown(r"$[X'X]^{-1}$")
    XtX_inv = np.linalg.inv(XtX)
    st.write(XtX_inv)

with form_col3:
    st.markdown(r"$X'q$")
    q = reg_data_leisure[["major"]].to_numpy()
    Xtq = np.dot(X.T, q)
    st.write(Xtq)

with form_col4:
    st.markdown(r"$(X'X)^{-1}X'q$")
    XtX_inv_Xtq = np.dot(XtX_inv, Xtq)
    st.write(XtX_inv_Xtq)

with form_col5:
    st.markdown(r"$\beta_{q}(X'X)^{-1}X'q$")
    bias = beta_est_major * XtX_inv_Xtq
    st.write(bias)

s1, c06, s2 = utl.wide_col()

with c06:
    model_major = smf.ols(formula="major ~ gpa + leisure", data=reg_data_leisure).fit()

    gamma_est_major_on_gpa = model_major.params["gpa"]
    gamma_est_major_on_leis = model_major.params["leisure"]

    st.markdown(
        r"What we did is identical to running the regression of omitted variable (econ) on included covariates (grade, leisure) and multiplying them by the $\beta$ estimate on omitted variable (econ) from the true DGP."
    )
    st.markdown(
        r"Sample estimates from econ ~ grade + leisure:  "
        + r"$\hat{\gamma}_{grade}$"
        + f"= {gamma_est_major_on_gpa:.3f}; "
        + r" $\hat{\gamma}_{leis}$"
        + f"= {gamma_est_major_on_leis:.3f}",
    )

    st.markdown(
        r"Multiply them both by $\hat{\beta}_{econ}=$"
        + f"{beta_est_major:.0f}"
        + r" to get the bias on $\hat{\alpha}_{grade}$ and $\hat{\alpha}_{leis}$"
    )
    st.markdown(
        r"OVB on $\hat{\alpha}_{grade}$: "
        + f"{beta_est_major*gamma_est_major_on_gpa:.0f}"
    )
    st.markdown(
        r"OVB on $\hat{\alpha}_{leis}$: "
        + f"{beta_est_major*gamma_est_major_on_leis:.0f}"
    )

    st.markdown(r"**Estimating OVB through partitioned regressions**")

    st.markdown(
        r"As another alternative, we can use FWL theorem for partitioned regressions to first net out the effects of one covariate, and estimate the OVB on the netted regression."
    )

    st.latex(
        r"""
        \begin{array}{l l}
            \text{True DGP:} & \tilde{salary}_i = \beta_0 + \beta_{grade} \tilde{grade}_i + \beta_{econ} \tilde{econ}_i + \varepsilon_i \\
            & \text{where $\tilde{z}_i$ is residuals from regressing variable $z$ on $leis$ alone} \\            
            \text{Estimated DGP (omit econ):} & \tilde{salary}_i = \alpha_0 + \alpha_{grade} \tilde{grade}_i + u_i \\
            \text{OVB on $grade$:} & E[\hat{\alpha}_{grade}] - \beta_{grade} = \beta_{econ} \frac{cov(\tilde{grade}, \tilde{econ})}{var(\tilde{grade})}\\
        \end{array}
        """
    )


# Step 1: Residualize salary, grade, and econ on leisure
model_salary_leis = smf.ols("salary ~ leisure", data=reg_data_leisure).fit()
tilde_salary = model_salary_leis.resid

model_grade_leis = smf.ols("gpa ~ leisure", data=reg_data_leisure).fit()
tilde_grade = model_grade_leis.resid

model_econ_leis = smf.ols("major ~ leisure", data=reg_data_leisure).fit()
tilde_econ = model_econ_leis.resid

# Step 2: Run the true netted reg of residualized salary on residualized grade and major
model_tilde_salary_true = smf.ols(
    "tilde_salary ~ tilde_grade + tilde_econ",
    data=pd.DataFrame(
        {
            "tilde_salary": tilde_salary,
            "tilde_grade": tilde_grade,
            "tilde_econ": tilde_econ,
        }
    ),
).fit()

beta_tilde_grade = model_tilde_salary_true.params["tilde_grade"]
beta_tilde_econ = model_tilde_salary_true.params["tilde_econ"]


# Step 3: Run the OVB netted reg of residualized salary on residualized grade
model_tilde_salary_ovb = smf.ols(
    "tilde_salary ~ tilde_grade",
    data=pd.DataFrame({"tilde_salary": tilde_salary, "tilde_grade": tilde_grade}),
).fit()

alpha_tilde_grade = model_tilde_salary_ovb.params["tilde_grade"]

# Step 3: Run reg of tilde_econ on tilde_grade
model_tilde_econ = smf.ols(
    "tilde_econ ~ tilde_grade",
    data=pd.DataFrame({"tilde_econ": tilde_econ, "tilde_grade": tilde_grade}),
).fit()

gamma_tilde_grade = model_tilde_econ.params["tilde_grade"]

# Step 4: Compute OVB using the formula
ovb_tilde_grade = beta_tilde_econ * gamma_tilde_grade

with c06:

    # Step 5: Display results
    st.markdown(
        r"Estimated $\hat{\beta}_{grade}$ from the true netted model is: "
        + f"{beta_tilde_grade:.0f}"
        + r", which is equal to $\hat{\beta}_{grade}$ without netting out leisure (confirms that FWL theorem works)."
    )

    st.markdown(
        r"Estimated $\hat{\alpha}_{grade}$:"
        + f"{alpha_tilde_grade:.0f}, which again confirms that FWL theorem works."
    )
    st.markdown(
        r"Estimated OVB on grade: $\hat{\alpha}_{grade} - \hat{\beta}_{grade} = $"
        + f"{alpha_tilde_grade - beta_tilde_grade:.0f}"
    )
    st.markdown(
        r"OVB on grade calculated using OVB formula: $\beta_{econ} \frac{cov(\tilde{grade}, \tilde{econ})}{var(\tilde{grade})}=$"
        + f"{beta_tilde_econ:.0f} x {gamma_tilde_grade:.3f} = {beta_tilde_econ * gamma_tilde_grade:.0f}"
    )

    st.markdown(
        r"""As expected, **all the methods give the same result**, but understanding each of them is useful
        to get better intuition behind the OVB (e.g., FWL shows clearly how a strong correlation between leisure and major can reduce the OVB on the grade coefficient,
        while regressing OV on covariates shows how the bias propagates to all coefficients even if only one is correlated with the OV)."""
    )
