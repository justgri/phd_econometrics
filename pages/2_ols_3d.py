import os
import random
import sys

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
import streamlit as st

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
input_col_x, input_col_sigma, input_col_n, input_col_0 = st.columns(4)
# Input beta 1 and 2
input_col_1, input_col_2 = st.columns((0.5, 1))

# Chart columns
chart1_col, chart2_col = st.columns(2)
# Resample data col
resample_col, _ = st.columns(2)
# Table and chart columns
_, table1_col, _, table2_col, _ = st.columns((0.25, 1, 0.01, 0.5, 0.25))

### Data viz part

### Theory part


### PAGE START ###
# Dashboard header

with c01:
    st.title("OLS in 3-dimensions")
    st.divider()
    st.header("1. Visualizing OLS in 3D")

    st.markdown(
        r"""Suppose the true population relationship between $X$ and $y$ is defined by the slider values below.
        You then draw a sample of size $n$ from that population and estimate OLS coefficients: $b_0$, $b_1$, and $b_2$.<br>
        In the population, $x_{k, i}$'s are i.i.d. with $x_k$ $\sim U(-5, 5)$ for $k=1,2$. Errors are homoscedastic, $\varepsilon \sim N(0, \sigma^2)$.
        """,
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        y_i = \beta_0 + \beta_1x_{1, i} + \beta_2x_{2, i} + \varepsilon_i \\
        \hat{y_i} = b_0 + b_1 x_{1, i} + b_2 x_{2, i} \\
        """
    )
    st.write("")

    st.markdown(
        r"""Given this model, we can plot 4 elements in both 2D and 3D graphs:""",
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        <span style="color:blue">Blue dots</span> are observed sample data points, $(y, X)$.<br>
        <span style="color:green">Green dots</span> are data points with predicted $y$ values, $(\hat{y}, X)$.<br>
        <span style="color:orange">Orange line</span> represents predicted $y$ values for the full range of $x_1$, holding $x_2$ constant at $x_2=0$.<br>
        <span style="color:purple">Purple line</span> represents predicted $y$ values for the full range of $x_2$, holding $x_1$ constant at $x_1=0$.<br>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"Move the sliders to see how the plots and sample estimates change by varying the population parameters.",
        unsafe_allow_html=True,
    )


def gen_reg_data(b0, b1, b2, sd, N, rseed):
    np.random.seed(rseed)

    beta = np.array([b0, b1, b2])
    # generate X1, X2, and error - max(N-slider) number of samples
    # keep X vectors separate for plotting
    X1 = np.random.uniform(-5, 5, 1000)
    X2 = np.random.uniform(-5, 5, 1000)
    eps = np.random.normal(0, sd, 1000)

    # take first N samples to show
    X1 = X1[:N]
    X2 = X2[:N]
    eps = eps[:N]

    X = np.column_stack((X1, X2))
    X = sm.add_constant(X)

    # Generate y = b0 + b1*X1 + b2*X2 + eps
    y = np.dot(X, beta) + eps

    # OLS regression with both X1 and X2
    model = sm.OLS(y, X).fit()
    y_hat = model.predict(X)
    # confidence intervals
    ci = model.get_prediction(X).conf_int(alpha=0.05)
    # true standard error
    s = model.mse_resid**0.5

    return {
        "y": y,
        "y_hat": y_hat,
        "beta": beta,
        "ci": ci,
        "s": s,
        "X": X,
        "model": model,
    }


if "reg_data" not in st.session_state:
    st.session_state.reg_data = gen_reg_data(
        0.0, -3.0, 3.0, 15.0, 200, st.session_state.random_seed
    )


with input_col_x:
    plot_x = st.radio(
        "**X perspective to plot**",
        ("*x1*", "*x2*"),
        horizontal=True,
    )
    plot_x = plot_x[1:-1]

# Intercept, β0
b0_cust = input_col_0.number_input(
    r"$\beta_0$",
    min_value=-10.0,
    max_value=10.0,
    value=0.0,
    step=1.0,
)

# Error SD, √var(ϵ)=σ
var_cust = input_col_sigma.number_input(
    r"$\sigma$",
    min_value=0.1,
    max_value=50.0,
    value=25.0,
    step=5.0,
)

# Sample size, n
n_cust = input_col_n.number_input(
    "$n$",
    min_value=10,
    max_value=1000,
    value=50,
    step=50,
)


# Slope for x1, β1
b1_cust = input_col_1.number_input(
    r"$\beta_1$",
    min_value=-10.0,
    max_value=10.0,
    value=-5.0,
    step=1.0,
)

# Slope for x2, β2
b2_cust = input_col_2.slider(
    r"$\beta_2$",
    min_value=-10.0,
    max_value=10.0,
    value=5.0,
    step=1.0,
)

st.session_state.reg_data = gen_reg_data(
    b0_cust,
    b1_cust,
    b2_cust,
    var_cust,
    n_cust,
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

    indices = list(range(len(data_custom["y"])))
    # Fitted line for one x
    x_range = np.linspace(-5, 5, 10)
    if show == "x1":
        y_from_one_x = b0_ols + b1_ols * x_range
        color_one_x = thm.cols_set1_px["orange"]
    elif show == "x2":
        y_from_one_x = b0_ols + b2_ols * x_range
        color_one_x = thm.cols_set1_px["purple"]

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
            name=f"ŷ = {b0_ols:.2f} + {b1_ols if show=='x1' else b2_ols:.2f}{name_fitted_2d}",
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
            range=[-100, 100],
            autorange=False,
            title="",  # Empty the title here as we'll add it as an annotation
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
            dict(
                xref="paper",
                yref="paper",
                x=-0.07,
                y=1,
                text="y",
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


with resample_col:
    if st.button("Resample data", type="primary"):
        st.session_state.random_seed = random.randint(0, 10000)

        st.session_state.reg_data = gen_reg_data(
            b0_cust,
            b1_cust,
            b2_cust,
            var_cust,
            n_cust,
            st.session_state.random_seed,
        )

with chart1_col:
    fig_3d = create_3d_plot(st.session_state.reg_data, view_var=plot_x)

    st.plotly_chart(fig_3d, theme=None, use_container_width=True)


with chart2_col:
    fig_2d = plot_ols_plotly(
        st.session_state.reg_data,
        [b0_cust, b1_cust, b2_cust],
        show=plot_x,
    )
    st.plotly_chart(
        fig_2d,
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


with table1_col:
    coef_table = gen_coef_table(
        [b0_cust, b1_cust, b2_cust], st.session_state.reg_data["model"]
    )

    st.markdown(coef_table, unsafe_allow_html=True)

with table2_col:
    stats_table = gen_stats_table(st.session_state.reg_data["model"])
    st.markdown(stats_table, unsafe_allow_html=True)

s0, c03, s1 = utl.wide_col()

with c03:
    st.markdown("### Interesting takeaways")

    with st.expander("Click to expand.", expanded=True):
        st.markdown(
            r"""
    
1. Looking from $x_1$ perspective and varying $\beta_2$ will lead to the changes in the vertical spread of green dots.<br>
    The larger $\beta_2$ is in absolute value, the wider the spread of green dots.
    Conversely, when $\beta_2=0$, the green dots are close to the orange line.<br>
    Try holding $\beta_2$ at 0 and varying $\sigma$ or $\beta_1$ - the green dots will stay close to the orange line.<br>

2. Higher $R^2$ means that the green dots are closer to the blue dots, i.e., fitted values are closer to the true sample values.
It is misleading to think that $R^2$ should be higher when green dots are closer to the orange line when looking from $x_1$ perspective.
$R^2$ coincidentally happens to be higher when green dots are closer to the orange line only when the blue dots are also close to the orange line, which depends on $\sigma$.<br>
    Try varying $\sigma$ and $\beta_2$ to see how the two affect $R^2$ separately.
            
    """,
            unsafe_allow_html=True,
        )

with c03:
    st.header("2. OLS theory")
    st.markdown(
        r"""All theory is described on the previous page of this app.<br>
        This visualization illustrates an example of OLS with homoscedastic errors (see **A4** below) and independent $x$'s.<br>
        In the future chapters, we will look how the plots and estimates change when the errors are heteroscedastic or when $x$'s are correlated.""",
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""Reminder of the OLS assumptions (from Greene Chapter 4, Table 4.1):<br>
        **A1. Linearity:** $\mathbf{y = X \beta + \varepsilon}$ <br>
        **A2. Full rank:** $\mathbf{X}$ is an $ n \times K$ matrix with rank $K$ (full column rank) <br>
        **A3. Exogeneity of the independent variables:** $E[\varepsilon_i | \mathbf{X}] = 0$ <br>
        **A4. Homoscedasticity and nonautocrrelation:** $E[\mathbf{\varepsilon \varepsilon'} | \mathbf{X}] = \sigma^2 \mathbf{I}_n$ <br>
        **A5. Stochastic or nonstochastic data:** $\mathbf{X}$ may be fixed or random.<br>
        **A6. Normal distribution:** $\varepsilon | \mathbf{X} \sim N(0, \sigma^2 \mathbf{I}_n)$ <br>
        """,
        unsafe_allow_html=True,
    )
