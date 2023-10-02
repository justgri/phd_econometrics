import os
import random
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from matplotlib import pyplot as plt
from scipy.stats import t
from st_pages import add_page_title

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

### PAGE CONFIGS ###

st.set_page_config(layout="wide")
utl.local_css("src/styles/styles_pages.css")

random_seed = 0

### Data viz part
# Generate and display fixed data (data points, reg line, confidence interval)
# Show table with coefficients with their standard errors
# Then let user to choose underlying parameters for a, b, variance of errors and N
# Plot on the same chart and add coefficients to the table for comparison
# Bonus: show how coef SE depends on N and var(e) graphically

### Theory part
# show matrix notation, verify that sklearn gives same as multiplying by hand
# some experiments with projection matrices (P, M)?

# create one column with consistent width
s1, c01, s2 = utl.wide_col()

### PAGE START ###
# Dashboard header

with c01:
    st.title("Week 1 - Ordinary Least Squares Estimation")
    st.header("1. OLS Visually")

    st.write(
        "Play around with sliders to see how the data and estimates change."
    )

    st.markdown(
        "<h3 style='text-align: left'> Visualizing how OLS estimates depend on true population parameters</h3>",
        unsafe_allow_html=True,
    )

    st.write(
        r"Suppose you have the following true population relationship between $X$ and $Y$, with parameters defined by slider values."
    )
    st.write(
        r"You then draw a sample of size $N$ from that population and estimate OLS coefficients, $\hat{\beta_0}$ and $\hat{\beta_1}$."
    )


def gen_lin_data(b0, b1, sd, N, rseed):
    np.random.seed(rseed)
    # generate x
    x = np.round(np.random.uniform(-10, 10, N), 1)
    # add constant
    x = np.column_stack((np.ones(N), x))
    # generate  error term
    e = np.random.normal(0, sd, N)

    # y = xB + e
    y = np.dot(x, np.array([b0, b1])) + e

    # fit reg
    model = sm.OLS(y, x).fit()

    # get fitted values and CI
    predictions = model.get_prediction(x)
    y_hat = predictions.predicted_mean
    y_hat_se = predictions.se_mean

    # get confidence interval
    ci = predictions.conf_int(alpha=0.05)  # 95% CI
    deg_freedom = x.shape[0] - x.shape[1]  # N - k
    # t_score = t.ppf(0.975, deg_freedom)
    # ci = np.column_stack(
    #     (y_hat - t_score * y_hat_se, y_hat + t_score * y_hat_se)
    # )

    # get error parameters
    e_hat = y - y_hat
    s = np.sqrt(np.sum(e_hat**2) / deg_freedom)

    # calculate R^2 manually
    y_bar = np.mean(y)
    ss_tot = np.sum((y - y_bar) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {
        "y": y,
        "x": x,
        "e": e,
        "e_hat": e_hat,
        "y_hat": y_hat,
        "s": s,
        "ci": ci,
        "model": model,
        "r_squared": r_squared,
    }


with c01:
    st.latex(
        r"""
            Y_i = \beta_0 + \beta_1X_i + \varepsilon_i \text{, where }  \varepsilon \sim N(0, \sigma^2)
        """
    )
    st.latex(r"""\hat{Y_i} = """ + r"""\hat{\beta_0} + \hat{\beta_1}X""")


s1, slider_col, s2 = st.columns(3)

with slider_col:
    # Sliders
    b0_cust = st.slider(
        r"Intercept, $\beta_0$",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
    )
    b1_cust = st.slider(
        r"Slope, $\beta_1$", min_value=-5.0, max_value=5.0, value=0.0, step=0.1
    )
    var_cust = st.slider(
        r"Error SD, $\sqrt{var(\varepsilon)} = \sigma$",
        min_value=0.1,
        max_value=20.0,
        value=10.0,
        step=0.1,
    )

    n_cust = st.slider(
        "Sample size, $N$",
        min_value=10,
        max_value=1000,
        value=500,
        step=10,
    )

custom_data = gen_lin_data(b0_cust, b1_cust, var_cust, n_cust, random_seed)


def plot_ols(data_custom, b0, b1):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0)  # remove margin

    # Sample data
    ax.scatter(
        data_custom["x"][:, 1],
        data_custom["y"],
        # label="Custom Data",
        color=thm.cols_set1_plt[1],
        alpha=0.5,
    )

    include_pop = False
    if include_pop:
        # True pop line
        x_values = np.linspace(-10, 10, 100)
        y_values = b0 + b1 * x_values
        if b1 >= 0:
            label = rf"$\bar{{y}} = {b0:.2f} + {b1:.2f}x$"
        else:
            label = rf"$\hat{{y}} = {b0:.2f} - {-b1:.2f}x$"

        ax.plot(
            x_values,
            y_values,
            label=label,
            color=thm.cols_set1_plt[1],
        )

    # Sample line
    b0_s = data_custom["model"].params[0]
    b1_s = data_custom["model"].params[1]

    if b1_s >= 0:
        label_s = rf"$\hat{{y}} = {b0_s:.2f} + {b1_s:.2f}x$"
    else:
        label_s = rf"$\hat{{y}} = {b0_s:.2f} - {-b1_s:.2f}x$"

    ax.plot(
        data_custom["x"][:, 1],
        data_custom["y_hat"],
        label=label_s,
        color=thm.cols_set1_plt[4],
    )

    sorted_indices = np.argsort(data_custom["x"][:, 1])
    sorted_x = data_custom["x"][:, 1][sorted_indices]
    sorted_ci_lower = data_custom["ci"][:, 0][sorted_indices]
    sorted_ci_upper = data_custom["ci"][:, 1][sorted_indices]

    ax.fill_between(
        sorted_x,
        sorted_ci_lower,
        sorted_ci_upper,
        color=thm.cols_set1_plt[4],
        alpha=0.3,
        label="95% Confidence Interval",
    )

    plt.xlim([-11, 11])
    plt.ylim([-50, 50])
    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.legend(loc="upper left", fontsize="small")

    return fig


def create_summary(data):
    coefficients = pd.DataFrame(
        {
            "Coefficient": [r"Intercept b_0", "Slope b_1"],
            "Population": [b0_cust, b1_cust],
            "Sample": [
                data["model"].params[0],
                data["model"].params[1],
            ],
            "Sample SE": [
                data["model"].bse[0],
                data["model"].bse[1],
            ],
        }
    )

    # Apply formatting to the "True Pop" and "Estimate" columns
    coefficients[["Population", "Sample", "Sample SE"]] = coefficients[
        ["Population", "Sample", "Sample SE"]
    ].applymap(lambda x: f"{x:.2f}")

    return coefficients


with slider_col:
    if st.button("Resample data", type="primary"):
        random_seed = random.randint(0, 10000)
        custom_data = gen_lin_data(
            b0_cust, b1_cust, var_cust, n_cust, random_seed
        )


coefficients = create_summary(custom_data)

# CSS styles for the table (center and header)
table_styler = [
    {
        "selector": "th",  # Apply to header cells
        "props": [("background-color", "lightblue")],
    },
    {
        "selector": "td",  # Apply to data cells
        "props": [
            ("font-size", "20px"),
            ("text-align", "center"),
            ("background-color", "white"),
        ],
    },
]

# Apply CSS styles to the DataFrame
styled_coefficients = coefficients.style.set_table_styles(table_styler)

# Create a centered and styled HTML representation of the DataFrame
styled_table = styled_coefficients.hide(axis="index").to_html()

# Define custom CSS to style the table and center it
table_css = """
<style>
    table {
        margin: 0 auto;
        text-align: center;
    }
</style>
"""

s0, c02, s1 = utl.narrow_col()

with c02:
    # display table and plot
    st.write(
        f"{table_css}{styled_table}",
        unsafe_allow_html=True,
    )
    st.write("")

    st.latex(
        f"N= {n_cust} ,"
        + f"R^2 = {custom_data['model'].rsquared:.2f}"
        + r", s = \sqrt{\frac{\mathbf{e'e}}{N - k}}"
        + f"= {custom_data['s']:.2f}"
    )

    _, col_plot, _ = utl.narrow_col()
    col_plot.pyplot(
        plot_ols(custom_data, b0_cust, b1_cust), use_container_width=True
    )


s0, c03, s1 = utl.wide_col()

with c03:
    st.markdown(
        "<h3 style='text-align: left'> Interesting takeaways</h3>",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
    1. $R^2 = 0$ in expectation if $\beta_1=0$ or if $X_i = \bar{X}$. Also $R^2$ is independent of the intercept.<br>

        $R^2 = \frac{ (\hat{y} - \bar{y})' (\hat{y} - \bar{y}) }{ (y - \bar{y})' (y - \bar{y}) }
        = \frac{\sum_{i=1}^{N} (\hat{y}_i - \bar{y})^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
        = \frac{\sum_{i=1}^{N} (\hat{\beta_1} (X_i - \bar{X}))^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$
        , because $\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{X}$
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
                    
    2. Variance of $\hat{\beta}$ (and thus their standard errors) does not depend on population $\beta$. <br>
    It depends on variance of errors $s^2$ (and thus $\sigma^2)$, $N-k$. and $X'X$.<br>
    Note that higher variance of $X$ leads to a lower variance of $\hat{\beta}$, which is intuitive because you cover a wider range of $X$s.<br>
    
    $\widehat{var(\mathbf{b}| \mathbf{X})} \equiv s^{2} \cdot (X'X)^{-1} = \frac{\mathbf{e'e}}{N - k} \dot (X'X)^{-1}$
""",
        unsafe_allow_html=True,
    )


with c03:
    st.header("2. OLS in matrix notation")
    st.write("Check out Matteo Courthoud's website for summary:")
    st.link_button(
        "OLS Algebra",
        "https://matteocourthoud.github.io/course/metrics/05_ols_algebra/",
        type="primary",
    )

    st.header("3. OLS assumptions")
    st.write("Greene Chapter 4, Table 4.1:")
    st.markdown(
        r"""
        **A1. Linearity:** $\mathbf{y = X \beta + \varepsilon}$ <br>
        **A2. Full rank:** $\mathbf{X}$ is an $ n \times K$ matrix with rank $K$ (full column rank) <br>
        **A3. Exogeneity of the independent variables:** $E[\varepsilon_i | \mathbf{X}] = 0$ <br>
        **A4. Homoscedasticity and nonautocrrelation:** $E[\mathbf{\varepsilon \varepsilon'} | \mathbf{X}] = \sigma^2 \mathbf{I}_n$ <br>
        **A5. Stochastic or nonstochastic data:** $\mathbf{X}$ may be fixed or random.<br>
        **A6. Normal distribution:** $\varepsilon | \mathbf{X} \sim N(0, \sigma^2 \mathbf{I}_n)$ <br>
        """,
        unsafe_allow_html=True,
    )

    st.header("4. Proofs to remember")
    st.write("TBD")
