import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from st_pages import add_page_title

import src.scripts.plotly_themes
import src.scripts.utils as utl

### PAGE CONFIGS ###

st.set_page_config(layout="wide")
utl.local_css("src/styles/styles_pages.css")

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
        "<h3 style='text-align: left'> Visualizing how OLS estimates depend on true population parameters.</h3>",
        unsafe_allow_html=True,
    )

    st.write(
        r"Suppose you have the following true population relationship between $X$ and $Y$, with parameters defined by slider values."
    )
    st.write(
        r"You then draw a sample of size $N$ from that population and estimate OLS coefficients, $\hat{\beta_0}$ and $\hat{\beta_1}$."
    )


def gen_lin_data(b0, b1, var, N):
    # np.random.seed(0)
    # generate x
    x = np.round(np.random.uniform(-10, 10, N), 1)
    # add constant
    x = np.column_stack((np.ones(N), x))
    # generate  error term
    e = np.random.normal(0, var, N)

    # y = xB + e
    y = np.dot(x, np.array([b0, b1])) + e

    # fit reg
    model = sm.OLS(y, x).fit()

    y_hat = model.predict(x)

    return {
        "y": y,
        "x": x,
        "e": e,
        "y_hat": y_hat,
        "model": model,
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
        r"Error variance, $var(\varepsilon) = \sigma^2$",
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

custom_data = gen_lin_data(b0_cust, b1_cust, var_cust, n_cust)


def plot_ols(data_custom):
    fig, ax = plt.subplots()
    fig.set_size_inches(3.5, 3.5)
    plt.subplots_adjust(left=0)  # remove margin

    # Custom data
    ax.scatter(
        data_custom["x"][:, 1],
        data_custom["y"],
        # label="Custom Data",
        color="blue",
        alpha=0.5,
    )

    ax.plot(
        data_custom["x"][:, 1],
        data_custom["y_hat"],
        label=f"y = {data_custom['model'].params[0]:.2f} + {data_custom['model'].params[1]:.2f}x",
        color="red",
    )

    plt.xlim([-11, 11])
    plt.ylim([-50, 50])
    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.legend(loc="upper left")

    return fig


def create_summary(data):
    coefficients = pd.DataFrame(
        {
            "Coefficient": [r"Intercept B_0", "Slope B_1"],
            "Population": [b0_cust, b1_cust],
            "Sample": [
                data["model"].params[0],
                data["model"].params[1],
            ],
            "Sample SE": [
                data["model"].bse[0],
                data["model"].bse[1],
            ],
            "N": [n_cust, ""],
            "R-sq": [f"{data['model'].rsquared:.2f}", ""],
        }
    )

    # Apply formatting to the "True Pop" and "Estimate" columns
    coefficients[["Population", "Sample", "Sample SE"]] = coefficients[
        ["Population", "Sample", "Sample SE"]
    ].applymap(lambda x: f"{x:.2f}")

    return coefficients


with slider_col:
    if st.button("Resample data"):
        custom_data = gen_lin_data(b0_cust, b1_cust, var_cust, n_cust)


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
            ("font-size", "14px"),
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

    st.pyplot(plot_ols(custom_data))

s0, c03, s1 = utl.wide_col()


with c03:
    st.header("2. Matrix notation for OLS")
    st.write("Check out Matteo Courthoud's website for more details:")
    st.link_button(
        "OLS Algebra",
        "https://matteocourthoud.github.io/course/metrics/05_ols_algebra/",
        type="primary",
    )
