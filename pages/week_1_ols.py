import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from st_pages import Page, add_page_title, show_pages

import plotly_themes as thm

# Dashboard header
add_page_title()

st.header("1. Ordinary Least Squares")
st.write(
    "Visualizing how OLS coefficient standard errors depend on the underlying true data."
)

# Pseudo-code
# Generate and display fixed data (data points, reg line, confidence interval)
# Show table with coefficients with their standard errors
# Then let user to choose underlying parameters for a, b, variance of errors and N
# Plot on the same chart and add coefficients to the table for comparison
# Bonus: show how coef SE depends on N and var(e) analytically


def gen_lin_data(N, a, b, var):
    np.random.seed(0)
    x = np.round(np.random.uniform(-10, 10, N), 1)
    e = np.random.normal(0, var, N)
    y = a + b * x + e

    # Fit a regression line for the variable data
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))

    return {"y": y, "x": x, "e": e, "y_pred": y_pred, "model": model}


fixed_data = gen_lin_data(100, 0, 1, 5)

# # Slider for 'a' and 'b' values
a_cust = st.slider(
    "Intercept B0", min_value=-5.0, max_value=5.0, value=0.0, step=0.1
)
b_cust = st.slider(
    "Slope B1", min_value=-5.0, max_value=5.0, value=0.0, step=0.1
)
var_cust = st.slider(
    "Error Variance", min_value=0.1, max_value=20.0, value=10.0, step=0.1
)

n_cust = st.slider(
    "Number of Samples", min_value=10, max_value=1000, value=500, step=10
)

custom_data = gen_lin_data(n_cust, a_cust, b_cust, var_cust)


def plot_ols(data, data_custom):
    fig, ax = plt.subplots()
    # Fixed data
    ax.scatter(
        data["x"],
        data["y"],
        # label="Initial Data",
        color="blue",
        alpha=0.5,
    )

    ax.plot(
        data["x"],
        data["y_pred"],
        label=f"Reference OLS: y = {data['model'].intercept_:.2f} + {data['model'].coef_[0]:.2f}x",
        color="blue",
    )

    # Custom data
    ax.scatter(
        data_custom["x"],
        data_custom["y"],
        # label="Custom Data",
        color="green",
        alpha=0.5,
    )

    ax.plot(
        data_custom["x"],
        data_custom["y_pred"],
        label=f"New OLS: y = {data_custom['model'].intercept_:.2f} + {data_custom['model'].coef_[0]:.2f}x",
        color="green",
    )

    plt.xlim([-11, 11])
    plt.ylim([-50, 50])

    plt.legend()

    return fig


# Display the plot in Streamlit
st.pyplot(plot_ols(fixed_data, custom_data))

# Now add a customly generated data


# # Fixed data points and their corresponding regression line
# x_fixed = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8])
# y_fixed = 0.5 + 1.5 * x_fixed  # Example fixed data points
# y_fixed_regression = 0.5 + 1.5 * x  # Fixed regression line for comparison


# # Plot the fixed data points and their fixed regression line
# ax.scatter(
#     x_fixed,
#     y_fixed,
#     label="Data Points (Fixed Line)",
#     color="green",
#     marker="o",
#     s=80,
# )
# ax.plot(
#     x,
#     y_fixed_regression,
#     label=f"Regression Line (Fixed Line): y = 0.5 + 1.5x",
#     linestyle="--",
#     color="green",
# )

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.legend()

# table HTML

# HTML code for a table with a spanned row
table_html = """
<table>
  <tr>
    <td rowspan="2">Spanned Text</td>
    <td>Row 1, Column 2</td>
    <td>Row 1, Column 3</td>
  </tr>
  <tr>
    <td>Row 2, Column 2</td>
    <td>Row 2, Column 3</td>
  </tr>
</table>
"""

# Display the HTML table using st.markdown()
st.markdown(table_html, unsafe_allow_html=True)


# Add table with coefficients

coefficients = pd.DataFrame(
    {
        "Coefficient": ["a", "b", "c", "d"],
        "Initial Value": [1, 2, 3, 4],
        "Regression Coefficient": [3, 4, 5, 6],
        "Data Points (N)": [5, 6, 7, 8],
    }
)

# Apply CSS styles to the DataFrame for improved table appearance
styles = [
    {
        "selector": "td",
        "props": [("font-size", "14px"), ("text-align", "center")],
    },
    {
        "selector": "th",
        "props": [("font-size", "16px"), ("text-align", "center")],
    },
    {
        "selector": "tr:nth-child(odd)",
        "props": [("background-color", "grey")],
    },
    {
        "selector": "tr:nth-child(even)",
        "props": [("background-color", "white")],
    },
    {
        "selector": "tr:first-child",
        "props": [("background-color", "lightgrey")],
    },
]

st.write(
    """
<style>
td, th {
    padding: 8px;
    font-family: Arial, sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.table(coefficients.style.set_table_styles(styles))
