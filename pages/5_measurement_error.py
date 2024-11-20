import os
import random
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from matplotlib import pyplot as plt
from scipy.stats import t

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

utl.local_css("src/styles/styles_pages.css")

random_seed = 0

# create one column with consistent width
s1, c01, s2 = utl.wide_col()


random_seed = 0

if "random_seed" not in st.session_state:
    st.session_state.random_seed = 0

# DASHBOARD COLUMNS
# Initial text column
s1, c01, s2 = utl.wide_col()
# Input cols
input_col_b1, input_col_s_eps, input_col_s_eta, input_col_n = st.columns(4)

s1, c02, s2 = utl.wide_col()

# Chart columns
chart1_col, chart2_col = st.columns(2)

# Resample data col
resample_col, _ = st.columns(2)
# chart3_col, chart4_col = st.columns(2)

### PAGE START ###
# Dashboard header

with c01:
    st.title("Measurement Error in Variables")
    st.divider()
    st.header("1. Error in Dependent Variable")

    st.markdown(
        r"""Assume a linear relationship between $y$ and $x$, satisfying standard OLS assumptions (in particular, $E[\varepsilon|x]=0$).
        However, instead of observing the exact $y$'s, you only have data on $y$'s measured with some white noise $\eta \sim N(0, \sigma_\eta^2)$:
        """,
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        \begin{array}{l r}
            \text{True DGP:} & y_i = \beta_0 + \beta_1 x_i+ \varepsilon_i \\
            \text{Observed y: } & \tilde{y_i} = y_i + \eta_i\\  
            \text{Regression: } & \tilde{y_i} = \beta_0 + \beta_1 x_i + \tilde{\varepsilon_i} \\
        \end{array}
        """
    )

    st.markdown(
        r"""Our goal is to check whether $\hat{\beta}$ coefficients are unbiased estimates for $\beta$ in the presence of measurement error in $y$.
        In addition, we will see what is the effect of measurement error on the precision of coefficients, comparing $var(\hat{\beta})$ in case when it would have been estimated with and without
        the measurement error.<br>
        Note that $\tilde{\varepsilon}$ can be expressed as $\tilde{\varepsilon} = \varepsilon + \eta$. Also, note that $\beta_0 = 0$ and $\varepsilon \sim N(0, \sigma_\varepsilon^2)$.<br>
        Vary the parameters above to see the effects of measurement error.""",
        unsafe_allow_html=True,
    )


def gen_reg_data(b0, b1, sd_eps, sd_eta, N, rseed):
    np.random.seed(rseed)

    # Generate X as a single independent variable
    X = np.random.uniform(-5, 5, 1000)
    eps = np.random.normal(0, sd_eps, 1000)  # Generate random noise

    # Take the first N samples
    X = X[:N]
    eps = eps[:N]

    # Generate the true y (without measurement error)
    y = b0 + b1 * X + eps

    # Introduce measurement error in y to create observed y_tilde
    eta = np.random.normal(0, sd_eta, N)  # Measurement error
    y_tilde = y + eta  # Observed y with measurement error

    # Add a constant to X for regression purposes
    X = sm.add_constant(X)

    # OLS regression with observed y_tilde
    model_with_error = sm.OLS(y_tilde, X).fit()
    y_hat_with_error = model_with_error.predict(X)

    predictions_with_me = model_with_error.get_prediction(X)
    ci_me = predictions_with_me.conf_int(alpha=0.05)

    # OLS regression with true y (no measurement error) for comparison
    model_true = sm.OLS(y, X).fit()
    y_hat_true = model_true.predict(X)

    return {
        "y_true": y,
        "y_tilde": y_tilde,
        "y_hat_with_error": y_hat_with_error,
        "y_hat_true": y_hat_true,
        "ci_me": ci_me,
        "beta": np.array([b0, b1]),
        "model_with_error": model_with_error,
        "model_true": model_true,
    }


if "reg_data" not in st.session_state:
    st.session_state.reg_data = gen_reg_data(
        0.0, 5.0, 15.0, 15.0, 200, st.session_state.random_seed
    )

b0_cust = 0

# Slope for x1, β1
b1_cust = input_col_b1.number_input(
    r"$\beta_1$",
    min_value=-10.0,
    max_value=10.0,
    value=5.0,
    step=1.0,
)

sd_eps_cust = input_col_s_eps.number_input(
    r"$\sigma_\varepsilon$",
    min_value=1.0,
    max_value=25.0,
    value=10.0,
    step=1.0,
)

sd_eta_cust = input_col_s_eta.number_input(
    r"$\sigma_\eta$",
    min_value=1.0,
    max_value=25.0,
    value=10.0,
    step=1.0,
)


n_cust = input_col_n.number_input(
    r"$n$",
    min_value=50,
    max_value=1000,
    value=100,
    step=50,
)

st.session_state.reg_data = gen_reg_data(
    b0_cust,
    b1_cust,
    sd_eps_cust,
    sd_eta_cust,
    n_cust,
    st.session_state.random_seed,
)


def plot_ols(data):
    fig, ax = plt.subplots(figsize=(8, 6))

    b0_true, b1_true = data["model_true"].params
    b0_obs, b1_obs = data["model_with_error"].params

    # Scatter plot for true y values (green)
    ax.scatter(
        data["model_true"].model.exog[:, 1],  # X values without the constant term
        data["y_true"],
        color=thm.set1_green,
        alpha=0.5,
        label=r"True $y$",
    )

    # Scatter plot for observed y_tilde values with measurement error (blue)
    ax.scatter(
        data["model_with_error"].model.exog[:, 1],  # X values without the constant term
        data["y_tilde"],
        color=thm.set1_blue,
        alpha=0.5,
        label=r"Observed $\tilde{y}$",
    )

    # Fit line for true model (green)
    sorted_indices = np.argsort(data["model_true"].model.exog[:, 1])
    sorted_x = data["model_true"].model.exog[:, 1][sorted_indices]
    sorted_y_hat_true = data["y_hat_true"][sorted_indices]

    ax.plot(
        sorted_x,
        sorted_y_hat_true,
        color=thm.set1_green,
        linewidth=2,
        label=f"True fit: $\hat{{y}} = {b0_true:.2f} + {b1_true:.2f}x$",
    )

    # Fit line for observed model with measurement error (blue)
    sorted_y_hat_with_error = data["y_hat_with_error"][sorted_indices]
    sorted_ci_lower = data["ci_me"][:, 0][sorted_indices]
    sorted_ci_upper = data["ci_me"][:, 1][sorted_indices]

    ax.plot(
        sorted_x,
        sorted_y_hat_with_error,
        color=thm.set1_blue,
        linewidth=2,
        linestyle="--",
        label=f"Observed fit: "
        + r"$\hat{\tilde{y}}$"
        + f"= {b0_obs:.2f} + {b1_obs:.2f}x",
    )

    ax.fill_between(
        sorted_x,
        sorted_ci_lower,
        sorted_ci_upper,
        color=thm.set1_blue,
        alpha=0.3,
        label="95% Confidence Interval",
    )

    # Labels and legend
    plt.xlim([-5.5, 5.5])
    plt.ylim([-50, 50])
    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.legend(loc="upper left", fontsize="small")

    return fig


with resample_col:
    if st.button("Resample data", type="primary"):
        st.session_state.random_seed = random.randint(0, 10000)
        st.session_state.reg_data = gen_reg_data(
            b0_cust,
            b1_cust,
            sd_eps_cust,
            sd_eta_cust,
            n_cust,
            st.session_state.random_seed,
        )

with chart1_col:
    st.markdown(r"#### Regression with $y$ and $\tilde{y}$")
    chart1_col.pyplot(plot_ols(st.session_state.reg_data), use_container_width=True)


# Monte Carlo analysis function for Streamlit
def monte_carlo_betas(b0, b1, sd_eps, sd_eta, N, rseed, num_simulations=100):
    true_betas = []
    observed_betas = []

    for i in range(num_simulations):
        # Generate data and get the slope estimates
        gen_data = gen_reg_data(b0, b1, sd_eps, sd_eta, N, rseed + i)
        true_betas.append(gen_data["model_true"].params[1])
        observed_betas.append(gen_data["model_with_error"].params[1])

    return true_betas, observed_betas


def plot_monte_carlo(betas_true, betas_obs):
    fig, ax = plt.subplots(figsize=(8, 6))

    b1 = b1_cust  # True beta value for reference

    # Overlapping histograms for true beta estimates and observed beta estimates
    ax.hist(
        betas_true,
        # bins=15,
        color="green",
        alpha=0.5,
        edgecolor="black",
        label="Betas from model without ME",
    )
    ax.hist(
        betas_obs,
        # bins=15,
        color="blue",
        alpha=0.5,
        edgecolor="black",
        label="Betas from model with ME",
    )

    # Vertical line for the true beta value
    ax.axvline(b1, color="red", linestyle="--", label=f"True beta = {b1:.1f}")

    # Labels and title
    ax.set_xlabel("Estimated Beta")
    ax.set_xlim([b1_cust - 2, b1_cust + 2])
    ax.set_ylim([0, 30])
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    return fig


num_sim_cust = 100
mc_true_betas, mc_obs_betas = monte_carlo_betas(
    b0_cust,
    b1_cust,
    sd_eps_cust,
    sd_eta_cust,
    n_cust,
    st.session_state.random_seed,
    num_sim_cust,
)

with chart2_col:
    st.markdown(r"#### $\beta$'s from " + f"{num_sim_cust}" + r" draws")
    st.pyplot(
        plot_monte_carlo(mc_true_betas, mc_obs_betas),
        use_container_width=True,
    )


s0, c03, s1 = utl.wide_col()

with c03:
    st.markdown("### Interesting Takeaways and Theory")

    with st.expander("Click to expand.", expanded=False):
        st.markdown(
            r"""    
1. Classical measurement error in the dependent variable does not lead to biased estimates.<br>
    Consider the regression $\tilde{y} = \beta_1 x + \tilde{\varepsilon}$. We can rewrite it as $y + \eta = \beta_1 x+ \tilde{\varepsilon}$ and further as $y = \beta_1 x + \tilde{\varepsilon} + \eta$.
    Note that this is exactly the true regression, except that the error term is $\varepsilon = \tilde{\varepsilon} + \eta$. Given that both $\tilde{\varepsilon}$ and $\eta$ are mean $0$
    and independent of $x$, we get $E[\tilde{\varepsilon} + \eta | x]=0$ as well. Therefore, even in the presence of the measurement error in $y$, $\hat{\beta}$ is unbiased, i.e.,
    $E[\hat{\beta}] = \beta + cov(x, \tilde{\varepsilon} + \eta) = \beta$ 



2. Variance of $\hat{\beta}$ increases with variance in the measurement error $\sigma_{\eta}^2$.<br>
""",
            unsafe_allow_html=True,
        )

        st.latex(
            r"""
    var(\hat{\beta}|x) = \frac{\sigma_{\tilde{\varepsilon}}^2}{\sum (x_i - \bar{x})^2} = \frac{\sigma_{\varepsilon}^2 + \sigma_\eta^2}{\sum (x_i - \bar{x})^2} >
    \frac{\sigma_{\varepsilon}^2}{\sum (x_i - \bar{x})^2}
        """
        )

s0, c04, s1 = utl.wide_col()

with c04:
    st.header("2. Error in Covariates")

    st.markdown(
        r"""All assumptions are the same as above, except now the measurement error $\eta \sim N(0, \sigma_\eta^2)$ affects $x$:
        """,
        unsafe_allow_html=True,
    )
    st.latex(
        r"""
        \begin{array}{l r}
            \text{True DGP:} & y_i = \beta_0 + \beta_1 x_i+ \varepsilon_i \\
            \text{Observed x: } & \tilde{x_i} = x_i + \eta_i\\  
            \text{Regression: } & y_i = \beta_0 + \beta_1 \tilde{x_i} + \tilde{\varepsilon_i}\\
        \end{array}
        """
    )

    st.markdown(
        r"""Just as before, our goal is to check whether $\hat{\beta}$ coefficients are unbiased estimates for $\beta$ in the presence of measurement error in $x$.<br>
        Note that differently from before, $\tilde{\varepsilon} = \varepsilon - \beta_1 \eta$. Same as before, $\beta_0 = 0$ and $\varepsilon \sim N(0, \sigma_\varepsilon^2)$.<br>
        Vary the parameters below to see the effects of measurement error:""",
        unsafe_allow_html=True,
    )


def gen_reg_data_with_x_error(b0, b1, sd_eps, sd_eta, N, rseed):
    np.random.seed(rseed)

    # Generate X as a single independent variable
    X_true = np.random.uniform(-5, 5, 1000)
    eps = np.random.normal(0, sd_eps, 1000)  # Generate random noise

    # Take the first N samples
    X_true = X_true[:N]
    eps = eps[:N]

    # Generate the true y
    y = b0 + b1 * X_true + eps

    # Introduce measurement error in X to create observed X_tilde
    eta = np.random.normal(0, sd_eta, N)  # Measurement error in X
    X_tilde = X_true + eta  # Observed X with measurement error

    # Add a constant to X for regression purposes
    X_with_error = sm.add_constant(X_tilde)
    X_true_with_const = sm.add_constant(X_true)

    # OLS regression with observed X_tilde
    model_with_error = sm.OLS(y, X_with_error).fit()
    y_hat_with_error = model_with_error.predict(X_with_error)

    predictions_with_me = model_with_error.get_prediction(X_with_error)
    ci_me = predictions_with_me.conf_int(alpha=0.05)

    # OLS regression with true X (no measurement error) for comparison
    model_true = sm.OLS(y, X_true_with_const).fit()
    y_hat_true = model_true.predict(X_true_with_const)

    return {
        "y_true": y,
        "X_true": X_true,
        "X_tilde": X_tilde,
        "y_hat_with_error": y_hat_with_error,
        "y_hat_true": y_hat_true,
        "ci_me": ci_me,
        "beta": np.array([b0, b1]),
        "model_with_error": model_with_error,
        "model_true": model_true,
    }


def plot_ols_x_error(data):
    fig, ax = plt.subplots(figsize=(8, 6))

    b0_true, b1_true = data["model_true"].params
    b0_obs, b1_obs = data["model_with_error"].params

    # Scatter plot for true X values
    ax.scatter(
        data["X_true"],
        data["y_true"],
        color=thm.set1_green,
        alpha=0.5,
        label=r"True $X$",
    )

    # Scatter plot for observed X_tilde values with measurement error
    ax.scatter(
        data["X_tilde"],
        data["y_true"],
        color=thm.set1_blue,
        alpha=0.5,
        label=r"Observed $\tilde{X}$",
    )

    # Fit line for true model
    sorted_indices = np.argsort(data["X_true"])
    sorted_x_true = data["X_true"][sorted_indices]
    sorted_y_hat_true = data["y_hat_true"][sorted_indices]

    ax.plot(
        sorted_x_true,
        sorted_y_hat_true,
        color=thm.set1_green,
        linewidth=2,
        label=f"True fit: $\hat{{y}} = {b0_true:.2f} + {b1_true:.2f}x$",
    )

    # Fit line for observed model with measurement error
    sorted_x_obs = data["X_tilde"][sorted_indices]
    sorted_y_hat_with_error = data["y_hat_with_error"][sorted_indices]
    sorted_ci_lower = data["ci_me"][:, 0][sorted_indices]
    sorted_ci_upper = data["ci_me"][:, 1][sorted_indices]

    ax.plot(
        sorted_x_obs,
        sorted_y_hat_with_error,
        color=thm.set1_blue,
        linewidth=2,
        linestyle="--",
        label=f"Observed fit: " + r"$\hat{y}$" + f"= {b0_obs:.2f} + {b1_obs:.2f}x$",
    )

    # ax.fill_between(
    #     sorted_x_obs,
    #     sorted_ci_lower,
    #     sorted_ci_upper,
    #     color=thm.set1_blue,
    #     alpha=0.3,
    #     label="95% Confidence Interval",
    # )

    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.legend(loc="upper left", fontsize="small")
    return fig


# Monte Carlo analysis function for Streamlit
def monte_carlo_betas_x_error(b0, b1, sd_eps, sd_eta, N, rseed, num_simulations=100):
    true_betas = []
    observed_betas = []

    for i in range(num_simulations):
        # Generate data and get the slope estimates
        gen_data = gen_reg_data_with_x_error(b0, b1, sd_eps, sd_eta, N, rseed + i)
        true_betas.append(gen_data["model_true"].params[1])
        observed_betas.append(gen_data["model_with_error"].params[1])

    return true_betas, observed_betas


def plot_monte_carlo_x_error(betas_true, betas_obs):
    fig, ax = plt.subplots(figsize=(8, 6))

    b1 = b1_cust  # True beta value for reference

    # Overlapping histograms for true beta estimates and observed beta estimates
    ax.hist(
        betas_true,
        color="green",
        alpha=0.5,
        edgecolor="black",
        label="Betas from model without ME",
    )
    ax.hist(
        betas_obs,
        color="blue",
        alpha=0.5,
        edgecolor="black",
        label="Betas from model with ME",
    )

    # Vertical line for the true beta value
    ax.axvline(b1, color="red", linestyle="--", label=f"True beta = {b1:.1f}")

    ax.set_xlabel("Estimated Beta")
    ax.set_xlim([-1, b1_cust + 2])
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    return fig


# Generate data and Monte Carlo results for the new scenario
st.session_state.reg_data_x_error = gen_reg_data_with_x_error(
    b0_cust,
    b1_cust,
    sd_eps_cust,
    sd_eta_cust,
    n_cust,
    st.session_state.random_seed,
)

mc_true_betas_x, mc_obs_betas_x = monte_carlo_betas_x_error(
    b0_cust,
    b1_cust,
    sd_eps_cust,
    sd_eta_cust,
    n_cust,
    st.session_state.random_seed,
    num_sim_cust,
)

# Chart columns
chart1_col, chart2_col = st.columns(2)

with chart1_col:
    st.markdown(r"#### Regression with $X$ and $\tilde{X}$")
    st.pyplot(
        plot_ols_x_error(st.session_state.reg_data_x_error), use_container_width=True
    )

with chart2_col:
    st.markdown(r"#### $\beta$'s from " + f"{num_sim_cust}" + r" draws (X error)")
    st.pyplot(
        plot_monte_carlo_x_error(mc_true_betas_x, mc_obs_betas_x),
        use_container_width=True,
    )
