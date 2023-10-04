import os
import random
import sys

import numpy as np
import pandas as pd
import src.scripts.plot_themes as thm
import src.scripts.utils as utl
import statsmodels.api as sm
import streamlit as st
from matplotlib import pyplot as plt
from scipy.stats import t
from st_pages import add_page_title

### PAGE CONFIGS ###

st.set_page_config(
    page_title="PhD Econometrics - OLS",
    page_icon="📈",
    layout="wide",
)
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
    st.title("Ordinary Least Squares Estimation")
    st.header("1. Visualizing OLS estimates")

    st.write("Play around with sliders to see how the data and estimates change.")
    st.write(
        r"Suppose you have the following true population relationship between $X$ and $Y$, with parameters defined by slider values."
    )
    st.write(
        r"You then draw a sample of size $n$ from that population and estimate OLS coefficients, $\hat{\beta_0}$ and $\hat{\beta_1}$."
    )


def gen_lin_data(b0, b1, sd, N, rseed):
    np.random.seed(rseed)
    # generate X
    K = 2
    X = np.random.uniform(-10, 10, (N, K - 1))
    X = np.column_stack((np.ones(N), X))

    # generate  error term
    eps = np.random.normal(0, sd, N)

    # y = xB + eps
    y = np.dot(X, np.array([b0, b1])) + eps

    # fit reg
    model = sm.OLS(y, X).fit()

    # get fitted values and CI
    predictions = model.get_prediction(X)
    y_hat = predictions.predicted_mean
    y_hat_se = predictions.se_mean

    # get 95% confidence interval
    ci = predictions.conf_int(alpha=0.05)  # 95% CI
    deg_freedom = X.shape[0] - X.shape[1]  # n - K

    # get CI manually - not needed
    # t_score = t.ppf(0.975, deg_freedom)
    # ci = np.column_stack(
    #     (y_hat - t_score * y_hat_se, y_hat + t_score * y_hat_se)
    # )

    # get error parameters - not needed
    e = y - y_hat
    s = np.sqrt(np.sum(e**2) / deg_freedom)

    # calculate R^2 manually - not needed
    y_bar = np.mean(y)
    ss_tot = np.sum((y - y_bar) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {
        "y": y,
        "x": X,
        "s": s,
        "y_hat": y_hat,
        "ci": ci,
        "model": model,
    }


with c01:
    st.latex(
        r"""
            Y_i = \beta_0 + \beta_1X_i + \varepsilon_i \text{, where }  \varepsilon \sim N(0, \sigma^2)
        """
    )
    st.latex(r"""\hat{Y_i} = """ + r"""\hat{\beta_0} + \hat{\beta_1}X""")


slider_col, s1, chart_col = st.columns((0.8, 0.1, 1))

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
        "Sample size, $n$",
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
        custom_data = gen_lin_data(b0_cust, b1_cust, var_cust, n_cust, random_seed)


coefficients = create_summary(custom_data)

with chart_col:
    chart_col.pyplot(plot_ols(custom_data, b0_cust, b1_cust), use_container_width=True)


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
        f"n= {n_cust} ,"
        + f"R^2 = {custom_data['model'].rsquared:.2f}"
        + r", s = \sqrt{\frac{\mathbf{e'e}}{n - K}}"
        + f"= {custom_data['s']:.2f}"
    )

s0, c03, s1 = utl.wide_col()

with c03:
    st.markdown("### Interesting takeaways")

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
    
        $\widehat{var(\mathbf{b}| \mathbf{X})} \equiv s^2 (X'X)^{-1} = \frac{\mathbf{e'e}}{n - K} (X'X)^{-1}$
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

s0, c04, s1 = utl.wide_col()

with c04:
    st.header("4. Theory with Code")

    def tabs_code_theory():
        return st.tabs(["Theory", "Code numpy", "Code statsmodels"])

    st.markdown(
        "#### Solving for OLS coefficients",
        unsafe_allow_html=True,
    )
    # OLS Theory vs Code
    f1_c1, f1_c2, f1_c3 = tabs_code_theory()
    with f1_c1:
        st.markdown(
            r"""
            Let $\mathbf{x}_i$ and $\beta$ be $K \times 1$ vectors, and $\mathbf{X}$ be $n \times K$ matrix of $\mathbf{x}_i$<br>
            $y_i = \mathbf{x}_i' \beta + \varepsilon_i, (i = 1, 2, ..., n)$<br>
            $\mathbf{y = X \beta + \varepsilon}$<br>
            <br>
            Let $\mathbf{e \equiv y - Xb}$ and solve for $\mathbf{b}$ that minimizes $\mathbf{e'e}$<br>
            $\mathbf{b = (X'X)^{-1}X'y}$ is then the OLS estimate $\beta$<br>
            $\mathbf{\hat{y} = X b}$ <br>
            <br>
            $\mathbf{P_{n \times n} \equiv X(X'X)^{-1}X'}$ is the projection or hat matrix<br>
            $\mathbf{PX = P}$ and $\mathbf{Py = \hat{y}}$<br>
            <br>
            $\mathbf{M_{n \times n} \equiv I_n - P}$ is the annihilator or residual maker matrix<br>
            $\mathbf{MX = 0}$ and $\mathbf{My = e}$<br>
            

    """,
            unsafe_allow_html=True,
        )

    with f1_c2:
        ols_code = """
        import numpy as np
        # Define initial parameters
        n = 1000 # sample size
        K = 2 # number of variables and coefficients
        sigma = 1 # standard deviation of error term
        X = np.random.uniform(-10, 10, (n, K - 1)) # generate random X
        X = np.column_stack((np.ones(n), X)) # add constant
        beta = np.random.uniform(-5, 5, K) # generate random true beta
        eps = np.random.normal(0, sigma, n) # generate random errors (disturbances)

        # Generate true y
        y = np.dot(X, beta) + eps

        # Solve for OLS coefficients
        b_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        y_hat = X.dot(b_ols)
        e = y - y_hat

        # Get projection and residual maker matrices
        P = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
        M = np.eye(n) - P

        """
        st.code(ols_code, language="python")

    with f1_c3:
        ols_code_b = """
        import statsmodels.api as sm
        # Generate data with numpy same as before
        # ...

        # Fit OLS model with statsmodels
        model = sm.OLS(y, X).fit()
        b_ols = model.params
        y_hat = model.predict(X)
        e = model.resid

        # statsmodels does not have built-in P and M matrices
        # since you can pull necessary estimates from model.get_prediction(X)

        """

        st.code(ols_code_b, language="python")

    st.divider()

    ### OLS Additional Concepts
    st.markdown(
        "#### World of Errors",
        unsafe_allow_html=True,
    )
    st.markdown(
        """NB: this got me confused for a minute because Hayashi and Greene classically disagree on what SSR means, so I'll follow Greene.""",
        unsafe_allow_html=True,
    )

    f2_c1, f2_c2, f2_c3 = tabs_code_theory()
    with f2_c1:
        st.markdown(
            r"""
            Error Sum of Squares (SSE) aka Sum of Squared Residuals (SSR or RSS, hence confusion):<br>
            $SSE = \sum_{i=1}^n (y_i-\hat{y_i})^2 = \sum_{i=1}^n (e_i)^2 =  \mathbf{e'e = \varepsilon' M \varepsilon}$ (this is SSR according to Hayashi)<br>

            Regression sum of squares (SSR) aka Explained Sum of Squares (ESS):<br>
            $SSR = \sum_{i=1}^n (\hat{y_i} - \bar{y})^2 = \sum_{i=1}^n (\hat{y_i} - \bar{\hat{y}})^2$<br>
            $SSR =  \mathbf{b'X'M^0Xb}$, where $\mathbf{M^0}$ is the centering matrix<br>

            Total sum of squares (SST) aka Total Variation:<br>
            $SST = \sum_{i=1}^n (y_i-\bar{y_i})^2 = \sum_{i=1}^n (\hat{y_i} - \bar{y})^2 + \sum_{i=1}^n (e_i)^2$ <br>
            $SST = \mathbf{y'M^0y = b'X'M^0Xb + e'e = SSR + SSE}$<br>
            
            OLS estimate of $\sigma^2$ aka Standard Error of the Regression (SER):<br>
            $s^2 \equiv \frac{SSE}{n-K} = \frac{\mathbf{e'e}}{n-K}$<br>
            $SER = \sqrt{s^2} = s$ (think of MSE)<br>
        
            Sampling error:<br>
            $\mathbf{b} - \beta = \mathbf{(X'X)^{-1}X'y - \beta} = \mathbf{(X'X)^{-1}X' \varepsilon}$ (sampling error)<br>
            
            Conditional variance of $\mathbf{b}$ and standard error of $b_k$:<br>
            $Var(\mathbf{b|X}) = s^2(X'X)^{-1}$<br>
            $SE(b_k) = \{[s^2(X'X)^{-1}]_{kk}\}^{1/2}$<br>
            (square root of *k*th diagonal element of the variance matrix)
         """,
            unsafe_allow_html=True,
        )

    with f2_c2:
        ols_code_err = """
        import numpy as np

        # Sum of squared errors
        SSE = e.dot(e)
        
        # Regression sum of squares
        y_hat_centered = y_hat - np.mean(y_hat)
        SSR = y_hat_centered.dot(y_hat_centered)

        # Total sum of squares
        y_centered = y - np.mean(y)
        SST = y_centered.dot(y_centered)

        # OLS estimate for sigma^2 and sigma
        s_sq = SSE / (n - K)
        s = np.sqrt(s_sq)

        # Sampling error (only b/c we know true beta values)
        sampling_error = b_ols - beta

        # Conditional variance and standard errors of beta
        var_b = s_sq * np.linalg.inv(X.T.dot(X))
        SE_b = np.sqrt(np.diag(var_b))
        """
        st.code(ols_code_err, language="python")

    with f2_c3:
        ols_code_err_b = """
        import statsmodels.api as sm
        model = sm.OLS(y, X).fit()

        # Sum of squared errors
        SSE = model.ssr # this is SSE according to Greene

        # Regression sum of squares
        SSR = model.ess
        
        # Total sum of squares
        SST = SSE + SSR
        
        # OLS estimate for sigma^2 and sigma
        s_sq = model.mse_resid
        s = model.mse_resid ** 0.5
        
        # Sampling error (only b/c we know true beta values)
        sampling_error = model.params - beta
        
        # Conditional variance and standard errors of beta
        var_b = model.cov_params()
        SE_b = model.bse
        """

        st.code(ols_code_err_b, language="python")

    st.divider()

    st.markdown("#### $R^2$ centered and $R^2$ adjusted")
    st.markdown(
        r"""NB: this requires a constant term to be included in the model, i.e. $\mathbf{X}$ contains a column of 1s.""",
        unsafe_allow_html=True,
    )

    f3_c1, f3_c2, f3_c3 = tabs_code_theory()

    with f3_c1:
        st.markdown(
            r"""            
            $R^2 = \frac{SSR}{SST} = \frac{SST - SSE}{SST} = 1 - \frac{SSE}{SST}= 1- \mathbf{\frac{e'e}{y'M^0y}}$<br>
            <br>
            $\bar{R}^2 = 1 - \frac{n - 1}{n - K} (1 - R^2)$""",
            unsafe_allow_html=True,
        )

    with f3_c2:
        r2_code = """
        import numpy as np
        e_dot_e = e.dot(e)
        y_bar = np.mean(y)
        y_centered = y - y_bar
        y_M0_y = y_centered.dot(y_centered)
        r_sq = 1 - e_dot_e / y_M0_y
        r_sq_adj = 1 - ((n - 1) / (n - K)) * (1 - r_sq)
        """
        st.code(r2_code, language="python")

    with f3_c3:
        r2_code_built_in = """
        import statsmodels.api as sm
        model = sm.OLS(y, X).fit()
        r_sq = model.rsquared
        r_sq_adj = model.rsquared_adj
        """

        st.code(r2_code_built_in, language="python")

    st.divider()

s0, c05, s1 = utl.wide_col()
with c05:
    st.header("5. Proofs to remember")
    sst_proof = "https://stats.stackexchange.com/questions/207841/why-is-sst-sse-ssr-one-variable-linear-regression/401299#401299"

    with st.expander("SST = SSR + SSE"):
        st.markdown(
            rf"Proof from Greene Section 3.5 (also see [Stack Exchange]({sst_proof})):<br>"
            + r"""
                $y_i - \bar{y} = \mathbf{x}_i'\mathbf{b} + e_i$<br>
                $y_i - \bar{y} = \hat{y}_i - \bar{y} + e_i = (\mathbf{x}_i - \mathbf{\bar{x}})'\mathbf{b} + e_i$<br>
                $\mathbf{M^0y= M^0Xb + M^0e}$<br>
                $SST = \mathbf{y'M^0y = b'X'M^0Xb + e'e} = SSR + SSE$<br>
                (need to expand between last two steps, but main trick is that $\mathbf{e'M^0X = e'X=0}$)<br>
                """,
            unsafe_allow_html=True,
        )

    with st.expander("Transformed Variables"):
        st.markdown(
            r"""
            Proof from Greene THEOREM 3.8:<br>
            Let $\mathbf{b}$ be the OLS coefficients from regressing $\mathbf{y}$ on $\mathbf{X}$.<br>
            Regress $\mathbf{y}$ on transformed variables $\mathbf{Z = XP}$, where $\mathbf{P}$ transforms columns of $\mathbf{X}$.<br>
            Then OLS coefficients are:<br>
            $\mathbf{d = (Z'Z)^{-1}Z'y = [(XP)'(XP)]^{-1}(XP)'y = [P'X'XP]^{-1}P'X'y =}$<br>
            $\mathbf{= P^{-1}(X'X)^{-1}P'^{-1}P'X'y = P^{-1}(X'X)^{-1}X'y = P^{-1}b}$<br>
            The vector of residuals is:<br>
            $\mathbf{u = y - Z(P^{-1}b) = y - XP(P^{-1}b) = y - Xb = e}$<br>
            Since residuals are identical and $\mathbf{y'M^0y}$ is unchanged, $R^2$ is also identical in two regressions.
            """,
            unsafe_allow_html=True,
        )
