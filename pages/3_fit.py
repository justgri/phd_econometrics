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

## Data viz part
# How R-sq is calculated
# How R-sq and other measures depend on sample size, error variance, and true parameters
# Also how much R-sq, AIC, and BIC are penalized by adding more regressors?

## Theory part
# TBD

# create one column with consistent width
s1, c01, s2 = utl.wide_col()

### PAGE START ###
# Dashboard header

with c01:
    st.title("Model fit measures")
    st.divider()
    st.header("1. Visualizing R-squared")

    st.markdown(
        r"""Suppose the true population relationship between $X$ and $y$ is linear and defined by the slider values below.
        Move the sliders to see how the model fit measures change with different population parameters.""",
        unsafe_allow_html=True,
    )


def gen_lin_data(b0, b1, sd, N, rseed):
    np.random.seed(rseed)
    # fix X values
    X_fix = [-2, -0.5, 2, 0.5, -1.25, 1.25, 0]
    # X_pos_ref = [0, 2, 6, 4, 1, 5, 3]

    K = 2
    X = np.array(X_fix[:N])
    X = np.column_stack((np.ones(len(X)), X))

    # generate  error term
    eps_fix = np.array([-1 * sd, sd, -sd, -sd, -1 * sd, 1 / 2 * sd, -sd])
    eps = np.array(eps_fix[:N])
    # eps = np.random.normal(0, sd, N)

    # y = xB + eps
    y = np.dot(X, np.array([b0, b1])) + eps

    # fit reg
    model = sm.OLS(y, X).fit()

    # get fitted values and CI
    predictions = model.get_prediction(X)
    y_hat = predictions.predicted_mean
    y_hat_se = predictions.se_mean

    # get error parameters
    e = y - y_hat
    deg_freedom = X.shape[0] - X.shape[1]  # n - K
    s = np.sqrt(np.sum(e**2) / deg_freedom)

    # calculate R^2 manually
    y_bar = np.mean(y)
    ss_tot = np.sum((y - y_bar) ** 2)
    ss_res = np.sum((y - y_hat) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return {
        "y": y,
        "x": X,
        "s": s,
        "y_hat": y_hat,
        "model": model,
    }


slider_col, s1, chart_col = st.columns((0.8, 0.1, 1))

with slider_col:
    # Sliders
    b0_cust = st.slider(
        r"Intercept, $\beta_0$",
        min_value=-1.5,
        max_value=1.5,
        value=0.0,
        step=0.1,
    )
    b1_cust = st.slider(
        r"Slope, $\beta_1$", min_value=-1.5, max_value=1.5, value=1.0, step=0.1
    )
    var_cust = st.slider(
        r"Error SD, $\sqrt{var(\varepsilon)} = \sigma$",
        min_value=0.1,
        max_value=3.0,
        value=2.0,
        step=0.1,
    )

    n_cust = st.slider(
        "Points displayed, $n$",
        min_value=3,
        max_value=7,
        value=4,
        step=1,
    )

custom_data = gen_lin_data(b0_cust, b1_cust, var_cust, n_cust, random_seed)


def plot_ols(data_custom, b0, b1):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0)  # remove margin
    # Define the upper and lower bounds for axes
    y_ub = 6
    y_lb = -y_ub
    x_ub = 3
    x_lb = -x_ub

    ### SAMPLE DATA
    ax.scatter(
        data_custom["x"][:, 1],
        data_custom["y"],
        # label="Custom Data",
        color=thm.cols_set1_plt[1],
        alpha=0.9,
        zorder=10,
        edgecolors="none",
        s=60,
    )

    ### REG LINE
    # Regression line
    b0_s = data_custom["model"].params[0]
    b1_s = data_custom["model"].params[1]

    if b1_s >= 0:
        label_s = rf"$\hat{{y}}_i = {b0_s:.2f} + {b1_s:.2f}x_i$"
    else:
        label_s = rf"$\hat{{y}}_i = {b0_s:.2f} - {-b1_s:.2f}x_i$"

    ax.plot(
        data_custom["x"][:, 1],
        data_custom["y_hat"],
        label=label_s,
        color=thm.cols_set1_plt[4],
        zorder=9,
        linewidth=2.5,
    )

    ### Y BAR
    y_mean = np.mean(data_custom["y"])
    plt.axhline(y=y_mean, color=thm.cols_set1_plt[2], linestyle="-", linewidth=2.5)

    # Y BAR text
    y_offset = 0.35 if b1_s > 0 else -0.45
    ax.text(
        x_lb + 0.1,
        y_mean + y_offset,
        r"$\bar{y}=$" + f"{y_mean:.2f}",
        color="green",
        verticalalignment="center",
    )

    ### REG RESIDUALS (for residual sum of squares)
    res_offset = 0.05
    has_lab_res = False
    for xi, yi, yhati in zip(
        data_custom["x"][:, 1], data_custom["y"], data_custom["y_hat"]
    ):
        if not has_lab_res:
            lab_res = r"$y_i - \hat{y_i}$ for SSE"
            has_lab_res = True
        else:
            lab_res = None  # No label for other lines

        plt.plot(
            [xi - res_offset, xi - res_offset],
            [yi, yhati],
            color=thm.cols_set1_plt[4],
            linestyle="--",
            alpha=0.9,
            label=lab_res,
            linewidth=2,
        )

    ### DEVIATIONS FROM THE MEAN (for total sum of squares)
    has_lab_mean = False
    # Add horizontal lines from each data point to the mean of Y (Total Sum of Squares)
    for xi, yi in zip(data_custom["x"][:, 1], data_custom["y"]):
        if not has_lab_mean:
            lab_mean = r"$y_i - \bar{y}\;$ for TSS"
            has_lab_mean = True
        else:
            lab_mean = None  # No label for other lines

        plt.plot(
            [xi + res_offset, xi + res_offset],
            [yi, y_mean],
            color=thm.cols_set1_plt[2],
            linestyle=":",
            alpha=0.9,
            label=lab_mean,
            linewidth=2,
        )

    # Add R-sq label
    r_squared = data_custom["model"].rsquared

    # Add R^2 to the top right corner
    rsq_offset = 0.1
    ax.text(
        x_ub - rsq_offset,
        y_ub - rsq_offset - 0.1,
        f"$R^2 = {r_squared:.2f}$",
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=10,
    )

    plt.xlim([x_lb, x_ub])
    plt.ylim([y_lb, y_ub])
    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")
    ax.yaxis.set_label_coords(-0.08, 0.5)

    legend_loc = "upper left" if data_custom["y_hat"][0] < 3.25 else "lower left"
    plt.legend(loc=legend_loc, fontsize="small")

    return fig


with chart_col:
    chart_col.pyplot(plot_ols(custom_data, b0_cust, b1_cust), use_container_width=True)


def generate_html_table(model):
    # Extract statistics
    ess = np.sum(model.resid**2)
    tss = np.sum((model.model.endog - np.mean(model.model.endog)) ** 2)
    r2 = model.rsquared
    r2_adj = model.rsquared_adj

    likelihood = model.llf
    aic = model.aic
    bic = model.bic
    N = model.nobs

    data = [
        ("SSE (Error Sum of Squares)", ess),
        ("TSS (Total Sum of Squares)", tss),
        ("R-squared", r2),
        ("Adjusted R-squared", r2_adj),
        # ("Likelihood", likelihood),
        # ("AIC", aic),
        # ("BIC", bic),
    ]

    html_string = """<table class="table" border="1">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>"""

    for name, value in data:
        html_string += f"<tr><td>{name}</td><td>{value:.2f}</td></tr>"

    html_string += "</tbody></table>"

    return html_string


table_html_string = generate_html_table(custom_data["model"])

_, formula_col, s1, table_col, _ = st.columns((0.2, 1, 0.1, 1, 0.2))

with table_col:
    st.markdown(table_html_string, unsafe_allow_html=True)

with formula_col:
    st.markdown(r"""<h5>Calculations</h5>""", unsafe_allow_html=True)
    st.markdown(
        r"""
        $SSE = \sum_{i=1}^n (y_i-\hat{y_i})^2 = \mathbf{e'e}$<br>
        $TSS = \sum_{i=1}^n (y_i-\bar{y_i})^2 = \mathbf{y'M^0y}$<br>
        $R^2 = 1 - \frac{SSE}{SST} = 1- \mathbf{\frac{e'e}{y'M^0y}}$<br>
        $\bar{R}^2 = 1 - \frac{n - 1}{n - K} (1 - R^2)$<br>
""",
        unsafe_allow_html=True,
    )
_, nb_col, _ = st.columns((0.1, 1, 0.1))
nb_col.markdown(
    r"NB: Adjusted R-squared $\bar{R}^2$ is heavily penalized here, because the sample size is very small.",
    unsafe_allow_html=True,
)


s0, c03, s1 = utl.wide_col()

with c03:
    st.markdown("### Interesting takeaways")

    with st.expander("Click to expand."):
        st.markdown(
            r"""
        1. $R^2$ is expected to be 0 if $\beta_1=0$ or if $X_i = \bar{X}$.<br>

            $R^2 = \frac{ (\hat{y} - \bar{y})' (\hat{y} - \bar{y}) }{ (y - \bar{y})' (y - \bar{y}) }
            = \frac{\sum_{i=1}^{N} (\hat{y}_i - \bar{y})^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}
            = \frac{\sum_{i=1}^{N} (\hat{\beta_1} (X_i - \bar{X}))^2}{\sum_{i=1}^{N} (y_i - \bar{y})^2}$
            , because $\hat{\beta_0} = \bar{y} - \hat{\beta_1}\bar{X}$

        2. $R^2$ is independent of the size of the intercept $b_0$ but is dependent on the size of the slope $b_1$.
        
            """,
            unsafe_allow_html=True,
        )

        # Explain why R^2 increases with \beta - what's the interpretation?

        # ADD FORMULAS FOR CONFIDENCE INTERVALS

        # related - show that R and CI are related?
        # check against statsmodels link below
        # https://www.statsmodels.org/0.9.0/_modules/statsmodels/regression/_prediction.html#PredictionResults
    #     st.markdown(
    #         r"""
    #     e. Confidence interval is visually more sensitive to sample size than to error variance, i.e.,
    #     if $n$ is large enough, even for high $\sigma^2$, the confidence interval is small because it depends only on $s^2$ and not on $\sigma^2$:<br>
    #     Also, CI is wider when further away from mean.<br>
    # """,
    #         unsafe_allow_html=True,
    #     )


with c03:
    st.header("2. Interpretation")
    st.markdown(r"""<h5>R-squared indicates:</h5>""", unsafe_allow_html=True)
    st.markdown(
        r"""
    1. "Whether variation in $x$ is a good predictor of variation in $y$" - useful for prediction accuracy (G Ch 3.5)<br>
    2. Proportion of total variation in $y$ that is accounted by variation in $x$ - useful for model comparison<br>
        NB: The dependent variable has to be of the same transformation (e.g., linear or log) for comparison.       
    3. How close the data are to the estimated model - useful to get a sense of over/underfitting
""",
        unsafe_allow_html=True,
    )

    st.markdown(r"""<h5>R-squared does NOT indicate:</h5>""", unsafe_allow_html=True)
    st.markdown(
        r"""
        1. Causality
        2. Unbiasedness of the coefficients
        3. Appropriateness of the model
        4. How reasonable OOS predictions are (e.g., high R-sq doesn't prevent forecasting of negative wages)

        NB: In general, **interpretation of the R-squared values is highly dependent on the context**. 
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""<h5>R-sq and R-sq adj. vs AIC and BIC</h5>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""Since R-sq increases by including more regressors, we need a measure that penalizes for adding regressors.
        It has been argued that the adjusted R-sq doesn't penalize heavily enough, thus AIC and BIC measures have been proposed.(Greene p. 47)<br>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""
        "Choosing a model based on the lowest AIC is logically the same as using $\bar{R}^2$ in the linear model, nonstatistical, albeit widely accepted.
        The AIC and BIC are information criteria, not fit measures as such." (Greene, p. 561)<br>
        APC has a direct relationship to $R^2$.<br>
        """,
        unsafe_allow_html=True,
    )

s0, c04, s1 = utl.wide_col()

with c04:
    st.header("3. Theory with code")

    def tabs_code_theory():
        return st.tabs(["Theory", "Code statsmodels", "Code numpy"])

    ### Error sums
    st.markdown(
        "#### Error sums",
        unsafe_allow_html=True,
    )
    st.markdown(
        """NB: Hayashi and Greene classically disagree on notation for the sum of squared residuals (SSE or SSR), so I'll follow Greene.""",
        unsafe_allow_html=True,
    )

    f2_c1, f2_c2, f2_c3 = tabs_code_theory()
    with f2_c1:
        st.markdown(
            r"""
            Error Sum of Squares (SSE) aka Sum of Squared Residuals (SSR or RSS, hence confusion):<br>
            $SSE = \sum_{i=1}^n (y_i-\hat{y_i})^2 = \sum_{i=1}^n (e_i)^2 =  \mathbf{e'e = \varepsilon' M \varepsilon}$<br>
            (this is SSR according to Hayashi)<br>

            Regression sum of squares (SSR) aka Explained Sum of Squares (ESS):<br>
            $SSR = \sum_{i=1}^n (\hat{y_i} - \bar{y})^2 = \sum_{i=1}^n (\hat{y_i} - \bar{\hat{y}})^2$<br>
            $SSR =  \mathbf{b'X'M^0Xb}$, where $\mathbf{M^0}$ is the centering matrix<br>

            Total sum of squares (SST) aka Total Variation:<br>
            $SST = \sum_{i=1}^n (y_i-\bar{y_i})^2 = \sum_{i=1}^n (\hat{y_i} - \bar{y})^2 + \sum_{i=1}^n (e_i)^2$ <br>
            $SST = \mathbf{y'M^0y = b'X'M^0Xb + e'e = SSR + SSE}$<br>
         """,
            unsafe_allow_html=True,
        )

    with f2_c2:
        ols_code_err_b = """
        import statsmodels.api as sm
        model = sm.OLS(y, X).fit()

        # Sum of squared errors
        SSE = model.ssr # this is SSE according to Greene

        # Regression sum of squares
        SSR = model.ess
        
        # Total sum of squares
        SST = SSE + SSR
        """

        st.code(ols_code_err_b, language="python")

    with f2_c3:
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
        """
        st.code(ols_code_err, language="python")

    st.divider()

    st.markdown("#### Model fit and selection measures")
    st.markdown(
        r"""
        NB: $R^2$ definition below requires a constant term to be included in the model.<br>
        """,
        unsafe_allow_html=True,
    )

    f3_c1, f3_c2, f3_c3 = tabs_code_theory()

    # Sources for AIC and BIC
    sas_source = "https://documentation.sas.com/doc/en/vfcdc/8.5/vfug/p0uawamu7dmtc2n1cllfwajyvlko.htm"
    stata_source = "https://www.stata.com/manuals13/restatic.pdf"
    stack_ex = "https://stats.stackexchange.com/questions/490056/aic-bic-formula-wrong-in-james-witten"

    with f3_c1:
        st.markdown(
            r"""          
            R-sq, Adjusted R-sq, and Pseudo R-sq:<br>
            $R^2 = \frac{SSR}{SST} = \frac{SST - SSE}{SST} = 1 - \frac{SSE}{SST}= 1- \mathbf{\frac{e'e}{y'M^0y}}$<br>
            $\bar{R}^2 = 1 - \frac{n - 1}{n - K} (1 - R^2)$<br>
            McFadden Pseudo  $R^2 = 1 - \frac{\text{ln} L}{\text{ln} L_0} = \frac{-\text{ln}(1-R^2)}{1+\text{ln}(2\pi) + \text{ln}(s_y^2)}$<br>
            
            Amemiya's Prediction Criterion (APC):<br>
            $APC=\frac{SSE}{n-K}(1+\frac{K}{n}) = SSE \frac{n+K}{n-K}$<br>

            AIC and BIC for OLS, when error variance is known (Greene p. 47):<br>
            $AIC = \text{ln}(\frac{SSE}{n}) + \frac{2K}{n}$<br>
            $BIC = \text{ln}(\frac{SSE}{n}) + \frac{\text{ln}(n) K}{n}$<br>
            
            AIC and BIC are more often calculated for any MLE as follows (Greene p. 561):<br>
            $AIC = -2 \text{ln}(L)+2K$<br>
            $BIC = -2 \text{ln}(L) + \text{ln}(n) K  $<br>
            
            In OLS, SSE is proportional to log-likelihood, so the two formulas would lead to the same model selection.<br>
            NB: Even for OLS, Python *statsmodels*, STATA *estat ic*, and R *lm* use the latter definition, whereas SAS uses the former multiplied by $n$.
            """,
            unsafe_allow_html=True,
        )

    with f3_c2:
        r2_code_built_in = """
        import statsmodels.api as sm
        import numpy as np

        model = sm.OLS(y, X).fit()

        # R-sq and R-sq adjusted
        r_sq = model.rsquared
        r_sq_adj = model.rsquared_adj

        # Pseudo R-sq
        ln_L = model.llf
        model_constant = sm.OLS(y, np.ones(n)).fit()
        ln_L_0 = model_constant.llf
        pseudo_r_sq = 1 - ln_L / ln_L_0

        # Amemiya's Prediction Criterion - no built-in module
        APC = (e.dot(e) / n) * (n + K) / (n - K)

        # AIC and BIC
        AIC = model.aic
        BIC = model.bic

        """

        st.code(r2_code_built_in, language="python")

    with f3_c3:
        r2_code = """
        import numpy as np
        # R-sq and R-sq adjusted
        y_centered = y - np.mean(y)
        r_sq = 1 - e.dot(e) / y_centered.dot(y_centered)
        r_sq_adj = 1 - ((n - 1) / (n - K)) * (1 - r_sq)

        # Pseudo R-sq
        var_y = np.var(y)
        pseudo_r_sq = (-1 * np.log(1 - r_sq) / (1 + np.log(2 * np.pi) + np.log(var_y)))
                
        # Amemiya's Prediction Criterion
        APC = (e.dot(e) / n) * (n + K) / (n - K)

        # AIC and BIC, first get log likelihood
        ln_L = (-n / 2) * (1 + np.log(2 * np.pi) + np.log(SSE / n))
        AIC = -2 * ln_L + 2 * K
        BIC = -2 * ln_L + K * np.log(n)
        """
        st.code(r2_code, language="python")

s0, c05, s1 = utl.wide_col()
with c05:
    st.header("4. More details")
    st.write("Check out Andrius Buteikis' book:")
    st.link_button(
        "Goodness-of-Fit",
        "http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/3-8-UnivarGoF.html",
        type="primary",
    )

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

    with st.expander("Relating two formulations of AIC (Greene pp. 47 and 561)"):
        st.markdown(
            r"""
            Not sure if this is useful, but it clarified things in my head.<br>
            
            Recall, $SSE = \mathbf{e'e}$<br>
            In the linear model with normally distributed disturbances, the maximized log likelihood is<br>
            $\text{ln} L = -\frac{n}{2} [1 + \text{ln}(2 \pi) + \text{ln}(\frac{SSE}{n})]$<br>
            Ignore the constants and notice that<br>
            $\text{ln} L \propto -\frac{n}{2} \text{ln}(\frac{SSE}{n})$<br>
            $-2 \text{ln} L \propto n \text{ln}(\frac{SSE}{n})$<br>
            $-2 \text{ln} L + 2K \propto n \text{ln}(\frac{SSE}{n}) + 2K$<br>
            $-2 \text{ln} L + 2K \propto \text{ln}(\frac{SSE}{n}) + \frac{2K}{n}$<br>
            Which we wanted to show.<br>
            Might have been enough to just state that $\text{ln} L \propto -\text{ln}(\frac{SSE}{n})$.
""",
            unsafe_allow_html=True,
        )

    # with st.expander(r"$0 \leq R^2 \leq 1$"):
    #     st.markdown("TBD")

    # with st.expander("R-sq increases with the number of regressors"):
    #     st.markdown("TBD")
