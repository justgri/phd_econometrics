import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

utl.local_css("src/styles/styles_pages.css")

random_seed = 0
s1, c1, s2 = utl.wide_col()

### PAGE START ###
# Dashboard header
with c1:
    st.title("Top 10 Theory Tools for Econometrics")
    st.divider()
    st.markdown(r"""<h3>According to Jeffrey Wooldridge</h3>""", unsafe_allow_html=True)


s1, c2, s2 = utl.wide_col()

#### EXAMPLE CHART FOR LINEARITY OF EXPECTATION AND VARIANCE
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 0


def render_lin_exp():
    header_col = st.container()

    def generate_data(a, b, expected_value):
        np.random.seed(st.session_state.random_seed)

        # Generate the first deviation (always positive)
        deviation1 = np.random.uniform(0.3, 1)
        deviation2 = np.random.uniform(deviation1 + 0.5, deviation1 + 1.5)
        deviations = np.array([deviation1, deviation2])

        # Calculate mirror deviations
        mirror_deviations = -deviations

        # Combine and adjust for the expected value
        X = np.concatenate([deviations, mirror_deviations]) + expected_value
        Y = a + b * X

        return X, Y

    def format_two_decimal(x):
        rounded_x = round(x, 2)
        return f"{rounded_x:.2f}" if abs(rounded_x) > 1e-6 else "0.00"

    def plot_expectation(X, Y, a, b):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.subplots_adjust(left=0)  # remove margin
        plt.xlabel("X", fontweight="bold")
        plt.ylabel("Y", fontweight="bold")
        y_lb, y_ub = -8, 8
        x_lb, x_ub = -4, 4

        ax.yaxis.set_label_coords(-0.08, 0.5)

        ax.scatter(X, Y, color=thm.set1_blue)

        # Regression Line (Deterministic Line based on the given a and b)
        x_values = [min(X), max(X)]  # first and last X
        y_values = a + b * np.array(x_values)
        ax.plot(x_values, y_values, color=thm.set1_orange, label=r"$Y = a + bX$")

        # X BAR line
        E_X = np.mean(X)
        ax.axvline(E_X, color=thm.set1_blue, linestyle="-", linewidth=2)

        ### X BAR text
        x_offset = -0.18
        ax.text(
            E_X + x_offset,
            7.0,
            r"$\bar{X}=$" + format_two_decimal(E_X),
            color=thm.set1_blue,
            verticalalignment="center",
            horizontalalignment="right",
        )

        # Y BAR line
        E_Y = np.mean(Y)
        ax.axhline(E_Y, color=thm.set1_green, linestyle="-", linewidth=2)

        # Y BAR text
        y_offset = 0.5
        ax.text(
            -3.9,
            E_Y + y_offset,
            r"$\bar{Y}=$" + format_two_decimal(E_Y),
            color=thm.set1_green,
            verticalalignment="center",
        )

        # Dashed lines from x_bar to each data point
        has_lab_x_res = False
        var_X = np.var(X, ddof=0)

        for xi, yi in zip(X, Y):
            if not has_lab_x_res:
                lab_x_res = r"$Var(X)=$" + format_two_decimal(var_X)
                has_lab_x_res = True
            else:
                lab_x_res = None  # No label for other lines
            ax.plot(
                [E_X, xi],
                [yi, yi],
                color=thm.set1_blue,
                linestyle="--",
                linewidth=1,
                label=lab_x_res,
            )

        # Dashed lines from y_bar to each data point
        has_lab_y_res = False
        var_Y = np.var(Y, ddof=0)

        for xi, yi in zip(X, Y):
            if not has_lab_y_res:
                lab_y_res = r"$Var(Y)=$" + format_two_decimal(var_Y)
                has_lab_y_res = True
            else:
                lab_y_res = None  # No label for other lines

            ax.plot(
                [xi, xi],
                [E_Y, yi],
                color=thm.set1_green,
                linestyle="--",
                linewidth=1,
                label=lab_y_res,
            )

        # Update the y-axis limits after everything has been plotted
        ax.set_ylim(y_lb, y_ub)
        ax.set_xlim([x_lb, x_ub])
        ax.legend(loc="lower left", frameon=True, fontsize="small")

        return fig

    slider_col, chart_col = st.columns((0.75, 1))

    with slider_col:
        st.write("**Choose the following parameters:**")

        E_X_slider = st.slider(
            r"Expectation $E[X]$",
            min_value=-1.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )

        a_slider = st.slider(
            r"Intercept $a$",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
        )
        b_slider = st.slider(
            r"Slope $b$", min_value=-2.0, max_value=2.0, value=1.0, step=0.1
        )

        X, Y = generate_data(a_slider, b_slider, E_X_slider)

        # Button to resample X
        if st.button("Resample X", type="primary"):
            st.session_state.random_seed += 1  # Increment seed by 1
            X, Y = generate_data(a_slider, b_slider, E_X_slider)

    with chart_col:
        st.pyplot(plot_expectation(X, Y, a_slider, b_slider))

    with header_col:
        if b_slider < 0:
            b_str = f"{b_slider:.2f}"
        else:
            b_str = f"+ {b_slider:.2f}"

        var_X_calc = np.var(X, ddof=0)  # ddof=0 ensures we get the population variance
        var_Y_calc = b_slider**2 * var_X_calc

        #         st.markdown(r"""$E[Y] = E[a+bX] = a + bE[X]$""", unsafe_allow_html=True)
        # st.markdown(r"""$Var(Y) = Var(a+bX) = b^2Var(X)$""", unsafe_allow_html=True)

        exp_eqn = (
            rf"$E[Y] = E[a+bX] = a + bE[X] = "
            + r"a + b \frac{\sum_{i=1}^N X_i}{N} ="
            + rf"{a_slider:.2f} {b_str} \times {E_X_slider:.2f} = {a_slider + b_slider * E_X_slider:.2f}$"
        )
        var_eqn = (
            rf"""$Var(Y) = Var(a+bX) = b^2Var(X) = ({b_slider:.2f})^2 """
            + r"\frac{\sum_{i=1}^N(X_i - E[X])^2}{N}"
            + rf"= {b_slider**2:.2f} \times {var_X_calc:.2f} = {var_Y_calc:.2f}$"
            ""
        )

        st.markdown(exp_eqn, unsafe_allow_html=True)
        st.markdown(var_eqn, unsafe_allow_html=True)


# Wooldridge Top 10
with c2:
    jmw_source = "https://docs.google.com/document/d/1GQ3qlD0_cNQdkh_iT-dBDXwNVIXk27WCZtmQ3Sgxz1o/mobilebasic#id.f68ahvvbxdaj"
    jmw_twitter = "https://twitter.com/jmwooldridge"
    tr_source = "https://bookdown.org/ts_robinson1994/10EconometricTheorems/"
    st.write(
        f"Source: Jeffrey Wooldridge [@jmwooldridge]({jmw_twitter}). For his tweets organized by topic check this - [link]({jmw_source})."
    )
    st.write(
        f"Thomas Robinson from LSE elaborate on each of them in this nice e-book - [link]({tr_source})."
    )

    # st.markdown(
    #     "<hr style='margin-top: 0; margin-bottom: 5px;'>",
    #     unsafe_allow_html=True,
    # )

    st.markdown(
        r"""
    <h5>1. Law of Iterated Expectations, Law of Total Variance </h5>
    
    $E[x] = E[E[x|y]]$ <br>
    $var[Y] = E[var[Y|X]] + var(E[Y|X])$

    <h5>2. Linearity of Expectations, Variance of a Sum</h5>

    $E[aX+bY+c] = aE[X] + bE[Y] + c$ <br>
    $var[aX+bY+c] = a^2 var(X) + b^2 var(Y) + 2ab Cov(X,Y)$
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    r"""<h4 style="text-align:center;">Visualize a Simple Example</h4>""",
    unsafe_allow_html=True,
)

st.markdown(
    r"""
    Consider a random variable $X$ and a random variable $Y = a + bX$, i.e., a linear function of $X$.<br>
    Move the sliders and observe how $Var(Y)$ only depends on $b$, but not on $E[X]$ or $a$,
    whereas $E[Y]$ changes linearly based on the combination of all three parameters.
    Also note that $E[Y]$ does not depend on $b$ if $E[X] = 0$ and vice-versa.""",
    unsafe_allow_html=True,
)

render_lin_exp()

s1, c3, s2 = utl.wide_col()

with c3:
    st.markdown(
        r"""
    <h5>3. Jensen's Inequality, Chebyshev's Inequality</h5>

    Jensen:
    $\varphi(\operatorname{E}[X]) \leq \operatorname{E} \left[\varphi(X)\right]$, where $X$ is a RV and $\varphi$ is convex <br>
    Chebyshev:
    $\Pr(|X-\mu|\geq k\sigma) \leq \frac{1}{k^2}$, where $X$ is a RV, $σ^2$ is finite, and $k>0$

    <h5>4. Linear Projection and its Properties</h5>

    $\mathbf{P} \equiv \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ and $\mathbf{M} \equiv \mathbf{I}_n - \mathbf{P}$ <br> 
    $\mathbf{P}\mathbf{X} = \mathbf{X}$ ("projection") and $\mathbf{M}\mathbf{X} = \mathbf{0}$ ("annihilator") <br>
    Also $\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}$ ("hat") and $\mathbf{e} = \mathbf{M}\mathbf{y}$ ("residual maker") <br>

    <h5>5. Weak Law of Large Numbers, Central Limit Theorem</h5>

    Weak LLN: $\lim_{n\to\infty}\Pr\!\left(\,|\overline{X}_n-\mu| < \varepsilon\,\right) = 1$, where $\varepsilon > 0$<br>
    CLT: $\sqrt{n}\left(\bar{X}_n - \mu\right)\ \xrightarrow{d}\ \mathcal{N}\left(0,\sigma^2\right)$

    <h5>6. Slutksy's Theorem, Continuous Convergence Theorem, Asymptotic Equivalence Lemma</h5>

    Slutsky: If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{p} c$ constant, then $X_n + Y_n \xrightarrow{d} X + c$ and $Y_nX_n \xrightarrow{d} cX$<br>
    CMT: If $X_n \xrightarrow{d} X$ and $g$ continuous, then $g(X_n) \xrightarrow{d} g(X)$ <br>
    Asymptotic equivalence: $X_n$ and $Y_n$ are asymptotically equivalent if $X_n - Y_n \xrightarrow{p} 0$
    
    <h5>7. Big Op, Little op, and the algebra of them</h5>

    Given $X_n$ is a RV, $a_n$ is constant, $\varepsilon >0$:<br>
    $X_n = O_p(a_n)$ means $P(|\frac{X_n}{a_n}| > \delta) < \epsilon, \forall n > N$<br>
    If $X_n = o_p(a_n)$, then $\frac{x_n}{a_n} = o_p(1)$, meaning $\lim_{n\to\infty} (P|\frac{X_n}{a_n}| \geq \epsilon) = 0, \forall\epsilon > 0$
    or in other words $\frac{X_n}{a_n} \xrightarrow{p} 0$<br>
    $o_p(a_n)$ implies $O_p(a_n)$ but not vice-versa.

    <h5>8. Delta Method</h5>

    Start with $X_n \sim N(\mu,\frac{\sigma^2}{n})$ and get $\frac{\sqrt{n}(X_n - \mu)}{\sigma} \sim N(0,1)$ <br>
    If $g$ is smooth then by Delta Method:<br>
    $\frac{\sqrt{n}(g(X_n) - g(\mu))}{|g'(\mu)|\sigma} \approx N(0,1)$ 
    and $g(X_n) \approx N\left(g(\mu), \frac{g'(\mu)^2\sigma^2}{n}\right)$<br>
    Note the approximation because $g(X_n)$ is an infinite sum.


    <h5>9. Frisch-Waugh Partialling Out</h5>

    Take $\mathbf{y}$ and two sets of regressors $\mathbf{X}_1$ and $\mathbf{X}_2$.
    FWL theorem claims that $\mathbf{b}_2$ will be equal after estimating in the following two forms:<br>
    $\mathbf{y} = \mathbf{X}_1 \mathbf{b}_1 + \mathbf{X}_2 \mathbf{b}_2 + \mathbf{e}$,
            then $\mathbf{b}_2 = (\mathbf{X}_2'\mathbf{X}_2)^{-1}\mathbf{X}_2'\mathbf{y}$ <br>
    $\mathbf{y}^* = \mathbf{X}^*_2 \mathbf{b}_2 + \mathbf{e}$,
        then $\mathbf{b}_2 = (\mathbf{X}^{* \prime}_2 \mathbf{X}^{*}_2 )^{-1} \mathbf{X}^{* \prime}_2 \mathbf{y}^*$, 
            where $\mathbf{X}^*_2 = \mathbf{M}_1 \mathbf{X}_2$ and $\mathbf{y}^* = \mathbf{M}_1 \mathbf{y}$<br>
    Explanation for notation in the second form:<br>
        i. Regress $\mathbf{y}$ on $\mathbf{X}_1$ and get the residuals $\mathbf{M}_1 \mathbf{y}$<br>
        ii. Regress $\mathbf{X}_2$ on $\mathbf{X}_1$ and get the residuals $\mathbf{M}_{1} \mathbf{X}_2$ <br>
        iii. Regress $\mathbf{M}_1 \mathbf{y}$ (residuals from i.) on $\mathbf{M}_1 \mathbf{X}_2$ (residuals from ii.), which will give $\mathbf{b}_2$ <br>


    <h5>10. For PD matrices A and B, A - B is PSD if and only if B^(-1) - A^(-1) is PSD: </h5>

    Sketch of the proof, "=>" direction:<br>
    Take arbitrary non-zero vector $x$, then $x' (A-B)x \geq 0$ and $A \geq B$<br>
    Then "divide" both sides by AB to get $A^{-1}AB^{-1} \geq A^{-1}BB^{-1}$ and then $B^{-1} \geq A^{-1}$ <br>
    Then $x' B^{-1} \geq x'A^{-1}$ and then $x' (B^{-1} - A^{-1})x \geq 0$<br>
    "<=" direction: replace $A$ with $B^{-1}$ and $B$ with $A^{-1}$ and repeat the same steps<br>

        """,
        unsafe_allow_html=True,
    )

    st.markdown(r"""<h4> Bonus (not Wooldridge)</h4>""", unsafe_allow_html=True)
    st.markdown(r"""11. **Concepts of Independence**""", unsafe_allow_html=True)

    with st.expander("Click to expand"):
        st.markdown(
            r"""
      
        Independence: $ P(X \cap Y) = P(X)P(Y) $ or $ P(X = x, Y = y) = P(X = x)P(Y = y) \; \forall x, y$. <br>
        Mean Independence: $ E(X | Y = y) = E(X) \; \forall y$.<br>
        Orthogonality: $\langle X, Y \rangle  = E[XY]=0$, but $\langle X, Y \rangle$ often defined as $Cov(X,Y)$, then
        $\langle X, Y \rangle = Cov(X,Y) = E[(X - E(X))(Y - E(Y))] = 0 $<br>
        Conditional Independence: $ P(X = x, Y = y | Z = z) = P(X = x | Z = z)P(Y = y | Z = z) \; \forall x, y, z$.<br>
        Linear Independence (in linear algebra):<br>
        Vectors $ \mathbf{v_1}, \mathbf{v_2}, ... \mathbf{v_n} $ are linearly independent if and only if no vector is zero and no vector can be expressed as a linear combination of the others.<br>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        r"""
        12. **Concepts of Convergence**<br>""",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        st.markdown(
            r"""
        Let $X$ be a random variable and $X_n$ a sequence of random variables.<br>
        i. Convergence in probability $X_n \xrightarrow{P} X$ if $ \lim_{n \to \infty} P(|X_n - X| > \epsilon) = 0 \; \forall \epsilon > 0$ <br>
        ii. Convergence almost surely: $X_n \xrightarrow{a.s.} X$ if $P(\lim_{n \to \infty} X_n = X) = 1$ <br>
        iii. Convergence in distribution: $X_n \xrightarrow{d} X$ if $\lim_{n \to \infty} F_{X_n}(x) = F_X(x) \; \forall x \in \mathbb{R}$ at which $F_X$ is continuous.<br>
        $ i. \implies ii. \implies iii.$ 

        """,
            unsafe_allow_html=True,
        )
