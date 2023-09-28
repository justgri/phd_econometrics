import streamlit as st
from st_pages import Page, add_page_title, show_pages, show_pages_from_config

import src.scripts.plotly_themes
import src.scripts.utils as utl

st.set_page_config(
    page_title="PhD Econometrics",
    page_icon="👋",
    layout="wide",
)

utl.local_css("src/styles/styles_home.css")

show_pages_from_config()


s1, c1, c2 = utl.wide_col()

with c1:
    st.title("Econometrics for PhD Students")
    st.sidebar.success("Select a page above.")

    st.markdown(
        "Trying to learn and enjoy the first year of Econ PhD. <br> Procrastinating productively. <br> All mistakes are my own.",
        unsafe_allow_html=True,
    )

    st.write(
        "Disclaimer: this might not look like PhD level stuff, because PhD and undergrad topics are largely overlapping, only the level of theoretical rigor differs.",
        "My goal is also to master the fundamentals rather than scratch the surface of more advanced topics.",
    )


s1, c2, s2 = utl.narrow_col()

with c2:
    st.markdown(
        "<h3 style='text-align: center'>Top 10 Things to Know by Wooldridge</h3>",
        unsafe_allow_html=True,
    )

    jmw_source = "https://docs.google.com/document/d/1GQ3qlD0_cNQdkh_iT-dBDXwNVIXk27WCZtmQ3Sgxz1o/mobilebasic#id.f68ahvvbxdaj"
    jmw_twitter = "https://twitter.com/jmwooldridge"
    st.write(
        f"Source: Jeffrey Wooldridge [@jmwooldridge]({jmw_twitter}), for tweets organized by topic - [link]({jmw_source})."
    )

    st.markdown(
        r"""
    1. Law of Iterated Expectations, Law of Total Variance <br>
    $E[x] = E[E[x|y]]$ <br>

    2. Linearity of Expectations, Variance of a Sum <br>
    $var[aX+bY+c] = a^2 var(X) + b^2 var(Y)$ 

    3. Jensen's Inequality, Chebyshev's Inequality <br>
    Jensen:<br>
    $\varphi(\operatorname{E}[X]) \leq \operatorname{E} \left[\varphi(X)\right]$, where $X$ is a RV and $\varphi$ is convex. <br> <br>
    Chebyshev:<br>
    $\Pr(|X-\mu|\geq k\sigma) \leq \frac{1}{k^2}$, where $X$ is a RV, $σ^2$ is finite, and $k>0$.

    4. Linear Projection and Its Properties

    5. Weak Law of Large Numbers, Central Limit Theorem <br>
    Weak LLN: $\lim_{n\to\infty}\Pr\!\left(\,|\overline{X}_n-\mu| < \varepsilon\,\right) = 1$, where $\varepsilon > 0$<br>
    CLT: $\sqrt{n}\left(\bar{X}_n - \mu\right)\ \xrightarrow{d}\ \mathcal{N}\left(0,\sigma^2\right)$

    6. Slutksy's Theorem, Continuous Convergence Theorem, Asymptotic Equivalence Lemma

    7. Big Op, Little op, and the algebra of them.

    8. Delta Method

    9. Frisch-Waugh Partialling Out

    10. For PD matrices A and B, A - B is PSD if and only if B^(-1) - A^(-1) is PSD.

    Additional (not Wooldridge) : <br>

    11. Concepts of Independence
    12. Projections
    """,
        unsafe_allow_html=True,
    )
    # \lim_{n\to\infty}\Pr\!\left(\,|\overline{X}_n-\mu| < \varepsilon\,\right) = 1.
