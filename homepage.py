import streamlit as st
from st_pages import Page, add_page_title, show_pages, show_pages_from_config

import scripts.plot_themes
import src.scripts.utils as utl

st.set_page_config(
    page_title="PhD Econometrics",
    page_icon="👋",
    layout="wide",
)

utl.local_css("src/styles/styles_home.css")
utl.external_css(
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
)
show_pages_from_config()

s1, c1, c2 = utl.wide_col()

# my LinkedIn and GitHub
linkedin_url = "https://www.linkedin.com/in/justinas-grigaitis/"
github_url = "https://github.com/justgri"

# Intro
with c1:
    st.title("Econometrics for PhD Students")
    st.sidebar.success("Select a page above.")

    st.markdown(
        "Trying to learn and enjoy the first year of Econ PhD. <br> Procrastinating productively. <br> All mistakes are my own.",
        unsafe_allow_html=True,
    )

    st.markdown(
        """**Disclaimer:** <br>
        This website does not represent the official curriculum taught at my university. <br>
        My goal is to master the fundamentals of a few topics rather than scratch the surface of many. <br>
        It might not even look like PhD level stuff, because topics are largely overlapping with the undergraduate course. <br>
        """,
        # Main difference is matrix algebra and proving everything along the way, which might not always be included here.
        # Hopefully it will give insights to both PhD students, undergrads, and others.
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        Please send me comments: 
    <a href="{linkedin_url}" target="_blank">
        <i class="fab fa-linkedin fa-lg"></i>
    </a>
    <a href="{github_url}" target="_blank">
        <i class="fab fa-github fa-lg"></i>
    </a>
    """,
        unsafe_allow_html=True,
    )


s1, c2, s2 = utl.narrow_col()

# Textbooks
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Reference Textbooks</h3>",
        unsafe_allow_html=True,
    )

    c2_1, s2_1, c2_2 = st.columns((1, 0.05, 1))

    with c2_1:
        st.image("src/images/intro_stock_watson.jpg", width=350)

    with c2_2:
        st.image("src/images/intro_greene.jpg", width=350)


# Other references
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Other References</h3>",
        unsafe_allow_html=True,
    )

    st.link_button(
        "NYU Lecture Notes by William H. Greene",
        "https://pages.stern.nyu.edu/~wgreene/Econometrics/Notes.htm",
        type="secondary",
    )

    st.link_button(
        "MIT Lecture Notes by Victor Chernozhukov",
        "https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/pages/lecture-notes/",
        type="secondary",
    )

    st.link_button(
        "TA notes by Matteo Courthoud",
        "https://matteocourthoud.github.io/course/metrics/",
        type="secondary",
    )

    st.link_button(
        "Graduate Econometrics YouTube playlist by Ben Lambert",
        "https://www.youtube.com/watch?v=GMVh02WGhoc&list=PLwJRxp3blEvaxmHgI2iOzNP6KGLSyd4dz",
        type="secondary",
    )

    st.link_button(
        "Intro to Econometrics with R by Professors at University of Duisburg-Essen",
        "https://www.econometrics-with-r.org/",
        type="secondary",
    )


# Wooldridge Top 10
with c2:
    st.markdown(
        "<h3 style='text-align: center'>10 Tools for Econometrics by Jeffrey Wooldridge</h3>",
        unsafe_allow_html=True,
    )

    jmw_source = "https://docs.google.com/document/d/1GQ3qlD0_cNQdkh_iT-dBDXwNVIXk27WCZtmQ3Sgxz1o/mobilebasic#id.f68ahvvbxdaj"
    jmw_twitter = "https://twitter.com/jmwooldridge"
    tr_source = "https://bookdown.org/ts_robinson1994/10EconometricTheorems/"
    st.write(
        f"Source: Jeffrey Wooldridge [@jmwooldridge]({jmw_twitter}), for tweets organized by topic - [link]({jmw_source})."
    )
    st.write(
        f"Thomas Robinson from LSE elaborate on each of them in this nice e-book - [link]({tr_source})."
    )

    with st.expander("Click to see Top 10 list below.", expanded=True):
        st.markdown(
            r"""
        1. **Law of Iterated Expectations, Law of Total Variance** <br>
        $E[x] = E[E[x|y]]$ <br>

        2. **Linearity of Expectations, Variance of a Sum** <br>
        $var[aX+bY+c] = a^2 var(X) + b^2 var(Y)$ 

        3. **Jensen's Inequality, Chebyshev's Inequality** <br>
        Jensen:<br>
        $\varphi(\operatorname{E}[X]) \leq \operatorname{E} \left[\varphi(X)\right]$, where $X$ is a RV and $\varphi$ is convex. <br> <br>
        Chebyshev:<br>
        $\Pr(|X-\mu|\geq k\sigma) \leq \frac{1}{k^2}$, where $X$ is a RV, $σ^2$ is finite, and $k>0$.

        4. **Linear Projection and its Properties**

        5. **Weak Law of Large Numbers, Central Limit Theorem** <br>
        Weak LLN: $\lim_{n\to\infty}\Pr\!\left(\,|\overline{X}_n-\mu| < \varepsilon\,\right) = 1$, where $\varepsilon > 0$<br>
        CLT: $\sqrt{n}\left(\bar{X}_n - \mu\right)\ \xrightarrow{d}\ \mathcal{N}\left(0,\sigma^2\right)$

        6. **Slutksy's Theorem, Continuous Convergence Theorem, Asymptotic Equivalence Lemma**

        7. **Big Op, Little op, and the algebra of them**

        8. **Delta Method**

        9. **Frisch-Waugh Partialling Out**

        10. **For PD matrices A and B, A - B is PSD if and only if B^(-1) - A^(-1) is PSD**

        Additional (not Wooldridge) : <br>

        11. **Concepts of Independence**
        """,
            unsafe_allow_html=True,
        )
