import streamlit as st
from st_pages import Page, add_page_title, show_pages, show_pages_from_config

import src.scripts.plot_themes
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
        # st.image("src/images/intro_stock_watson.jpg", width=350)
        st.image("src/images/intro_hayashi.jpg", width=350)

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

with c2:
    st.markdown(
        "<h3 style='text-align: center'>What is Econometrics?</h3>",
        unsafe_allow_html=True,
    )

    econometrica_public = "https://www.sv.uio.no/econ/om/tall-og-fakta/nobelprisvinnere/ragnar-frisch/published-scientific-work/rf-published-scientific-works/rf1933c.pdf"
    econometrica_jstor = "https://www.jstor.org/stable/i332704"

    st.markdown(
        r"""
        "\[E\]conometrics is by no means the same as economic statistics.
        Nor is it identical with what we call general economic theory,
        although a considerable portion of this theory has a definitely quantitative character.
        Nor should econometrics be taken as synonomous with the application of mathematics to economics.
        Experience has shown that each of these three view-points, that of statistics, economic theory, and mathematics, is a necessary, but not by itself a sufficient,
        condition for a real understanding of the quan- titative relations in modern economic life.
        **It is the unification of all three that is powerful.
        And it is this *unification* that constitutes econometrics.**"
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""Frisch, Ragnar. “Editor’s Note.” Econometrica, vol. 1, no. 1, 1933, pp. 1–4. JSTOR, http://www.jstor.org/stable/1912224. Accessed 1 Oct. 2023. (free access [link]({econometrica_public}))""",
        unsafe_allow_html=True,
    )

# Preliminary ToC

with c2:
    st.markdown(
        "<h3 style='text-align: center'>Tentative Table of Contents</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand", expanded=False):
        st.write(
            "Chapters follow Hayahshi *Econometrics* (1st ed.). Chapters from Greene *Eceonometric Analysis* (8th ed.) given in the parentheses."
        )
        st.write(
            "Subsections are likely to change depending on which topics I find most interesting or challenging."
        )

        st.markdown(
            r"""
    <div class="numbered-header">
        <b>Chapter 1: Finite-Sample Properties of OLS</b><br>
    </div>
        
    <div class="numbered">
        1. OLS algebra (Greene Ch 2 - 3.2) <br>
        2. FWL and Goodness of fit measures (Greene Ch 3.3 - 4) <br>
        3. Hypothesis testing (Greene Ch 5)
        4. Omitted Variable Bias (Greene Ch 4)<br>
        
    </div>

    <br>

    <div class="numbered-header">
        <b>Chapter 2: Large-Sample Theory</b><br>
    </div>

    <div class="numbered">
        5. Limit theorems, "delta-method" <br>
        6. Law of large numbers <br>
        7. Large sample OLS properties <br>
    </div>

    <br>

    <div class="numbered-header">
        <b>Chapter 3: Generalized Method of Moments</b><br>
    </div>

    <div class="numbered">
        8. Endogeneity bias <br>
        9. Instrumental Variables <br>
        10. IV-related tests <br>
        11. GMM properties
    </div>

    <br>
    
    <div class="numbered-header">
        <b>Chapter 4: Multiple-Equation GMM</b>
    </div>

    <div class="numbered">
        12. TBD <br>
        13. TBD <br>
    </div>

    <br>

    Next semester - Panel Data, Time Series, Cointegration, and MLE. <br>
    Bonus if time permits (it never does) - Monte Carlo, bootstrapping, gradient descent, causal ML, etc.
    """,
            unsafe_allow_html=True,
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

    with st.expander("Click to expand", expanded=True):
        st.markdown(
            r"""
        1. **Law of Iterated Expectations, Law of Total Variance** <br>
        $E[x] = E[E[x|y]]$ <br>
        $var[Y] = E[var[Y|X]] + var(E[Y|X])$

        2. **Linearity of Expectations, Variance of a Sum** <br>
        $E[aX+bY+c] = aE[X] + bE[Y] + c$ <br>
        $var[aX+bY+c] = a^2 var(X) + b^2 var(Y)$ 

        3. **Jensen's Inequality, Chebyshev's Inequality** <br>
        Jensen:
        $\varphi(\operatorname{E}[X]) \leq \operatorname{E} \left[\varphi(X)\right]$, where $X$ is a RV and $\varphi$ is convex <br>
        Chebyshev:
        $\Pr(|X-\mu|\geq k\sigma) \leq \frac{1}{k^2}$, where $X$ is a RV, $σ^2$ is finite, and $k>0$

        4. **Linear Projection and its Properties** <br>
        $\mathbf{P} \equiv \mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$ and $\mathbf{M} \equiv \mathbf{I}_n - \mathbf{P}$ <br> 
        $\mathbf{P}\mathbf{X} = \mathbf{X}$ ("projection") and $\mathbf{M}\mathbf{X} = \mathbf{0}$ ("annihilator") <br>
        Also $\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}$ ("hat") and $\hat{\mathbf{e}} = \mathbf{M}\mathbf{y}$ ("residual maker") <br>

        5. **Weak Law of Large Numbers, Central Limit Theorem** <br>
        Weak LLN: $\lim_{n\to\infty}\Pr\!\left(\,|\overline{X}_n-\mu| < \varepsilon\,\right) = 1$, where $\varepsilon > 0$<br>
        CLT: $\sqrt{n}\left(\bar{X}_n - \mu\right)\ \xrightarrow{d}\ \mathcal{N}\left(0,\sigma^2\right)$

        6. **Slutksy's Theorem, Continuous Convergence Theorem, Asymptotic Equivalence Lemma**<br>
        Slutsky: If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{p} c$ constant, then $X_n + Y_n \xrightarrow{d} X + c$ and $Y_nX_n \xrightarrow{d} cX$<br>
        CMT: If $X_n \xrightarrow{d} X$ and $g$ continuous, then $g(X_n) \xrightarrow{d} g(X)$ <br>
        Asymptotic equivalence: $X_n$ and $Y_n$ are asymptotically equivalent if $X_n - Y_n \xrightarrow{p} 0$
        
        7. **Big Op, Little op, and the algebra of them** <br>
        Given $X_n$ is a RV, $a_n$ is constant, $\varepsilon >0$:<br>
        $X_n = O_p(a_n)$ means $P(|\frac{X_n}{a_n}| > \delta) < \epsilon, \forall n > N$<br>
        If $X_n = o_p(a_n)$, then $\frac{x_n}{a_n} = o_p(1)$, meaning $\lim_{n\to\infty} (P|\frac{X_n}{a_n}| \geq \epsilon) = 0, \forall\epsilon > 0$
        or in other words $\frac{X_n}{a_n} \xrightarrow{p} 0$<br>
        $o_p(a_n)$ implies $O_p(a_n)$ but not vice-versa.

        8. **Delta Method**<br>
        Start with $X_n \sim N(\mu,\frac{\sigma^2}{n})$ and get $\frac{\sqrt{n}(X_n - \mu)}{\sigma} \sim N(0,1)$ <br>
        If $g$ is smooth then by Delta Method $\frac{\sqrt{n}(g(X_n) - g(\mu))}{|g'(\mu)|\sigma} \approx N(0,1)$ 
        and $g(X_n) \approx N\left(g(\mu), \frac{g'(\mu)^2\sigma^2}{n}\right)$<br>
        Note the approximation because $g(X_n)$ is an infinite sum


        9. **Frisch-Waugh Partialling Out**<br>
        "It's the residual variance that matters" <br>
        Take $\mathbf{y}$ and two sets of regressors $\mathbf{X}_1$ and $\mathbf{X}_2$.
        FWL theorem claims that $\mathbf{b}_2$ will be equal after estimating in the following two forms:<br>
        A. $\mathbf{y} = \mathbf{X}_1 \mathbf{b}_1 + \mathbf{X}_2 \mathbf{b}_2 + \mathbf{e}$,
                then $\mathbf{b}_2 = (\mathbf{X}_2'\mathbf{X}_2)^{-1}\mathbf{X}_2'\mathbf{y}$ <br>
        B. $\mathbf{y}^* = \mathbf{X}^*_2 \mathbf{b}_2 + \mathbf{e}$,
          then $\mathbf{b}_2 = (\mathbf{X}^{* \prime}_2 \mathbf{X}^{*}_2 )^{-1} \mathbf{X}^{* \prime}_2 \mathbf{y}^*$, 
                where $\mathbf{X}^*_2 = \mathbf{M}_1 \mathbf{X}_2$ and $\mathbf{y}^* = \mathbf{M}_1 \mathbf{y}$<br>
        Explanation for B:<br>
         i. Regress $\mathbf{y}$ on $\mathbf{X}_1$ and get the residuals $\mathbf{M}_1 \mathbf{y}$<br>
         ii. Regress $\mathbf{X}_2$ on $\mathbf{X}_1$ and get the residuals $\mathbf{M}_{1} \mathbf{X}_2$ <br>
         iii. Regress $\mathbf{M}_1 \mathbf{y}$ (residuals from i.) on $\mathbf{M}_1 \mathbf{X}_2$ (residuals from ii.), which will give $\mathbf{b}_2$ <br>
  
    
        10. **For PD matrices** $\mathbf{A}$ **and** $\mathbf{B}$, $\mathbf{A} - \mathbf{B}$ **is PSD if and only if**
         $\mathbf{B}^{-1} - \mathbf{A}^{-1}$ **is PSD** <br>
         Sketch of the proof, "=>" direction:<br>
         Take arbitrary non-zero vector $x$, then $x' (A-B)x \geq 0$ and $A \geq B$<br>
         Then "divide" both sides by AB to get $A^{-1}AB^{-1} \geq A^{-1}BB^{-1}$ and then $B^{-1} \geq A^{-1}$ <br>
         Then $x' B^{-1} \geq x'A^{-1}$ and then $x' (B^{-1} - A^{-1})x \geq 0$<br>
         "<=" direction: replace $A$ with $B^{-1}$ and $B$ with $A^{-1}$ and repeat the same steps<br>


        Bonus (not Wooldridge) : <br>

        11. **Concepts of Independence**

        12. **Concepts of Convergence**
        """,
            unsafe_allow_html=True,
        )
