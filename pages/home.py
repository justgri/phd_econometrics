import streamlit as st

import src.scripts.plot_themes
import src.scripts.utils as utl

utl.local_css("src/styles/styles_home.css")
utl.external_css(
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"
)

s1, c1, c2 = utl.wide_col()

# my LinkedIn, GitHub, and email
linkedin_url = "https://www.linkedin.com/in/justinas-grigaitis/"
github_url = "https://github.com/justgri"
email_url = "mailto:justinas.grigaitis@econ.uzh.ch"

# Intro
with c1:
    # Title iterations
    # Econometrics for PhD Students
    # PhD Econometrics for Everyone
    # PhD Econometrics for All
    # PhD for All: Econometrics

    st.title("PhD for All: Econometrics")

    # Subheader iterations
    # Intuitive visuals. Rigorous theory. Reproducible code.
    # Intuitive visuals. Rigorous theory. Simple code.
    # Intuitive visuals. Rigorous theory. Easy code.
    # Interactive visuals. Rigorous theory. Simple code.

    st.markdown(
        '<span style="font-size: 28px; display: block; margin-bottom: 5px;">*Interactive visuals. Rigorous theory. Simple code.*</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='margin-top: 0; margin-bottom: 5px;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""Learning and helping others learn along the way.<br>
            Explaining PhD concepts intuitively.<br>
            Procrastinating productively.""",
        unsafe_allow_html=True,
    )
    st.markdown(
        r"""Targeted at **grad students**, but useful for **professionals** and **undergrads** alike.""",
        unsafe_allow_html=True,
    )

    st.markdown("My other Econ PhD apps:")

    but_col1, but_col2, _ = st.columns((1, 1, 2))

    but_col1.link_button(
        "PhD Macroeconomics",
        "https://phd-macroeconomics.streamlit.app/",
        type="secondary",
    )

    but_col2.link_button(
        "PhD Microeconomics",
        "https://phd-microeconomics.streamlit.app/",
        type="secondary",
    )

    st.markdown(
        f"""
        Please send me feedback:<br>
    <a href="{linkedin_url}" target="_blank">
        <i class="fab fa-linkedin fa-lg"></i>
    </a>
    <a href="{email_url}" target="_blank">
        <i class="fas fa-envelope fa-lg"></i>
    </a>
    <a href="{github_url}" target="_blank">
        <i class="fab fa-github fa-lg"></i>
    </a>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""<u>**Disclaimer:**</u> <br>
        This website does not represent the official curriculum taught at my university. <br>
        My goal is to cover fewer topics in greater depth rather than scratch the surface of many. <br>
        All mistakes are my own.
        """,
        # Main difference is matrix algebra and proving everything along the way, which might not always be included here.
        unsafe_allow_html=True,
    )


_, note_col, _ = st.columns((0.02, 1, 1.5))
with note_col:
    with st.expander("For mobile users:"):
        st.write("Sidebar leads to other pages within this app.")
        st.image("src/images/intro_mobile_sidebar.png", width=200)

s1, c2, s2 = utl.wide_col()

# Preliminary ToC
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Table of Contents</h3>",
        unsafe_allow_html=True,
    )

    with st.expander("Click to expand"):
        # Page links - potentially hrefs with st.experimental_set_query_params()
        path_tools = "https://phd-econometrics.streamlit.app/Must-know"
        path_ols = "https://phd-econometrics.streamlit.app/Linear%20Regression"
        path_ols_3d = "https://phd-econometrics.streamlit.app/OLS%20in%203D"
        path_fit = "https://phd-econometrics.streamlit.app/Fit%20Measures"

        st.markdown(
            r"""
            Hyperlinks lead to the corresponding pages on this website.""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""[**Top 10 theory tools**]({path_tools})"""
            + r""" **that everyone should know according to Jeffrey Wooldridge**
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            rf"""
        <div class="numbered-header">
            <b>Chapter 1: Finite-Sample Properties of OLS</b><br>
        </div>
        
        <div class="numbered">
            1. <a href="{path_ols}" target="_blank">OLS estimation</a> (Greene Ch 2 - 3.2)<br>
            2. <a href="{path_ols_3d}" target="_blank">OLS visualization in 3D</a> (Greene Ch 2 - 3.2)<br>
            3. <a href="{path_fit}" target="_blank">Fit measures</a> (Greene Ch 3.5 and Ch 5.8) <br>
            4. Data problems (OVB, measurement error, missing data - Greene Ch. 4.9)<br>
            5. Hypothesis testing (Greene Ch 5) <br>
            6. Functional form (Greene Ch 6.5)<br>
            7. PCA (Greene Ch 4.9)
        </div>

        <br>

        <div class="numbered-header">
            <b>Chapter 2: Large-Sample Theory</b><br>
        </div>

        <div class="numbered">
            8. Limit theorems, "delta-method" <br>
            9. Law of large numbers <br>
            10. Large sample OLS properties <br>
        </div>

        <br>

        <div class="numbered-header">
            <b>Chapter 3: Generalized Method of Moments</b><br>
        </div>

        <div class="numbered">
            11. Endogeneity <br>
            12. Instrumental Variables <br>
            13. IV-related tests <br>
            14. GMM properties
        </div>

        <br>
        
        <div class="numbered-header">
            <b>Chapter 4: Multiple-Equation GMM</b>
        </div>

        <div class="numbered">
            15. TBD <br>
            16. TBD <br>
        </div>

        <br>

        
        
        Next semester - Panel Data, Time Series, Cointegration, and MLE. <br>
        Bonus data science - causal ML, deep learning, classification algorithms, etc.

        Section headers follow Hayahshi *Econometrics* (1st ed.).<br>
        Chapters from Greene *Eceonometric Analysis* (8th ed.) given in the parentheses.<br>
        Subsections are likely to change depending on which topics I find most interesting or challenging.<br>

        """,
            unsafe_allow_html=True,
        )


with c2:
    st.markdown(
        "<h3 style='text-align: center'>What is Econometrics?</h3>",
        unsafe_allow_html=True,
    )

    econometrica_public = "https://www.sv.uio.no/econ/om/tall-og-fakta/nobelprisvinnere/ragnar-frisch/published-scientific-work/rf-published-scientific-works/rf1933c.pdf"
    econometrica_jstor = "https://www.jstor.org/stable/i332704"

    with st.expander("Click to expand"):
        st.markdown(
            r"""
            "\[E\]conometrics is by no means the same as economic statistics.
            Nor is it identical with what we call general economic theory,
            although a considerable portion of this theory has a definitely quantitative character.
            Nor should econometrics be taken as synonomous with the application of mathematics to economics.
            Experience has shown that each of these three view-points, that of statistics, economic theory, and mathematics, is a necessary, but not by itself a sufficient,
            condition for a real understanding of the quan- titative relations in modern economic life.
            **It is the unification of all three that is powerful.
            And it is this *unification* that constitutes econometrics.**" (emphasis added)
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""Frisch, Ragnar. “Editor’s Note.” Econometrica, vol. 1, no. 1, 1933, pp. 1–4. JSTOR, http://www.jstor.org/stable/1912224. Accessed 1 Oct. 2023. (free access [link]({econometrica_public}))""",
            unsafe_allow_html=True,
        )


# Textbooks
with c2:
    st.markdown(
        "<h3 style='text-align: center'>Reference Textbooks</h3>",
        unsafe_allow_html=True,
    )

    c2_1, s2_1, c2_2 = st.columns((1, 0.05, 1))

    with c2_1:
        # st.image("src/images/intro_stock_watson.jpg", width=350)
        st.image("src/images/intro_hayashi.jpg", width=300)

    with c2_2:
        st.image("src/images/intro_greene.jpg", width=300)


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
        "An Introduction to Statistical Learning with R/Python (free textbooks)",
        "https://www.statlearning.com/",
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

    st.link_button(
        "Intro to Econometrics with R by Professors at Sciences Po - more focus on practical examples and intuition",
        "https://scpoecon.github.io/ScPoEconometrics/index.html",
        type="secondary",
    )

    st.link_button(
        "Intro to Econometrics with Streamlit by Florian Chávez (coreso.ch)",
        "https://econometric-illustrations.streamlit.app/",
        type="secondary",
    )
