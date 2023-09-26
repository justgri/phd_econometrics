import streamlit as st
from st_pages import Page, add_page_title, show_pages

st.set_page_config(
    page_title="PhD Econometrics",
    page_icon="👋",
)

# add_page_title()

show_pages(
    [
        Page("homepage.py", "Introduction and Top 10", "🏠"),
        Page("pages/week_1_ols.py", "Week 1 - OLS", "📖"),
    ]
)

st.title("Econometrics for PhD Students")
st.sidebar.success("Select a page above.")

st.markdown(
    "Trying to learn and enjoy the first year Econ PhD. <br> All mistakes are my own.",
    unsafe_allow_html=True,
)


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
    """
1. Law of Iterated Expectations, Law of Total Variance <br>
$E[x] = E[E[x|y]]$ <br>

2. Linearity of Expectations, Variance of a Sum <br>
$var[aX+bY+c] = a^2 var(X) + b^2 var(Y)$ 

3. Jensen's Inequality, Chebyshev's Inequality

4. Linear Projection and Its Properties

5. Weak Law of Large Numbers, Central Limit Theorem

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
