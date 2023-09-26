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

st.write("Just for learning purposes.")


st.markdown(
    "<h3 style='text-align: center'>Top 10 Things to Know by Wooldridge</h3>",
    unsafe_allow_html=True,
)

st.write("Source: Jeffrey Wooldridge (@jmwooldridge)")

st.markdown(
    """
1. Linearity of Expectation, Variance of Sum <br>
$var[aX+bY+c] = a^2 var(X) + b^2 var(Y)$
2. Law of Iterated Expectations, Law of Total Variance <br>
$E[x] = E[E[x|y]]$ <br>
3. Law of Large Numbers, Central Limit Theorem
4. Slutsky's Theorem
5. Continuous Mapping Theorem
6. Delta method
7. Frisch–Waugh–Lovell Theorem
8. Concepts of Independence
9. Inequalities
10. Linear Projection
""",
    unsafe_allow_html=True,
)
