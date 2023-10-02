import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from st_pages import add_page_title

import src.scripts.plot_themes as thm
import src.scripts.utils as utl

### PAGE CONFIGS ###

st.set_page_config(layout="wide")
utl.local_css("src/styles/styles_pages.css")

# create one column with consistent width
_, col_top, _ = utl.wide_col()

### PAGE INTRO ###

with col_top:
    st.title("Lecture Title")
    st.header("Lecture Header with more details")

    st.write("Brief description of the page contents.")

### START OF CONTENT ###


### VISUALS ###
_, c1, _ = utl.wide_col()

with c1:
    st.markdown(
        "<h3 style='text-align: left'> 1. Visual part with visuals</h3>",
        unsafe_allow_html=True,
    )

    st.write(r"This is some text about what's below - general idea.")
    st.write(r"It will contain some plot or tables.")

_, slider_col, _ = st.columns(3)

with slider_col:
    st.markdown(
        "<h4 style='text-align: left'>Choose parameters</h3>",
        unsafe_allow_html=True,
    )

    lam_cust = st.slider(
        r"Risk aversion parameter, $\lambda$",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
    )


def plot_utility(lam):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.subplots_adjust(left=0)  # remove margin

    x = np.linspace(0, 3, 100)

    ax.plot(
        x,
        x**lam,
        color=thm.cols_set1_plt[1],
        label=r"$u(c) = c^\lambda$",
    )
    plt.xlabel(r"$c$")
    plt.ylabel(r"$u(c)$")

    plt.xlim([0, 3])
    plt.ylim([0, 5])

    plt.xlabel("X", fontweight="bold")
    plt.ylabel("Y", fontweight="bold")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.legend(loc="upper left", fontsize="small")

    return fig


_, c1, _ = utl.narrow_col()

with c1:
    _, col_plot, _ = utl.narrow_col()
    col_plot.pyplot(plot_utility(lam_cust), use_container_width=True)

_, c2, _ = utl.wide_col()

with c2:
    st.markdown(
        "<h4 style='text-align: left'>Potentially interesting takeaways</h4>",
        unsafe_allow_html=True,
    )

    st.markdown("Takeaways are interesting to some people but not to others.")


### THEORY ###
_, c3, _ = utl.wide_col()

with c3:
    st.markdown(
        "<h3 style='text-align: left'> 2. Theory part with formulas</h3>",
        unsafe_allow_html=True,
    )

    st.write(r"This is some text about what's below - general idea.")
    st.write(r"It will contain some formulas")

### EXERCISES ###
_, c4, _ = utl.wide_col()

with c4:
    st.markdown(
        "<h3 style='text-align: left'> 3. Exercise part with problems</h3>",
        unsafe_allow_html=True,
    )

    st.write(r"This is some text about what's below - general idea.")
    st.write(
        r"It will contain some exercise questions with or without solutions."
    )


### PROOFS ###
_, c5, _ = utl.wide_col()

with c5:
    st.markdown(
        "<h3 style='text-align: left'> 4. Proofs to remember</h3>",
        unsafe_allow_html=True,
    )

    st.write(r"This is some text about what's below - general idea.")
    st.write(
        r"It will contain some proofs, i.e., formulas that flow coherently."
    )
