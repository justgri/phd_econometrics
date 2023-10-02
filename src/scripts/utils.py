import streamlit as st


def local_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def external_css(file_url):
    st.markdown(
        f'<link href="{file_url}" rel="stylesheet">', unsafe_allow_html=True
    )


def wide_col():
    return st.columns((0.2, 1, 0.2))


def narrow_col():
    return st.columns((0.35, 1, 0.35))


def two_cols():
    return st.columns((0.1, 1, 0.2, 1, 0.1))
