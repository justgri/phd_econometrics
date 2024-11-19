import streamlit as st

import src.scripts.plot_themes
import src.scripts.utils as utl

st.set_page_config(
    page_title="PhD Econometrics",
    page_icon="📈",
    layout="wide",
)


intro_page = st.Page("pages/home.py", title="Introduction", icon="🏠")
tools_apge = st.Page("pages/0_tools.py", title="Must-know", icon="🛠")
ols_page = st.Page("pages/1_ols.py", title="Linear Regression", icon="📈")
ols3d_page = st.Page("pages/2_ols_3d.py", title="OLS in 3D", icon="🔍")
fit_page = st.Page("pages/3_fit.py", title="Fit Measures", icon="🎯")
ovb_page = st.Page("pages/4_ovb.py", title="Omitted Variables", icon="🤔")
me_page = st.Page("pages/5_measurement_error.py", title="Measurement Error", icon="📐")


pg = st.navigation(
    [intro_page, tools_apge, ols_page, ols3d_page, fit_page, ovb_page, me_page]
)

pg.run()
