"""
PEMTwin AI — AI-Powered Digital Twin for PEM Electrolyser Performance
Prediction and Optimization.

Main application entry point. Renders a professional, top-navigated
Streamlit dashboard that reuses pre-trained GPR / RF surrogate models,
optimisation results and validation artefacts produced offline (see
Machine Learning / Optimization / Validation folders of the source
repository) -- nothing here is retrained.
"""

import streamlit as st
from streamlit_option_menu import option_menu

from utils.styling import inject_css, footer

st.set_page_config(
    page_title="PEMTwin AI — PEM Electrolyser Digital Twin",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_css()

# ---------------------------------------------------------------------
# Page registry
# ---------------------------------------------------------------------
from pages_modules import home, digital_twin, explainable_ai, optimization, validation, about  # noqa: E402

PAGES = {
    "Home": {"module": home, "icon": "house"},
    "Digital Twin": {"module": digital_twin, "icon": "cpu"},
    "Explainable AI": {"module": explainable_ai, "icon": "search"},
    "Optimization": {"module": optimization, "icon": "bullseye"},
    "Validation": {"module": validation, "icon": "shield-check"},
    "About": {"module": about, "icon": "info-circle"},
}
PAGE_NAMES = list(PAGES.keys())

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


def go_to(page_name: str):
    st.session_state.current_page = page_name
    st.rerun()


# ---------------------------------------------------------------------
# Brand header row (logo + title + quick "About Project" button)
# ---------------------------------------------------------------------
brand_col, spacer_col, about_col = st.columns([3, 5, 1.3])
with brand_col:
    st.markdown(
        """
        <div class="pemtwin-brand">
            <div class="pemtwin-logo">H₂</div>
            <div>
                <div class="pemtwin-title">PEMTwin AI</div>
                <div class="pemtwin-subtitle">Digital Twin for PEM Electrolyser</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with about_col:
    st.write("")
    if st.button("ℹ️ About Project", use_container_width=True):
        go_to("About")

# ---------------------------------------------------------------------
# Top navigation bar
# ---------------------------------------------------------------------
selected = option_menu(
    menu_title=None,
    options=PAGE_NAMES,
    icons=[PAGES[p]["icon"] for p in PAGE_NAMES],
    orientation="horizontal",
    default_index=PAGE_NAMES.index(st.session_state.current_page),
    styles={
        "container": {"padding": "4px 0", "background-color": "#FFFFFF", "border-bottom": "1px solid #E2E8F0"},
        "icon": {"color": "#2563EB", "font-size": "15px"},
        "nav-link": {
            "font-size": "14.5px", "font-weight": "600", "color": "#475569",
            "text-align": "center", "margin": "0 4px", "padding": "10px 16px",
            "border-radius": "10px",
        },
        "nav-link-selected": {"background-color": "#EFF6FF", "color": "#1D4ED8"},
    },
    key="top_nav_menu",
)

if selected != st.session_state.current_page:
    st.session_state.current_page = selected

st.write("")

# ---------------------------------------------------------------------
# Render the selected page
# ---------------------------------------------------------------------
active_page = PAGES[st.session_state.current_page]["module"]
try:
    active_page.render(go_to)
except Exception as e:
    st.error(f"This page couldn't be rendered because of an unexpected error: {e}")
    st.exception(e)

footer()
