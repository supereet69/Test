"""Reusable styling / UI component helpers for PEMTwin AI."""
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
CSS_PATH = ROOT_DIR / "assets" / "style.css"


def inject_css():
    """Load the project's stylesheet, if present, without crashing when
    it's missing."""
    if CSS_PATH.exists():
        css = CSS_PATH.read_text()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def style_table(obj):
    """Force a white background + black text on any dataframe/Styler
    passed to st.dataframe(). Works whether you pass a plain DataFrame
    or one you've already chained .style.format(...) onto.

    Usage:
        st.dataframe(style_table(my_df), use_container_width=True)
        st.dataframe(style_table(my_df.style.format({...})), ...)
    """
    styler = obj if isinstance(obj, pd.io.formats.style.Styler) else obj.style
    return styler.set_properties(**{
        "background-color": "white",
        "color": "black",
        "border-color": "#E2E8F0",
    }).set_table_styles([
        {"selector": "th", "props": [
            ("background-color", "#F1F5F9"),
            ("color", "#0F172A"),
            ("font-weight", "700"),
        ]}
    ])


def render_brand_header():
    """Small brand lockup shown above the navigation bar."""
    st.markdown(
        """
        <div class="pemtwin-brand" style="margin-bottom:6px;">
            <div class="pemtwin-logo">H₂</div>
            <div>
                <div class="pemtwin-title">PEMTwin AI</div>
                <div class="pemtwin-subtitle">Digital Twin for PEM Electrolyser</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label, value, unit="", color_class="kpi-blue", icon="", delta_text=None, delta_color="#059669"):
    delta_html = f'<div class="kpi-delta" style="color:{delta_color}">{delta_text}</div>' if delta_text else ""
    st.markdown(
        f"""
        <div class="kpi-card {color_class}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div class="kpi-label">{label}</div>
                <div style="font-size:20px;">{icon}</div>
            </div>
            <div class="kpi-value">{value} <span class="kpi-unit">{unit}</span></div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    sub_html = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div style="margin: 6px 0 14px 0;">
            <div class="section-title">{title}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card_open():
    st.markdown('<div class="card">', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def pill(text, style="pill-blue"):
    return f'<span class="pill {style}">{text}</span>'


def footer():
    st.markdown(
        """
        <div class="pemtwin-footer">
            PEMTwin AI &nbsp;•&nbsp; Digital Twin for PEM Electrolyser &nbsp;•&nbsp; AI Powered Insights<br>
            © 2026 All Rights Reserved
        </div>
        """,
        unsafe_allow_html=True,
    )
