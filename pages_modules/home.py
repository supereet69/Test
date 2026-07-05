"""Home Dashboard page."""

import streamlit as st
from utils.styling import section_header, card_open, card_close, pill
from utils.data_loader import get_project_stats


def render(go_to):
    stats = get_project_stats()

    # ---------------- Hero ----------------
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown('<div class="hero-badge">⚙️ AI-Powered Digital Twin</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">PEM Electrolyser<br>Performance Digital Twin</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="hero-sub">
            A machine-learning digital twin that replaces expensive ANSYS Fluent CFD
            simulations with validated Random Forest and Gaussian Process Regression
            surrogates -- delivering instant predictions of hydrogen production,
            efficiency, voltage loss and degradation, with explainable AI and
            multi-objective optimisation built in.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button(" Open Digital Twin", use_container_width=True, type="primary"):
                go_to("Digital Twin")
        with b2:
            if st.button(" View Optimization", use_container_width=True):
                go_to("Optimization")
        with b3:
            if st.button(" View Validation", use_container_width=True):
                go_to("Validation")

    with right:
        card_open()
        st.markdown("**Project Snapshot**")
        st.markdown(
            f"""
            <div style="display:flex;flex-direction:column;gap:10px;margin-top:8px;">
                <div>{pill(f"{stats['n_simulations']} CFD Simulations", "pill-blue")}</div>
                <div>{pill(f"{stats['n_models']} Trained Models", "pill-green")}</div>
                <div>{pill(f"{stats['n_targets']} Performance Targets", "pill-purple")}</div>
                <div>{pill(f"{stats['n_pareto']} Pareto Solutions", "pill-orange")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        card_close()

    st.write("")
    st.write("")

    # ---------------- Workflow ----------------
    section_header("AI Pipeline: CFD → Machine Learning → Optimization",
                    "The complete workflow underpinning this digital twin, from high-fidelity simulation to actionable operating strategy.")

    steps = [
        ("🧬", "CFD Simulation", "312 ANSYS Fluent runs across a Latin Hypercube DOE spanning Temperature, Voltage & Conductivity"),
        ("🌲", "Random Forest", "500-tree ensemble benchmark surrogate — robust, nonlinear, minimal preprocessing"),
        ("📈", "Gaussian Process", "Scaled RBF + White kernel GPR — best accuracy, native uncertainty quantification"),
        ("🔍", "Explainable AI", "SHAP attribution reveals which operating variables drive each prediction"),
        ("🎯", "NSGA-II Optimization", "Multi-objective search over the validated GPR surrogate yields a Pareto front"),
    ]
    cols = st.columns(len(steps))
    for c, (icon, title, desc) in zip(cols, steps):
        with c:
            st.markdown(
                f"""
                <div class="nav-card">
                    <div class="nav-card-icon">{icon}</div>
                    <div style="font-weight:700;font-size:14.5px;margin-bottom:4px;">{title}</div>
                    <div style="font-size:12.5px;color:#475569;line-height:1.4;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.write("")

    # ---------------- Model summary ----------------
    section_header("Model Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            """<div class="card"><div class="kpi-label">Input Variables</div>
            <div class="kpi-value" style="font-size:24px;">3</div>
            <div style="font-size:12.5px;color:#475569;">Temperature, Voltage, Conductivity</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            """<div class="card"><div class="kpi-label">Output Targets</div>
            <div class="kpi-value" style="font-size:24px;">4</div>
            <div style="font-size:12.5px;color:#475569;">H₂ Production, Efficiency, Voltage Loss, Degradation</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            """<div class="card"><div class="kpi-label">Best Surrogate</div>
            <div class="kpi-value" style="font-size:24px;">GPR</div>
            <div style="font-size:12.5px;color:#475569;">External R² up to 0.9955 (H₂ Production)</div></div>""",
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            """<div class="card"><div class="kpi-label">Optimizer</div>
            <div class="kpi-value" style="font-size:24px;">NSGA-II</div>
            <div style="font-size:12.5px;color:#475569;">200 non-dominated Pareto solutions</div></div>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")

    # ---------------- Navigation cards ----------------
    section_header("Explore the Application")
    nav_items = [
        ("🧭", "Digital Twin", "Predict PEM electrolyser performance interactively using the validated GPR models.", "Digital Twin"),
        ("🔍", "Explainable AI", "SHAP-based interpretation of the Random Forest surrogate models.", "Explainable AI"),
        ("🎯", "Optimization", "Explore the NSGA-II Pareto front and recommended operating conditions.", "Optimization"),
        ("✅", "Validation", "Cross-validation and external validation metrics for RF and GPR.", "Validation"),
        ("ℹ️", "About", "Methodology, workflow, software stack, authors and references.", "About"),
    ]
    cols = st.columns(len(nav_items))
    for c, (icon, title, desc, target) in zip(cols, nav_items):
        with c:
            st.markdown(
                f"""
                <div class="nav-card">
                    <div class="nav-card-icon">{icon}</div>
                    <div style="font-weight:700;font-size:15px;margin-bottom:4px;">{title}</div>
                    <div style="font-size:12.5px;color:#475569;line-height:1.4;min-height:58px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open →", key=f"nav_{target}", use_container_width=True):
                go_to(target)
