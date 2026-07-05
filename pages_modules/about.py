"""About page — project objective, methodology, software stack, authors, references."""

import streamlit as st
from utils.styling import section_header, card_open, card_close


def render(go_to):
    section_header("About This Project")

    card_open()
    st.markdown("#### 🎯 Project Objective")
    st.write(
        "Develop and validate machine-learning surrogate models — Random Forest and Gaussian "
        "Process Regression — that reproduce high-fidelity ANSYS Fluent CFD simulations of a "
        "Proton Exchange Membrane (PEM) electrolyser, then use the validated surrogate together "
        "with SHAP interpretability and NSGA-II multi-objective optimisation to identify optimal "
        "operating conditions for hydrogen production, efficiency, voltage loss and degradation."
    )
    card_close()

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        card_open()
        st.markdown("#### 🧪 CFD Simulation")
        st.write(
            "A steady-state, pressure-based ANSYS Fluent model with Butler–Volmer electrochemical "
            "kinetics, osmotic drag, capillary pressure and multiphase flow was built for a resolved-layer "
            "PEM electrolyser (current collectors, porous transport layers, catalyst layers and Nafion "
            "membrane). A Latin Hypercube Sampling DOE generated 312 training simulations and 25 "
            "independent external-validation simulations across Temperature (313–383 K), Cell Voltage "
            "(1.5–2.0 V) and Electrolyte Conductivity (3–20 S/m)."
        )
        card_close()
    with col2:
        card_open()
        st.markdown("#### 🤖 Machine Learning Pipeline")
        st.write(
            "Two surrogate families were trained per target: a 500-tree Random Forest (no scaling "
            "required) and a Gaussian Process Regression pipeline (StandardScaler + Constant×RBF+White "
            "kernel, 10 optimizer restarts). GPR was shown to require feature scaling — an unscaled GPR "
            "collapsed to R² ≈ 0 because Conductivity's wide numeric range dominated the RBF distance "
            "metric. After scaling, five-fold CV R² reached ≥ 0.99 for every target."
        )
        card_close()

    st.write("")
    col3, col4 = st.columns(2)
    with col3:
        card_open()
        st.markdown("#### 🔍 Explainable AI")
        st.write(
            "SHAP (SHapley Additive exPlanations) was applied to the Random Forest models to attribute "
            "each prediction to Temperature, Voltage and Conductivity. Cell Voltage dominates Efficiency "
            "and Voltage Loss, while Hydrogen Production and Degradation Index depend jointly on all "
            "three operating variables."
        )
        card_close()
    with col4:
        card_open()
        st.markdown("#### 🎯 Optimization")
        st.write(
            "The validated GPR surrogate was coupled with NSGA-II (via the pymoo framework) to "
            "simultaneously maximise Hydrogen Production and Efficiency while minimising Voltage Loss "
            "and Degradation Index, producing a 200-point Pareto front. High Production, Balanced and "
            "Conservative operating points were then selected using an ideal-point distance method."
        )
        card_close()

    st.write("")
    card_open()
    st.markdown("#### 🧭 Overall Workflow")
    st.code(
        "PEM Electrolyzer → CFD Model (ANSYS Fluent) → DOE (Latin Hypercube Sampling)\n"
        "        → 312 Simulations → Dataset → Random Forest + GPR\n"
        "        → 5-Fold Cross Validation → External Validation → SHAP\n"
        "        → NSGA-II → Optimal Operating Conditions",
        language=None,
    )
    card_close()

    st.write("")
    card_open()
    st.markdown("#### 🛠️ Software Used")
    st.markdown(
        "- **ANSYS Fluent** — coupled electrochemical–thermal–fluid CFD simulation\n"
        "- **Python / scikit-learn** — Random Forest & Gaussian Process Regression surrogate modelling\n"
        "- **pyDOE2** — Latin Hypercube Sampling design of experiments\n"
        "- **SHAP** — model interpretability and feature attribution\n"
        "- **pymoo** — NSGA-II multi-objective optimisation\n"
        "- **Streamlit / Plotly** — this interactive digital-twin dashboard"
    )
    card_close()

    st.write("")
    card_open()
    st.markdown("#### 👥 Authors")
    st.markdown(
        "**Mass Transfer-II Project Based Laboratory (CH363IA)** — Department of Chemical Engineering, "
        "RV College of Engineering, Bengaluru\n\n"
        "- Dwishan R Acharya (1RV23CH017)\n"
        "- Sriprada Ramesh (1RV23CH037)\n"
        "- Supreet Naik (1RV23CH038)\n"
        "- Umaid Raj Singh (1RV23CH040)\n\n"
        "Faculty in Charge: Dr. P L Muralidhara · HoD: Dr. Jagadish H Patil"
    )
    card_close()

    st.write("")
    card_open()
    st.markdown("#### 📚 Key References")
    st.markdown(
        "1. Carmo, M. et al. *A comprehensive review on PEM water electrolysis.* Int. J. Hydrogen Energy, 2013.\n"
        "2. Breiman, L. *Random forests.* Machine Learning, 2001.\n"
        "3. Rasmussen, C. E. & Williams, C. K. I. *Gaussian Processes for Machine Learning.* MIT Press, 2006.\n"
        "4. Lundberg, S. M. & Lee, S.-I. *A unified approach to interpreting model predictions.* NeurIPS, 2017.\n"
        "5. Deb, K. et al. *A fast and elitist multiobjective genetic algorithm: NSGA-II.* IEEE Trans. Evol. Comput., 2002.\n"
        "6. Blank, J. & Deb, K. *pymoo: Multi-objective optimization in Python.* IEEE Access, 2020.\n\n"
        "*Full reference list available in the accompanying project report.*"
    )
    card_close()
