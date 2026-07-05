"""Digital Twin — interactive prediction page using the trained GPR models."""

import streamlit as st
import plotly.graph_objects as go
from utils.styling import section_header, card_open, card_close
from utils.data_loader import load_gpr_models, predict_with_model, load_main_dataset, TARGET_META
from utils.helpers import health_status, engineering_interpretation, make_gauge, build_prediction_report


def render(go_to):
    section_header(
        "Digital Twin — PEM Electrolyser Performance Prediction",
        "Adjust the operating conditions to obtain real-time predictions from the validated Gaussian Process Regression surrogate models.",
    )

    gpr_models, missing = load_gpr_models()
    if missing:
        st.warning(
            "Some GPR model files could not be loaded, so a few outputs may be unavailable: "
            + ", ".join(missing)
        )

    dataset, err = load_main_dataset()

    # Determine slider ranges from the underlying CFD dataset when available,
    # falling back to the documented DOE ranges otherwise.
    if dataset is not None and {"Temperature", "Voltage", "Conductivity"}.issubset(dataset.columns):
        t_min, t_max = float(dataset["Temperature"].min()), float(dataset["Temperature"].max())
        v_min, v_max = float(dataset["Voltage"].min()), float(dataset["Voltage"].max())
        c_min, c_max = float(dataset["Conductivity"].min()), float(dataset["Conductivity"].max())
    else:
        t_min, t_max = 313.15, 383.15
        v_min, v_max = 1.5, 2.0
        c_min, c_max = 3.0, 20.0

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        card_open()
        st.markdown("#### Operating Conditions")
        st.caption("Adjust the operating parameters of the PEM electrolyser")

        temperature = st.slider("Temperature (K)", min_value=round(t_min, 2), max_value=round(t_max, 2),
                                 value=round((t_min + t_max) / 2, 2), step=0.1)
        voltage = st.slider("Voltage (V)", min_value=round(v_min, 3), max_value=round(v_max, 3),
                             value=round((v_min + v_max) / 2, 3), step=0.001)
        conductivity = st.slider("Conductivity (S/m)", min_value=round(c_min, 2), max_value=round(c_max, 2),
                                  value=round((c_min + c_max) / 2, 2), step=0.1)

        st.write("")
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)
        card_close()

    # Run predictions (auto-update, "Predict" just gives a tactile confirmation)
    predictions = {}
    stds = {}
    for tgt, model in gpr_models.items():
        if model is None:
            predictions[tgt] = None
            stds[tgt] = None
            continue
        try:
            pred, std = predict_with_model(model, temperature, voltage, conductivity, return_std=True)
            predictions[tgt] = pred
            stds[tgt] = std
        except Exception as e:
            predictions[tgt] = None
            stds[tgt] = None

    with right:
        card_open()
        st.markdown("#### Model Predictions")
        st.caption("Predicted performance metrics of the PEM electrolyser (with 95% confidence interval where available)")

        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)
        kpi_cols = {"h2": k1, "eff": k2, "vloss": k3, "deg": k4}

        for tgt, col in kpi_cols.items():
            meta = TARGET_META[tgt]
            with col:
                val = predictions[tgt]
                std = stds[tgt]
                if val is None:
                    st.markdown(
                        f"""<div class="kpi-card {meta['color']}">
                        <div class="kpi-label">{meta['label']}</div>
                        <div class="kpi-value" style="font-size:18px;color:#94A3B8;">Model unavailable</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
                    continue
                if tgt == "h2":
                    val_str = f"{val:.6f}"
                else:
                    val_str = f"{val:.4f}"
                ci_str = f"± {1.96*std:.2e} (95% CI)" if std is not None else ""
                st.markdown(
                    f"""
                    <div class="kpi-card {meta['color']}">
                        <div style="display:flex;justify-content:space-between;">
                            <div class="kpi-label">{meta['label']}</div>
                            <div style="font-size:18px;">{meta['icon']}</div>
                        </div>
                        <div class="kpi-value">{val_str} <span class="kpi-unit">{meta['unit']}</span></div>
                        <div class="kpi-delta" style="color:#64748B;">{ci_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        card_close()

    st.write("")

    # ---------------- Gauges ----------------
    if all(v is not None for v in predictions.values()):
        section_header("Engineering Gauges")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge(predictions["h2"] * 1e5, "H₂ Production (×10⁻⁵ mol/s)", 0, 8, color="#2563EB"),
                             use_container_width=True, config={"displayModeBar": False})
        with g2:
            st.plotly_chart(make_gauge(predictions["eff"], "Efficiency", 0, 1, color="#059669"),
                             use_container_width=True, config={"displayModeBar": False})
        with g3:
            st.plotly_chart(make_gauge(predictions["vloss"], "Voltage Loss (V)", 0, 1, color="#D97706", reverse=True),
                             use_container_width=True, config={"displayModeBar": False})
        with g4:
            st.plotly_chart(make_gauge(predictions["deg"], "Degradation Index", 0, 1, color="#7C3AED", reverse=True),
                             use_container_width=True, config={"displayModeBar": False})

        # ---------------- Health indicator + interpretation ----------------
        status, css_class, color = health_status(predictions["eff"], predictions["deg"], predictions["vloss"])
        interp = engineering_interpretation(
            temperature, voltage, conductivity,
            predictions["h2"], predictions["eff"], predictions["vloss"], predictions["deg"]
        )

        h_col, i_col = st.columns([1, 2.2])
        with h_col:
            card_open()
            st.markdown("**Stack Health Indicator**")
            st.markdown(
                f"""<div style="margin-top:10px;font-size:16px;font-weight:700;color:{color};">
                <span class="health-dot {css_class}"></span>{status}
                </div>""",
                unsafe_allow_html=True,
            )
            st.caption("Rule-of-thumb bucketing based on predicted Efficiency, Voltage Loss and Degradation Index — not a model output.")
            card_close()
        with i_col:
            card_open()
            st.markdown("**Engineering Interpretation**")
            st.write(interp)
            card_close()

        st.write("")

        # ---------------- Download report ----------------
        inputs = {
            "Temperature (K)": f"{temperature:.2f}",
            "Voltage (V)": f"{voltage:.3f}",
            "Conductivity (S/m)": f"{conductivity:.2f}",
        }
        outputs = {
            "Hydrogen Production (mol/s)": f"{predictions['h2']:.6e}",
            "Efficiency": f"{predictions['eff']:.4f}",
            "Voltage Loss (V)": f"{predictions['vloss']:.4f}",
            "Degradation Index": f"{predictions['deg']:.4f}",
            "Stack Health": status,
        }
        report_text = build_prediction_report(inputs, outputs)
        st.download_button(
            "⬇️ Download Prediction Report",
            data=report_text,
            file_name="pemtwin_prediction_report.txt",
            mime="text/plain",
            use_container_width=False,
        )
    else:
        st.info("Load the GPR model files under `models/` to see gauges, health status and the downloadable report.")

    st.write("")
    st.markdown(
        """
        <div class="card-flat">
        ℹ️ These predictions are generated using Gaussian Process Regression (GPR) models trained on
        312 high-fidelity ANSYS Fluent CFD simulations of the PEM electrolyser (five-fold CV R² ≥ 0.99,
        external validation R² > 0.995 for Hydrogen Production). Use the sliders to explore how operating
        conditions affect performance.
        </div>
        """,
        unsafe_allow_html=True,
    )
