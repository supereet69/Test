"""Validation page — cross-validation & external validation results for RF and GPR."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import section_header, card_open, card_close, style_table
from utils.data_loader import load_all_reference_data, TARGET_META

HOVERLABEL = dict(bgcolor="white", font_color="#0F172A", font_size=13, bordercolor="#E2E8F0")

# Mapping between the friendly target keys and the actual/predicted CSV columns
TARGET_COLS = {
    "h2": {"actual": "Faraday_H2_mol", "pred": "Pred_H2", "label": "Hydrogen Production"},
    "eff": {"actual": "Efficiency", "pred": "Pred_Efficiency", "label": "Efficiency"},
    "vloss": {"actual": "Voltage_loss", "pred": "Pred_VoltageLoss", "label": "Voltage Loss"},
    "deg": {"actual": "Degradation_index", "pred": "Pred_Degradation", "label": "Degradation Index"},
}


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(resid))
    rmse = np.sqrt(np.mean(resid ** 2))
    mape = np.mean(np.abs(resid / y_true)) * 100 if np.all(y_true != 0) else np.nan
    return r2, mae, rmse, mape


def render(go_to):
    section_header(
        "Model Validation",
        "Five-fold cross-validation and independent external validation of the Random Forest and Gaussian Process Regression surrogates.",
    )

    data, errors = load_all_reference_data()
    model_comp = data.get("model_comparison")
    rf_5fold = data.get("rf_5fold")
    rf_ext = data.get("rf_external")
    gpr_ext_metrics = data.get("gpr_external_metrics")
    ansys = data.get("testresults_ansys")
    model_preds = data.get("testresults_model")

    # ---------------- Model comparison summary ----------------
    section_header("Model Comparison Summary")
    if model_comp is not None:
        card_open()
        st.dataframe(style_table(model_comp), use_container_width=True, hide_index=True)
        card_close()
    else:
        st.warning(f"Model comparison table unavailable ({errors.get('model_comparison')}).")

    st.write("")

    # ---------------- R2 comparison chart ----------------
    if model_comp is not None:
        section_header("R² Comparison: RF vs GPR (5-Fold CV & External Validation)")
        targets = model_comp["Target"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="RF (5-Fold CV)", x=targets, y=model_comp["RF (5-Fold R²)"], marker_color="#93C5FD"))
        fig.add_trace(go.Bar(name="RF (External)", x=targets, y=model_comp["RF (External R²)"], marker_color="#2563EB"))
        fig.add_trace(go.Bar(name="GPR (5-Fold CV)", x=targets, y=model_comp["GPR (5-Fold R²)"], marker_color="#FDE68A"))
        fig.add_trace(go.Bar(name="GPR (External)", x=targets, y=model_comp["GPR (External R²)"], marker_color="#D97706"))
        fig.update_layout(
            barmode="group", height=400, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="R²", legend=dict(orientation="h", y=-0.25), font=dict(color="#0F172A"),
            hoverlabel=HOVERLABEL,
        )
        card_open()
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        card_close()

    st.write("")

    # ---------------- Cross-validation details ----------------
    section_header("Five-Fold Cross-Validation — Random Forest")
    if rf_5fold is not None:
        card_open()
        st.dataframe(
            style_table(rf_5fold.style.format({c: "{:.4f}" for c in rf_5fold.columns if c != "Target"})),
            use_container_width=True, hide_index=True,
        )
        card_close()
    else:
        st.warning(f"RF cross-validation results unavailable ({errors.get('rf_5fold')}).")

    st.write("")

    # ---------------- External validation metrics table ----------------
    section_header("External Validation Metrics")
    ext_col1, ext_col2 = st.columns(2)
    with ext_col1:
        st.markdown("**Random Forest**")
        if rf_ext is not None:
            card_open()
            st.dataframe(style_table(rf_ext), use_container_width=True, hide_index=True)
            card_close()
        else:
            st.warning(f"Unavailable ({errors.get('rf_external')}).")
    with ext_col2:
        st.markdown("**Gaussian Process Regression**")
        if gpr_ext_metrics is not None:
            card_open()
            st.dataframe(style_table(gpr_ext_metrics), use_container_width=True, hide_index=True)
            card_close()
        else:
            st.warning(f"Unavailable ({errors.get('gpr_external_metrics')}).")

    st.write("")

    # ---------------- Prediction vs Actual + Residuals ----------------
    section_header("Prediction vs Actual (External Validation Set, GPR)")
    if ansys is None or model_preds is None:
        st.warning("External validation prediction/actual data unavailable.")
        return

    target_key = st.selectbox(
        "Select target variable", options=list(TARGET_COLS.keys()),
        format_func=lambda k: TARGET_COLS[k]["label"], key="val_target",
    )
    cols = TARGET_COLS[target_key]
    y_true = ansys[cols["actual"]].values
    y_pred = model_preds[cols["pred"]].values
    r2, mae, rmse, mape = compute_metrics(y_true, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="kpi-card kpi-blue"><div class="kpi-label">R²</div><div class="kpi-value">{r2:.4f}</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="kpi-card kpi-green"><div class="kpi-label">MAE</div><div class="kpi-value" style="font-size:20px;">{mae:.3e}</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="kpi-card kpi-orange"><div class="kpi-label">RMSE</div><div class="kpi-value" style="font-size:20px;">{rmse:.3e}</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="kpi-card kpi-purple"><div class="kpi-label">MAPE</div><div class="kpi-value">{mape:.2f}%</div></div>""", unsafe_allow_html=True)

    st.write("")

    pv_col, res_col = st.columns(2)
    with pv_col:
        fig_pv = go.Figure()
        fig_pv.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                                     marker=dict(color="#2563EB", size=9, line=dict(width=1, color="white")),
                                     name="Predictions"))
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        fig_pv.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#DC2626", dash="dash"), name="Ideal"))
        fig_pv.update_layout(
            title="Predicted vs Actual", height=360, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Actual (CFD)", yaxis_title="Predicted (GPR)", font=dict(color="#0F172A"),
            hoverlabel=HOVERLABEL,
        )
        card_open()
        st.plotly_chart(fig_pv, use_container_width=True, config={"displayModeBar": False})
        card_close()

    with res_col:
        residuals = y_true - y_pred
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                      marker=dict(color="#059669", size=9, line=dict(width=1, color="white"))))
        fig_res.add_hline(y=0, line_dash="dash", line_color="#94A3B8")
        fig_res.update_layout(
            title="Residual Plot", height=360, margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Predicted", yaxis_title="Residual (Actual − Predicted)", font=dict(color="#0F172A"),
            hoverlabel=HOVERLABEL,
        )
        card_open()
        st.plotly_chart(fig_res, use_container_width=True, config={"displayModeBar": False})
        card_close()

    st.write("")

    section_header("Error Distribution")
    fig_hist = go.Figure(go.Histogram(x=residuals, marker_color="#7C3AED", nbinsx=12))
    fig_hist.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Residual", yaxis_title="Count", font=dict(color="#0F172A"),
        hoverlabel=HOVERLABEL,
    )
    card_open()
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
    card_close()

    st.write("")
    st.download_button(
        "⬇️ Download External Validation Comparison (CSV)",
        data=pd.DataFrame({"Actual": y_true, "Predicted": y_pred, "Residual": residuals}).to_csv(index=False),
        file_name=f"validation_{target_key}.csv", mime="text/csv",
    )
