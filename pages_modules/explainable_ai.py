"""Explainable AI page — SHAP analysis of the trained Random Forest models.

All SHAP values are computed on the fly from the real, pre-trained RF
models and the real CFD dataset (no dummy / synthetic SHAP values).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.styling import section_header, card_open, card_close
from utils.data_loader import load_rf_models, load_main_dataset, TARGET_META

FEATURES = ["Temperature", "Voltage", "Conductivity"]
FEATURE_COLORS = {"Temperature": "#DC2626", "Voltage": "#2563EB", "Conductivity": "#059669"}

# Reused on every Plotly chart so hover tooltips are always readable
# (white background, dark text) regardless of the app's theme.
HOVERLABEL = dict(bgcolor="white", font_color="#0F172A", font_size=13, bordercolor="#E2E8F0")


@st.cache_data(show_spinner=False)
def compute_shap(_model, X: pd.DataFrame, target_key: str):
    """Compute SHAP values for a tree-based model. Cached on the target
    key + data shape (the leading underscore keeps the model object out
    of Streamlit's hashing, since fitted estimators aren't hashable)."""
    import shap
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    return np.asarray(shap_values), explainer.expected_value


def render(go_to):
    section_header(
        "Explainable AI — SHAP Analysis",
        "Interpreting the Random Forest surrogate models with SHapley Additive exPlanations (real values, computed live from the trained models).",
    )

    rf_models, missing = load_rf_models()
    if missing:
        st.warning("Some Random Forest model files could not be loaded: " + ", ".join(missing))

    dataset, err = load_main_dataset()
    if dataset is None:
        st.error(f"Could not load the reference dataset needed for SHAP analysis ({err}).")
        return

    target_key = st.selectbox(
        "Select target variable",
        options=list(TARGET_META.keys()),
        format_func=lambda k: TARGET_META[k]["label"],
    )
    model = rf_models.get(target_key)
    if model is None:
        st.error(f"The Random Forest model for {TARGET_META[target_key]['label']} is unavailable.")
        return

    feature_order = list(getattr(model, "feature_names_in_", FEATURES))
    X = dataset[feature_order].copy()

    n_sample = st.slider("Sample size used for SHAP computation", min_value=50,
                          max_value=min(300, len(X)), value=min(150, len(X)), step=10)
    X_sample = X.sample(n=n_sample, random_state=42).reset_index(drop=True)

    with st.spinner("Computing SHAP values from the trained Random Forest model..."):
        shap_values, expected_value = compute_shap(model, X_sample, target_key)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order_idx = np.argsort(mean_abs)[::-1]
    ordered_features = [feature_order[i] for i in order_idx]

    # ---------------- Feature importance (mean |SHAP|) ----------------
    section_header("Feature Importance (mean |SHAP value|)")
    fi_fig = go.Figure(go.Bar(
        x=mean_abs[order_idx],
        y=ordered_features,
        orientation="h",
        marker_color=[FEATURE_COLORS.get(f, "#2563EB") for f in ordered_features],
    ))
    fi_fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="mean(|SHAP value|)", yaxis=dict(autorange="reversed"),
        font=dict(color="#0F172A"),
        hoverlabel=HOVERLABEL,
    )
    card_open()
    st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})
    card_close()

    st.write("")

    # ---------------- SHAP Summary (beeswarm-style) plot ----------------
    section_header("SHAP Summary Plot", "Each point is one CFD operating condition; colour encodes the feature's own value.")
    summary_fig = go.Figure()
    rng = np.random.default_rng(0)
    for row_i, feat in enumerate(ordered_features):
        col_i = feature_order.index(feat)
        vals = shap_values[:, col_i]
        fvals = X_sample[feat].values
        norm = (fvals - fvals.min()) / (fvals.max() - fvals.min() + 1e-12)
        jitter = rng.uniform(-0.28, 0.28, size=len(vals))
        summary_fig.add_trace(go.Scatter(
            x=vals, y=[row_i + j for j in jitter],
            mode="markers",
            marker=dict(size=6, color=norm, colorscale="Bluered", showscale=(row_i == 0),
                        colorbar=dict(title="Feature value", x=1.02) if row_i == 0 else None,
                        line=dict(width=0)),
            name=feat, showlegend=False,
            hovertemplate=f"{feat}: %{{text}}<br>SHAP: %{{x:.3e}}<extra></extra>",
            text=[f"{v:.3g}" for v in fvals],
        ))
    summary_fig.update_layout(
        height=320, margin=dict(l=10, r=60, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickmode="array", tickvals=list(range(len(ordered_features))), ticktext=ordered_features),
        xaxis_title="SHAP value (impact on model output)",
        font=dict(color="#0F172A"),
        hoverlabel=HOVERLABEL,
    )
    card_open()
    st.plotly_chart(summary_fig, use_container_width=True, config={"displayModeBar": False})
    card_close()

    st.write("")

    # ---------------- Dependence plot ----------------
    section_header("SHAP Dependence Plot")
    dep_feature = st.selectbox("Feature", options=ordered_features, key="dep_feat")
    color_feature = st.selectbox("Colour by", options=[f for f in feature_order if f != dep_feature], key="dep_color")

    col_i = feature_order.index(dep_feature)
    color_i = feature_order.index(color_feature)
    dep_fig = go.Figure(go.Scatter(
        x=X_sample[dep_feature], y=shap_values[:, col_i],
        mode="markers",
        marker=dict(size=7, color=X_sample[color_feature], colorscale="Bluered",
                    colorbar=dict(title=color_feature), line=dict(width=0)),
    ))
    dep_fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=dep_feature, yaxis_title=f"SHAP value for {dep_feature}",
        font=dict(color="#0F172A"),
        hoverlabel=HOVERLABEL,
    )
    card_open()
    st.plotly_chart(dep_fig, use_container_width=True, config={"displayModeBar": False})
    card_close()

    st.write("")

    # ---------------- Local explanation (waterfall) ----------------
    section_header("Local Prediction Explanation (Waterfall)", "Explains one individual operating condition's prediction.")
    row_idx = st.slider("Operating condition index (from the sampled set)", 0, len(X_sample) - 1, 0)
    row_shap = shap_values[row_idx]
    row_vals = X_sample.iloc[row_idx]
    base = float(np.mean(expected_value)) if hasattr(expected_value, "__len__") else float(expected_value)
    prediction = base + row_shap.sum()

    order_local = np.argsort(np.abs(row_shap))[::-1]
    labels = [f"{feature_order[i]} = {row_vals[feature_order[i]]:.3g}" for i in order_local]
    values = [row_shap[i] for i in order_local]

    wf = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(values) + ["total"],
        y=["Base value"] + labels[::-1],
        x=[base] + values[::-1],
        connector={"line": {"color": "#CBD5E1"}},
        increasing={"marker": {"color": "#DC2626"}},
        decreasing={"marker": {"color": "#2563EB"}},
        totals={"marker": {"color": "#0F172A"}},
    ))
    wf.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0F172A"),
        hoverlabel=HOVERLABEL,
    )
    card_open()
    st.plotly_chart(wf, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Base (expected) value: {base:.4g} → Model prediction for this condition: {prediction:.4g}")
    card_close()

    # ---------------- Plain-English explanation ----------------
    top_feat = ordered_features[0]
    st.write("")
    st.markdown(
        f"""
        <div class="card-flat">
        <b>Engineering explanation:</b> for <b>{TARGET_META[target_key]['label']}</b>, the Random Forest
        model relies most heavily on <b>{top_feat}</b>, consistent with the correlation and feature-importance
        analysis in the project report. Points coloured red in the summary plot above have high feature values;
        their horizontal position shows whether that pushed the prediction up (right) or down (left).
        For Efficiency and Voltage Loss, Voltage dominates almost completely, reflecting their near-direct
        physical dependence on the applied cell potential. Hydrogen Production and Degradation Index depend
        more evenly on all three operating variables, reflecting coupled electrochemical and transport effects.
        </div>
        """,
        unsafe_allow_html=True,
    )
