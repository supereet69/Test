"""Engineering-logic helpers: health scoring, plain-English interpretation,
and shared Plotly chart builders."""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

ACCENT = "#2563EB"
GREEN = "#059669"
ORANGE = "#D97706"
RED = "#DC2626"
PURPLE = "#7C3AED"


def health_status(efficiency, degradation_index, voltage_loss):
    """Very simple, transparent rule-based health scoring used purely for
    the operator-facing indicator on the Digital Twin page. Not a model
    prediction -- just a bucketing of the model outputs."""
    score = 0
    score += 1 if efficiency >= 0.70 else (0.5 if efficiency >= 0.60 else 0)
    score += 1 if degradation_index <= 0.25 else (0.5 if degradation_index <= 0.45 else 0)
    score += 1 if voltage_loss <= 0.45 else (0.5 if voltage_loss <= 0.65 else 0)
    pct = score / 3.0

    if pct >= 0.8:
        return "Healthy", "health-good", GREEN
    elif pct >= 0.4:
        return "Monitor", "health-warn", ORANGE
    else:
        return "Attention Needed", "health-bad", RED


def engineering_interpretation(temperature, voltage, conductivity, h2, eff, vloss, deg):
    """Plain-English, physically-grounded interpretation of a prediction,
    consistent with the SHAP / correlation findings reported in the
    project report (Voltage dominates Efficiency & Voltage Loss; the
    Temperature-Voltage-Conductivity triplet jointly governs H2 Production
    and Degradation)."""
    notes = []

    if voltage >= 1.85:
        notes.append(
            "Voltage is near the upper end of the validated range (1.5–2.0 V), "
            "which drives higher hydrogen output but also raises voltage loss "
            "and long-term degradation risk."
        )
    elif voltage <= 1.6:
        notes.append(
            "Voltage is on the conservative side, favouring efficiency and "
            "membrane longevity over raw hydrogen throughput."
        )
    else:
        notes.append("Voltage sits in a balanced mid-range operating band.")

    if temperature >= 365:
        notes.append(
            "Elevated operating temperature improves reaction kinetics and "
            "ionic transport, supporting higher hydrogen production."
        )
    elif temperature <= 325:
        notes.append(
            "Lower operating temperature reduces thermal stress on the "
            "membrane assembly, at some cost to production rate."
        )

    if conductivity <= 6:
        notes.append(
            "Electrolyte conductivity is low, which can limit ionic transport "
            "and constrain achievable current density."
        )
    elif conductivity >= 15:
        notes.append(
            "High electrolyte conductivity supports efficient ion transport "
            "across the membrane."
        )

    if deg >= 0.5:
        notes.append(
            "The predicted Degradation Index is elevated -- sustained operation "
            "here would be expected to accelerate membrane wear."
        )
    elif deg <= 0.15:
        notes.append(
            "The predicted Degradation Index is low, consistent with a "
            "durability-oriented operating point."
        )

    return " ".join(notes)


def make_gauge(value, title, min_val=0, max_val=1, suffix="", color=ACCENT, reverse=False):
    """A compact Plotly gauge used for KPI visual context."""
    band_low = "#DCFCE7" if not reverse else "#FEE2E2"
    band_mid = "#FEF9C3"
    band_high = "#FEE2E2" if not reverse else "#DCFCE7"
    span = max_val - min_val
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": suffix, "font": {"size": 26, "color": "#0F172A"}},
            title={"text": title, "font": {"size": 13, "color": "#475569"}},
            gauge={
                "axis": {"range": [min_val, max_val], "tickcolor": "#94A3B8"},
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [min_val, min_val + span * 0.33], "color": band_low},
                    {"range": [min_val + span * 0.33, min_val + span * 0.66], "color": band_mid},
                    {"range": [min_val + span * 0.66, max_val], "color": band_high},
                ],
            },
        )
    )
    fig.update_layout(
        height=190,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def build_prediction_report(inputs: dict, outputs: dict) -> str:
    """Build a plain-text prediction report for download."""
    lines = []
    lines.append("=" * 60)
    lines.append("PEMTwin AI - Digital Twin Prediction Report")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("OPERATING CONDITIONS")
    lines.append("-" * 60)
    for k, v in inputs.items():
        lines.append(f"  {k:<28}: {v}")
    lines.append("")
    lines.append("PREDICTED PERFORMANCE (Gaussian Process Regression)")
    lines.append("-" * 60)
    for k, v in outputs.items():
        lines.append(f"  {k:<28}: {v}")
    lines.append("")
    lines.append("Model basis: GPR surrogate trained on 312 ANSYS Fluent CFD")
    lines.append("simulations; five-fold CV R2 >= 0.99, external validation")
    lines.append("R2 > 0.995 for Hydrogen Production (see Validation page).")
    lines.append("=" * 60)
    return "\n".join(lines)
