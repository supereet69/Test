"""
Central data & model loading module for PEMTwin AI.

All paths are resolved relative to this file's location, so the app
works regardless of the current working directory or where the project
folder is placed on disk. No file paths are hardcoded to an absolute
location.
"""

from pathlib import Path
import warnings
import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Root-relative paths
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

GPR_TARGETS = ["h2", "eff", "vloss", "deg"]
RF_TARGETS = ["h2", "eff", "vloss", "deg"]

TARGET_META = {
    "h2": {
        "label": "Hydrogen Production",
        "unit": "mol/s",
        "column": "Faraday_H2_mol",
        "icon": "🧪",
        "color": "kpi-blue",
        "higher_is_better": True,
    },
    "eff": {
        "label": "Efficiency",
        "unit": "",
        "column": "Efficiency",
        "icon": "⚡",
        "color": "kpi-green",
        "higher_is_better": True,
    },
    "vloss": {
        "label": "Voltage Loss",
        "unit": "V",
        "column": "Voltage_loss",
        "icon": "🔻",
        "color": "kpi-orange",
        "higher_is_better": False,
    },
    "deg": {
        "label": "Degradation Index",
        "unit": "",
        "column": "Degradation_index",
        "icon": "📉",
        "color": "kpi-purple",
        "higher_is_better": False,
    },
}


def _safe_path(path: Path):
    """Return the path if it exists, else None (never raises)."""
    return path if path.exists() else None


# ---------------------------------------------------------------------
# Model loading (cached as resources -- objects persist across reruns)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_gpr_models():
    """Load the four pre-trained Gaussian Process Regression pipelines."""
    models = {}
    missing = []
    for tgt in GPR_TARGETS:
        p = _safe_path(MODELS_DIR / f"gpr_{tgt}.pkl")
        if p is None:
            missing.append(f"gpr_{tgt}.pkl")
            models[tgt] = None
            continue
        try:
            models[tgt] = joblib.load(p)
        except Exception as e:
            models[tgt] = None
            missing.append(f"gpr_{tgt}.pkl ({e})")
    return models, missing


@st.cache_resource(show_spinner=False)
def load_rf_models():
    """Load the four pre-trained Random Forest models."""
    models = {}
    missing = []
    for tgt in RF_TARGETS:
        p = _safe_path(MODELS_DIR / f"rf_{tgt}.pkl")
        if p is None:
            missing.append(f"rf_{tgt}.pkl")
            models[tgt] = None
            continue
        try:
            models[tgt] = joblib.load(p)
        except Exception as e:
            models[tgt] = None
            missing.append(f"rf_{tgt}.pkl ({e})")
    return models, missing


def model_feature_order(model):
    """Return the exact feature-name order a fitted sklearn model expects.
    Falls back to the canonical [Temperature, Voltage, Conductivity] order
    if the model doesn't expose feature_names_in_ (e.g. older sklearn)."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return ["Temperature", "Voltage", "Conductivity"]


def predict_with_model(model, temperature, voltage, conductivity, return_std=False):
    """Build a single-row DataFrame with the correct column order/names
    for the given model (RF and GPR pipelines were trained with different
    column orders) and return its prediction."""
    values = {"Temperature": temperature, "Voltage": voltage, "Conductivity": conductivity}
    order = model_feature_order(model)
    row = pd.DataFrame([[values[c] for c in order]], columns=order)
    if return_std:
        try:
            pred, std = model.predict(row, return_std=True)
            return float(pred[0]), float(std[0])
        except TypeError:
            pred = model.predict(row)
            return float(pred[0]), None
    pred = model.predict(row)
    return float(pred[0])


# ---------------------------------------------------------------------
# CSV / dataset loading (cached as data)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(filename: str):
    """Load a CSV from the data directory, gracefully returning None
    (and the exception text) if it isn't found or fails to parse."""
    p = _safe_path(DATA_DIR / filename)
    if p is None:
        return None, f"File not found: {filename}"
    try:
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def load_main_dataset():
    df, err = load_csv("dataset.csv")
    return df, err


@st.cache_data(show_spinner=False)
def load_all_reference_data():
    """Bulk-load every reference CSV used across the app. Missing files
    are reported but never crash the app."""
    files = {
        "dataset": "dataset.csv",
        "doe_points": "DOE_points.csv",
        "testdata": "Testdata.csv",
        "testresults_ansys": "Testresults_ansys.csv",
        "testresults_model": "Testresults_model.csv",
        "model_comparison": "model_comparison.csv",
        "rf_5fold": "5fold_RF.csv",
        "rf_external": "RF_External_Validation.csv",
        "gpr_external_preds": "GPRexternal_validation_results.csv",
        "gpr_external_metrics": "GPRmetric_external.csv",
        "gpr_initial_metrics": "inital_GPR_metrics.csv",
        "operating_conditions": "operating_conditions.csv",
        "pareto": "pareto.csv",
    }
    out = {}
    errors = {}
    for key, fname in files.items():
        df, err = load_csv(fname)
        out[key] = df
        if err:
            errors[key] = err
    return out, errors


def get_project_stats():
    """Compute headline statistics shown on the Home dashboard."""
    data, _ = load_all_reference_data()
    dataset = data.get("dataset")
    pareto = data.get("pareto")
    n_sims = len(dataset) if dataset is not None else 312
    n_pareto = len(pareto) if pareto is not None else 200
    n_models = 8  # 4 GPR + 4 RF
    n_targets = 4
    return {
        "n_simulations": n_sims,
        "n_pareto": n_pareto,
        "n_models": n_models,
        "n_targets": n_targets,
    }
