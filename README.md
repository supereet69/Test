# PEMTwin AI — PEM Electrolyser Digital Twin Dashboard

An AI-powered digital twin dashboard for PEM electrolyser performance prediction,
explainability, and multi-objective optimisation. Built with Streamlit, reusing
the pre-trained Gaussian Process Regression (GPR) and Random Forest (RF)
surrogate models, optimisation results, and validation artefacts from the
source project (see `AI-Driven Optimization of Electrolyzer Performance`
report). **No models are retrained by this app** — it only loads and serves
the existing `.pkl` files and CSV outputs.

## Project structure

```
pemtwin_dashboard/
├── app.py                  # Main entry point — top navigation & routing
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml         # Light theme, blue accent
├── assets/
│   └── style.css           # Industrial white/blue visual theme
├── utils/
│   ├── data_loader.py      # Cached model & CSV loading (path-independent)
│   ├── styling.py          # CSS injection + reusable UI components
│   └── helpers.py          # Health scoring, interpretation text, gauges
├── pages_modules/
│   ├── home.py             # Home Dashboard
│   ├── digital_twin.py     # Interactive GPR prediction page
│   ├── explainable_ai.py   # Live SHAP analysis on the RF models
│   ├── optimization.py     # NSGA-II Pareto front & recommendations
│   ├── validation.py       # Cross-validation & external validation
│   └── about.py            # Methodology, software, authors, references
├── models/                 # Copied .pkl surrogate models (GPR + RF)
└── data/                   # Copied reference CSVs (dataset, validation, Pareto...)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app resolves all model/data paths relative to its own folder
(`utils/data_loader.py`), so it can be moved or deployed anywhere without
code changes.

## Notes on the underlying models

- **GPR models** (`models/gpr_*.pkl`) are `sklearn.Pipeline` objects
  (`StandardScaler` → `GaussianProcessRegressor` with a `Constant × RBF +
  WhiteKernel`). They expose `feature_names_in_ = ['Temperature', 'Voltage',
  'Conductivity']`.
- **RF models** (`models/rf_*.pkl`) are `RandomForestRegressor` objects with
  `feature_names_in_ = ['Voltage', 'Conductivity', 'Temperature']`.
- Because the two families were trained with **different column orders**,
  `utils/data_loader.predict_with_model()` always builds a named
  `pandas.DataFrame` matching each model's own `feature_names_in_` before
  calling `.predict()`, so predictions are correct regardless of which
  model is used.
