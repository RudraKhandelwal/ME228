# ME228 — Fatigue Strength Prediction & Inverse Alloy Design

## Requirements

```
pip install numpy pandas scikit-learn xgboost optuna scipy joblib matplotlib seaborn
```

Python 3.9+ recommended.

---

## File Overview

| File | Purpose |
|------|---------|
| `data.csv` | NIMS MatNavi Fatigue Dataset (N=437, 26 features + target) |
| `Project 2.0.py` | **Main pipeline** — feature selection, plateau-stopped Optuna tuning, model training, OOF diagnostics, and artefact persistence |
| `gen_slide_plots.py` | Regenerates all 10 `slides_plots/` figures used in the report |
| `theory_check.py` | VC dimension bookkeeping, Hoeffding bounds, bootstrap RMSE CI, course-baseline benchmark |
| `external_validation.py` | Out-of-distribution probe (matbench) and canonical AISI grade checks |
| `app.py` | Streamlit interactive app (forward prediction + PoF + inverse design) |

### Pre-trained artefacts (`models/`)

| File | Contents |
|------|----------|
| `nc_xgb.pkl` | Tuned XGBoost for non-carburised regime |
| `c_rf.pkl` | Tuned Random Forest for carburised regime |
| `nc_features.json` | 16 selected features for NC expert |
| `c_features.json` | 7 selected features for C expert |
| `metadata.json` | CV-RMSE, best hyperparameters, and Optuna search metadata for both experts |

### Generated figures (`slides_plots/`)

Produced by `gen_slide_plots.py`; consumed by `report_concise.tex`.

---

## How to Run

### 1. Retrain from scratch (slow — adaptive Optuna, up to 200 trials each)

```bash
python "Project 2.0.py"
```

Overwrites `models/` artefacts and `slides_plots/` figures.

### 2. Theory checks (fast — uses saved models)

```bash
python theory_check.py
```

Prints VC table, Hoeffding bounds, bootstrap CI, course-baseline RMSE table.

### 3. Regenerate figures only

```bash
python gen_slide_plots.py
```

### 4. External validation

```bash
python external_validation.py
```

### 5. Interactive app

```bash
streamlit run app.py
```

---

## Pipeline Summary

1. **Regime gate** — hard split on `CT == 930` (carburised vs. non-carburised).
2. **Leak-free feature selection** — `VIFThenShapSelector` runs VIF filter + SHAP scout *inside* each CV fold; no leakage from test folds.
3. **Hyperparameter tuning** — Optuna TPE, 5-fold CV per regime, with plateau stopping after the CV RMSE stops improving.
4. **OOF evaluation** — 5-fold OOF residuals used for Shapiro-Wilk normality test; empirical CDF (NC) or Gaussian (C) for PoF.
5. **External validation** — stratified 20 % holdout + canonical AISI grade spot-check.
6. **Inverse design** — neighbourhood sampling → kNN data-hull guard (p99) → Pareto front over (cost, PoF).
