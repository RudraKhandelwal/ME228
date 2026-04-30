# ME228 — Fatigue Strength Predictor & Alloy Recommender

Final project for **ME228: Applied Data Science and Machine Learning**.
Two-branch ML pipeline that predicts the rotating-bending fatigue strength of
low-alloy steels from chemistry + heat-treatment parameters, recommends
cost-efficient alloys via Pareto search, and ships as an interactive
Streamlit GUI.

## Highlights

- **Two-branch model.** Carburized (C) and non-carburized (NC) steels are
  modelled separately because their residual distributions and feature
  importances differ.
  - NC branch: XGBoost — CV RMSE **19.91 ± 2.55 MPa** (n≈340)
  - C  branch: Random Forest — CV RMSE **40.42 ± 14.62 MPa** (n≈97)
- **Probability-of-failure (PoF) overlay.** Empirical CDF for NC (residuals
  fail normality), Gaussian for C (residuals pass Shapiro + KS).
- **Inverse design.** kNN-guarded grid search over heat-treatment +
  chemistry space, ranked by a cost model that includes Ni/Cr/Mo/Cu price.
- **External validation.** OOD checks against MatBench-style plain-C grades,
  hull-periphery sensitivity, seed-stability of top-K bands.
- **Interactive GUI.** SHAP explanations, what-if sliders, Pareto front.

See [report_final.pdf](report_final.pdf) for the writeup and
[CROSSCHECK.md](CROSSCHECK.md) for the independent verification log.

## Repo layout

```
app.py                  Streamlit GUI (Phase 4 — final)
Project 1.0.py          Phase 1 — EDA + first regressor
Project 2.0.py          Phase 2 — two-branch pipeline + tuning
Project 3.0.py          Phase 3 — inverse design + cost model (bug-fixed)
crosscheck.py           Independent reproduction of headline numbers
external_validation.py  OOD + hull-periphery + seed-stability tests
theory_check.py         VC / Hoeffding sanity checks
gen_slide_plots.py      Regenerates figures used in the report and slides
data.csv                Source dataset (NIMS-style fatigue table)
models/                 Trained pickles, feature lists, metadata, diagnostics
slides_plots/           PNGs used in the report and final slide deck
report_final.{tex,pdf}  Final report
report.{tex,pdf}        Earlier draft retained for comparison
progress_report_1.pdf   Mid-semester progress report
CROSSCHECK.md           Severity-tagged findings + remediation log
```

## Running the GUI

```bash
python -m venv .venv && source .venv/bin/activate
pip install streamlit pandas numpy scikit-learn xgboost joblib shap scipy matplotlib
streamlit run app.py
```

Models are loaded from `models/` and the dataset from `data.csv` — both are
checked in, so no retraining step is required to use the GUI.

## Reproducing the numbers

```bash
python crosscheck.py            # CV RMSE, residual tests, top-K stability
python external_validation.py   # OOD + hull-periphery + matbench-style checks
python theory_check.py          # VC dimension / Hoeffding bounds
python gen_slide_plots.py       # Regenerates slides_plots/*.png
```

## Author

Rudra Khandelwal — IIT Bombay, ME228 (Spring 2026).
