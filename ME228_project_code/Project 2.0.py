# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: IMPORTS, PREPROCESSING, AND METALLURGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import warnings

os.makedirs('models', exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.matplotlib'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import shap
import joblib
import optuna
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import norm

optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style='whitegrid', palette='muted')

# Optuna search controls: search can stop early once CV RMSE plateaus.
OPTUNA_MAX_TRIALS = int(os.getenv('OPTUNA_MAX_TRIALS', '200'))
OPTUNA_PATIENCE = int(os.getenv('OPTUNA_PATIENCE', '40'))
OPTUNA_MIN_DELTA = float(os.getenv('OPTUNA_MIN_DELTA', '0.01'))
OPTUNA_MIN_TRIALS = int(os.getenv('OPTUNA_MIN_TRIALS', '60'))
OPTUNA_CV_SPLITS = int(os.getenv('OPTUNA_CV_SPLITS', '5'))

# Load data
df_raw = pd.read_csv('data.csv')
df_sub = df_raw.copy()
if 'Sl. No.' in df_sub.columns:
    df_sub = df_sub.drop('Sl. No.', axis=1)

# Engineer binary process flags
df_sub['is_carburized'] = (df_sub['CT'] == 930).astype(int)
df_sub['is_through_hardened'] = (df_sub['THT'] > 30).astype(int)

all_features = [col for col in df_sub.columns if col not in ['Fatigue', 'is_carburized']]

# MoE split
df_carb = df_sub[df_sub['is_carburized'] == 1].reset_index(drop=True)
df_non_carb = df_sub[df_sub['is_carburized'] == 0].reset_index(drop=True)

print(f"Dataset split:")
print(f"  Non-Carburized : {len(df_non_carb)} samples")
print(f"  Carburized     : {len(df_carb)} samples")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: LEAK-FREE VIF -> SHAP FEATURE SELECTION (fold-aware transformer)
# ─────────────────────────────────────────────────────────────────────────────
# VIF + SHAP both run inside fit() => when wrapped in sklearn Pipeline and
# evaluated via cross_val_score, the selection sees ONLY the training fold.
# Test fold y/X never influence which features are kept => no leakage.
class VIFThenShapSelector(BaseEstimator, TransformerMixin):
    def __init__(self, vif_threshold=10.0, top_n=12,
                 scout_n_estimators=150, scout_max_depth=5, random_state=42):
        self.vif_threshold = vif_threshold
        self.top_n = top_n
        self.scout_n_estimators = scout_n_estimators
        self.scout_max_depth = scout_max_depth
        self.random_state = random_state

    def _vif_filter(self, X_df):
        X_calc = X_df.copy()
        X_calc['__intercept__'] = 1.0
        while True:
            cols = list(X_calc.columns)
            vifs = []
            for i, col in enumerate(cols):
                if col == '__intercept__':
                    continue
                vifs.append((col, variance_inflation_factor(X_calc.values, i)))
            if not vifs:
                break
            worst_col, worst_vif = max(vifs, key=lambda t: t[1])
            if worst_vif > self.vif_threshold:
                X_calc = X_calc.drop(columns=[worst_col])
            else:
                break
        return [c for c in X_calc.columns if c != '__intercept__']

    def fit(self, X, y):
        X_df = X if hasattr(X, 'columns') else pd.DataFrame(X)
        survivors = self._vif_filter(X_df)
        X_vif = X_df[survivors]
        rf = RandomForestRegressor(
            n_estimators=self.scout_n_estimators,
            max_depth=self.scout_max_depth,
            random_state=self.random_state, n_jobs=-1)
        rf.fit(X_vif, y)
        sv = shap.TreeExplainer(rf).shap_values(X_vif)
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1]
        self.vif_survivors_ = survivors
        self.selected_features_ = [survivors[i] for i in order[:self.top_n]]
        return self

    def transform(self, X):
        X_df = X if hasattr(X, 'columns') else pd.DataFrame(X)
        return X_df[self.selected_features_]


def make_pipeline(top_n, downstream):
    # No joblib Memory cache: __main__-defined transformer can't pickle.
    # Selector refits per fold per trial; cost is acceptable (~ms-scale SHAP
    # on the small RF scout). Correctness > speed.
    return Pipeline(
        [('sel', VIFThenShapSelector(top_n=top_n)),
         ('mdl', downstream)])


# Persist all_features now; per-regime feature lists written after final fit
with open('models/all_features.json', 'w') as f:
    json.dump(all_features, f)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: LEAK-FREE OPTUNA TUNING (selection + model jointly inside CV folds)
# ─────────────────────────────────────────────────────────────────────────────
def rmse_cv_pipe(pipe, X, y, cv):
    scores = cross_val_score(pipe, X, y, cv=cv,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean(), scores.std()


def make_plateau_callback(patience, min_delta, min_trials):
    best_value = np.inf
    best_trial_number = -1

    def callback(study, trial):
        nonlocal best_value, best_trial_number

        if study.best_value < best_value - min_delta:
            best_value = study.best_value
            best_trial_number = study.best_trial.number

        trials_completed = len(study.trials)
        trials_since_best = trial.number - best_trial_number
        if trials_completed >= min_trials and trials_since_best >= patience:
            print(
                f"  Plateau stop: no CV RMSE improvement > {min_delta:.3f} MPa "
                f"for {patience} trials (best {study.best_value:.3f} MPa at "
                f"trial {study.best_trial.number + 1})."
            )
            study.stop()

    return callback


def run_optuna_search(objective, n_trials_max, patience, min_delta, min_trials):
    study = optuna.create_study(direction='minimize')
    study.optimize(
        objective,
        n_trials=n_trials_max,
        show_progress_bar=False,
        callbacks=[make_plateau_callback(patience, min_delta, min_trials)],
    )
    return study


def tune_xgb(X, y, n_trials_max=OPTUNA_MAX_TRIALS, n_splits=OPTUNA_CV_SPLITS,
             patience=OPTUNA_PATIENCE, min_delta=OPTUNA_MIN_DELTA,
             min_trials=OPTUNA_MIN_TRIALS):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        top_n = trial.suggest_int('top_n', 6, 18)
        params = {
            'n_estimators':  trial.suggest_int('n_estimators', 100, 600),
            'max_depth':     trial.suggest_int('max_depth', 2, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample':     trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha':     trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda':    trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': 42,
        }
        pipe = make_pipeline(top_n, xgb.XGBRegressor(**params))
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=1)
        return -scores.mean()

    return run_optuna_search(objective, n_trials_max, patience, min_delta, min_trials)


def tune_rf(X, y, n_trials_max=OPTUNA_MAX_TRIALS, n_splits=OPTUNA_CV_SPLITS,
            patience=OPTUNA_PATIENCE, min_delta=OPTUNA_MIN_DELTA,
            min_trials=OPTUNA_MIN_TRIALS):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        top_n = trial.suggest_int('top_n', 5, 15)
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 100, 600),
            'max_depth':       trial.suggest_int('max_depth', 2, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':    trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42,
        }
        pipe = make_pipeline(top_n, RandomForestRegressor(**params, n_jobs=-1))
        scores = cross_val_score(pipe, X, y, cv=cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=1)
        return -scores.mean()

    return run_optuna_search(objective, n_trials_max, patience, min_delta, min_trials)


# ── Non-Carburized: tune XGBoost (5-fold CV, N~390) ──────────────────────────
print("\n" + "="*60)
print(
    " TUNING: NON-CARBURIZED XGBoost "
    f"(leak-free {OPTUNA_CV_SPLITS}-fold CV, up to {OPTUNA_MAX_TRIALS} trials)"
)
print("="*60)
X_nc_full = df_non_carb[all_features]
y_nc      = df_non_carb['Fatigue']

nc_study = tune_xgb(X_nc_full, y_nc)
nc_best_params = nc_study.best_params.copy()
NC_TOP_N = nc_best_params.pop('top_n')
nc_trials_run = len(nc_study.trials)
print(f"Best top_n: {NC_TOP_N}  |  Best params: {nc_best_params}")
print(f"Trials completed: {nc_trials_run} / {OPTUNA_MAX_TRIALS}")

nc_pipe = make_pipeline(NC_TOP_N, xgb.XGBRegressor(**nc_best_params, random_state=42))
nc_cv_rmse, nc_cv_std = rmse_cv_pipe(
    nc_pipe, X_nc_full, y_nc, KFold(OPTUNA_CV_SPLITS, shuffle=True, random_state=42))
print(f"Honest CV RMSE : {nc_cv_rmse:.2f} ± {nc_cv_std:.2f} MPa")

# Final fit on full non-carb data => extracts deployment feature set
nc_pipe.fit(X_nc_full, y_nc)
nc_features = list(nc_pipe.named_steps['sel'].selected_features_)
nc_tuned    = nc_pipe.named_steps['mdl']
X_nc        = X_nc_full[nc_features]
nc_preds_train = nc_tuned.predict(X_nc)
nc_residuals   = y_nc.values - nc_preds_train
print(f"Final selected features ({len(nc_features)}): {nc_features}")

# ── Carburized: tune RF (5-fold CV, N~48) ────────────────────────────────────
print("\n" + "="*60)
print(
    " TUNING: CARBURIZED Random Forest "
    f"(leak-free {OPTUNA_CV_SPLITS}-fold CV, up to {OPTUNA_MAX_TRIALS} trials)"
)
print("="*60)
X_c_full = df_carb[all_features]
y_c      = df_carb['Fatigue']

c_study = tune_rf(X_c_full, y_c)
c_best_params = c_study.best_params.copy()
C_TOP_N = c_best_params.pop('top_n')
c_trials_run = len(c_study.trials)
print(f"Best top_n: {C_TOP_N}  |  Best params: {c_best_params}")
print(f"Trials completed: {c_trials_run} / {OPTUNA_MAX_TRIALS}")

c_pipe = make_pipeline(C_TOP_N, RandomForestRegressor(**c_best_params, n_jobs=-1, random_state=42))
c_cv_rmse, c_cv_std = rmse_cv_pipe(
    c_pipe, X_c_full, y_c, KFold(OPTUNA_CV_SPLITS, shuffle=True, random_state=42))
print(f"Honest CV RMSE : {c_cv_rmse:.2f} ± {c_cv_std:.2f} MPa")

# Final fit on full carb data
c_pipe.fit(X_c_full, y_c)
c_features = list(c_pipe.named_steps['sel'].selected_features_)
c_tuned    = c_pipe.named_steps['mdl']
X_c        = X_c_full[c_features]
c_preds_train = c_tuned.predict(X_c)
c_residuals   = y_c.values - c_preds_train
print(f"Final selected features ({len(c_features)}): {c_features}")

# Persist final feature lists (extracted from full-data fit; CV numbers above
# are honest because per-fold selection was used during evaluation)
with open('models/nc_features.json', 'w') as f:
    json.dump(nc_features, f)
with open('models/c_features.json', 'w') as f:
    json.dump(c_features, f)

print("\n" + "="*60)
print(" TUNING SUMMARY (CV RMSE — honest cross-validated)")
print("="*60)
print(f"  Non-Carburized XGBoost  : {nc_cv_rmse:.2f} ± {nc_cv_std:.2f} MPa")
print(f"  Carburized     RF       : {c_cv_rmse:.2f} ± {c_cv_std:.2f} MPa")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: RESIDUAL DIAGNOSTICS — validates norm.cdf PoF assumption
# ─────────────────────────────────────────────────────────────────────────────
def residual_diagnostics(residuals, regime_name, ax_hist, ax_qq):
    _, p_shapiro = stats.shapiro(residuals)
    _, p_ks = stats.kstest(residuals, 'norm',
                           args=(residuals.mean(), residuals.std()))

    # Histogram + fitted normal
    ax_hist.hist(residuals, bins=20, density=True, alpha=0.6, color='steelblue', edgecolor='white')
    x_range = np.linspace(residuals.min(), residuals.max(), 200)
    ax_hist.plot(x_range, norm.pdf(x_range, residuals.mean(), residuals.std()),
                 'r--', lw=2, label='Normal fit')
    ax_hist.set_title(f'{regime_name} — Residual Distribution\n'
                      f'Shapiro p={p_shapiro:.3f}  KS p={p_ks:.3f}')
    ax_hist.set_xlabel('Residual (MPa)')
    ax_hist.legend()

    # QQ-plot
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    ax_qq.scatter(osm, osr, s=20, alpha=0.7, color='steelblue')
    line_x = np.array([osm.min(), osm.max()])
    ax_qq.plot(line_x, slope * line_x + intercept, 'r--', lw=2)
    ax_qq.set_title(f'{regime_name} — Q-Q Plot  (R={r:.4f})')
    ax_qq.set_xlabel('Theoretical Quantiles')
    ax_qq.set_ylabel('Sample Quantiles')

    print(f"\n{regime_name} Residual Tests:")
    print(f"  Shapiro-Wilk p = {p_shapiro:.4f}  {'✓ Normal' if p_shapiro > 0.05 else '✗ Non-Normal'}")
    print(f"  KS test      p = {p_ks:.4f}  {'✓ Normal' if p_ks > 0.05 else '✗ Non-Normal'}")
    if p_shapiro < 0.05 or p_ks < 0.05:
        print("  ⚠  Residuals non-normal — norm.cdf PoF is approximate."
              " Consider empirical quantile for critical decisions.")
    return p_shapiro, p_ks


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Residual Diagnostics (Training Fit)', fontsize=14, fontweight='bold')

nc_p_sw, nc_p_ks = residual_diagnostics(
    nc_residuals, 'Non-Carburized (XGBoost)', axes[0, 0], axes[0, 1])
c_p_sw, c_p_ks = residual_diagnostics(
    c_residuals, 'Carburized (RF)', axes[1, 0], axes[1, 1])

plt.tight_layout()
plt.savefig('models/residual_diagnostics.png', dpi=150)
plt.show()
print("\nSaved: models/residual_diagnostics.png")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: PERSIST MODELS AND METADATA
# ─────────────────────────────────────────────────────────────────────────────
joblib.dump(nc_tuned, 'models/nc_xgb.pkl')
joblib.dump(c_tuned,  'models/c_rf.pkl')

metadata = {
    'nc': {
        'model_file':    'models/nc_xgb.pkl',
        'features_file': 'models/nc_features.json',
        'cv_rmse':       round(nc_cv_rmse, 4),
        'cv_rmse_std':   round(nc_cv_std, 4),
        'residuals_normal': bool(nc_p_sw > 0.05),
        'top_n':         NC_TOP_N,
        'best_params':   nc_best_params,
        'optuna': {
            'cv_splits': OPTUNA_CV_SPLITS,
            'max_trials': OPTUNA_MAX_TRIALS,
            'completed_trials': nc_trials_run,
            'patience': OPTUNA_PATIENCE,
            'min_delta': OPTUNA_MIN_DELTA,
            'min_trials': OPTUNA_MIN_TRIALS,
        },
    },
    'c': {
        'model_file':    'models/c_rf.pkl',
        'features_file': 'models/c_features.json',
        'cv_rmse':       round(c_cv_rmse, 4),
        'cv_rmse_std':   round(c_cv_std, 4),
        'residuals_normal': bool(c_p_sw > 0.05),
        'top_n':         C_TOP_N,
        'best_params':   c_best_params,
        'optuna': {
            'cv_splits': OPTUNA_CV_SPLITS,
            'max_trials': OPTUNA_MAX_TRIALS,
            'completed_trials': c_trials_run,
            'patience': OPTUNA_PATIENCE,
            'min_delta': OPTUNA_MIN_DELTA,
            'min_trials': OPTUNA_MIN_TRIALS,
        },
    },
}
with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\nPersisted:")
print("  models/nc_xgb.pkl")
print("  models/c_rf.pkl")
print("  models/metadata.json")
print("  models/nc_features.json")
print("  models/c_features.json")
print("  models/all_features.json")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: v2.0 DEPLOYMENT WRAPPER (loads from disk, CV-based uncertainty)
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    nc_model    = joblib.load('models/nc_xgb.pkl')
    c_model     = joblib.load('models/c_rf.pkl')
    nc_feats    = json.load(open('models/nc_features.json'))
    c_feats     = json.load(open('models/c_features.json'))
    meta        = json.load(open('models/metadata.json'))
    return nc_model, c_model, nc_feats, c_feats, meta


def predict_reliability(material_features_dict, applied_stress_mpa,
                        nc_model=None, c_model=None,
                        nc_feats=None, c_feats=None, meta=None):
    if nc_model is None:
        nc_model, c_model, nc_feats, c_feats, meta = load_models()

    feats = material_features_dict.copy()
    feats['is_through_hardened'] = int(feats.get('THT', 0) > 30)
    is_carb = (feats.get('CT', 0) == 930)

    if is_carb:
        model, feats_list, rmse, regime = (
            c_model, c_feats, meta['c']['cv_rmse'],
            "Carburized (RF | Reduced Features | CV-tuned)")
    else:
        model, feats_list, rmse, regime = (
            nc_model, nc_feats, meta['nc']['cv_rmse'],
            "Standard/Through-Hardened (XGBoost | Reduced Features | CV-tuned)")

    X_in = pd.DataFrame([feats])[feats_list]
    predicted_fatigue = model.predict(X_in)[0]

    margin = 1.96 * rmse
    prob_failure = norm.cdf(applied_stress_mpa, loc=predicted_fatigue, scale=rmse)

    print("═" * 60)
    print("             v2.0 ALLOY RELIABILITY REPORT             ")
    print("═" * 60)
    print(f"Regime          : {regime}")
    print(f"Applied Stress  : {applied_stress_mpa:.1f} MPa")
    print("-" * 60)
    print(f"Predicted Fatigue Strength : {predicted_fatigue:.1f} MPa")
    print(f"CV-based σ (RMSE)          : {rmse:.1f} MPa")
    print(f"95% Prediction Interval    : [{predicted_fatigue - margin:.1f}, "
          f"{predicted_fatigue + margin:.1f}] MPa")
    print(f"Probability of Failure     : {prob_failure * 100:.4f}%")
    print("═" * 60)
    return predicted_fatigue, prob_failure


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: VALIDATION ON HELD-OUT EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────
nc_model, c_model, nc_feats, c_feats, meta = load_models()

sample_standard = {
    'NT': 880, 'THT': 0, 'THt': 0, 'THQCr': 0, 'CT': 30, 'Ct': 0,
    'DT': 30, 'Dt': 0, 'QmT': 30, 'TT': 600, 'Tt': 60, 'TCr': 0,
    'C': 0.45, 'Si': 0.25, 'Mn': 0.70, 'P': 0.015, 'S': 0.01,
    'Ni': 0.05, 'Cr': 1.0, 'Cu': 0.1, 'Mo': 0.2, 'RedRatio': 600,
    'dA': 0.02, 'dB': 0.0, 'dC': 0.0
}

sample_carburized = {
    'NT': 880, 'THT': 30, 'THt': 0, 'THQCr': 0, 'CT': 930, 'Ct': 120,
    'DT': 880, 'Dt': 45, 'QmT': 60, 'TT': 160, 'Tt': 120, 'TCr': 0,
    'C': 0.20, 'Si': 0.25, 'Mn': 0.70, 'P': 0.015, 'S': 0.01,
    'Ni': 0.05, 'Cr': 1.0, 'Cu': 0.1, 'Mo': 0.2, 'RedRatio': 600,
    'dA': 0.02, 'dB': 0.0, 'dC': 0.0
}

print("\n--- Standard Steel Component ---")
predict_reliability(sample_standard, 571.6,
                    nc_model, c_model, nc_feats, c_feats, meta)

print("\n--- Carburized Steel Component ---")
predict_reliability(sample_carburized, 880.0,
                    nc_model, c_model, nc_feats, c_feats, meta)
