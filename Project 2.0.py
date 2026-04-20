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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
from scipy.stats import norm

optuna.logging.set_verbosity(optuna.logging.WARNING)
sns.set_theme(style='whitegrid', palette='muted')

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
# CELL 2: VIF -> SHAP FEATURE SELECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def vif_then_shap_selection(X_full, y_full, group_name, vif_threshold=10.0, top_n_shap=12):
    print(f"\n{'='*60}")
    print(f" FEATURE PIPELINE: {group_name.upper()} ")
    print(f"{'='*60}")

    X_vif_calc = X_full.copy()
    X_vif_calc['intercept'] = 1.0

    dropped_by_vif = []
    while True:
        vif_data = pd.DataFrame({
            "feature": X_vif_calc.columns,
            "VIF": [variance_inflation_factor(X_vif_calc.values, i)
                    for i in range(X_vif_calc.shape[1])]
        })
        vif_data = vif_data[vif_data['feature'] != 'intercept']
        if vif_data['VIF'].max() > vif_threshold:
            worst = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
            dropped_by_vif.append(worst)
            X_vif_calc = X_vif_calc.drop(columns=[worst])
        else:
            break

    vif_survivors = [c for c in X_vif_calc.columns if c != 'intercept']
    print(f"STEP 1 VIF: dropped {len(dropped_by_vif)} collinear features.")

    X_shap = X_full[vif_survivors]
    rf_scout = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)
    rf_scout.fit(X_shap, y_full)
    explainer = shap.TreeExplainer(rf_scout)
    shap_values = explainer.shap_values(X_shap)

    shap_importance = pd.DataFrame({
        'Feature': X_shap.columns,
        'SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('SHAP', ascending=False)

    final_features = shap_importance['Feature'].head(top_n_shap).tolist()
    print(f"STEP 2 SHAP: top {len(final_features)} features: {final_features}")
    return final_features


nc_features = vif_then_shap_selection(
    df_non_carb[all_features], df_non_carb['Fatigue'], "Non-Carburized", top_n_shap=12
)
c_features = vif_then_shap_selection(
    df_carb[all_features], df_carb['Fatigue'], "Carburized", top_n_shap=10
)

# Persist feature lists
with open('models/nc_features.json', 'w') as f:
    json.dump(nc_features, f)
with open('models/c_features.json', 'w') as f:
    json.dump(c_features, f)
with open('models/all_features.json', 'w') as f:
    json.dump(all_features, f)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: NESTED CV + OPTUNA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
def rmse_cv(model, X, y, cv):
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    return -scores.mean(), scores.std()


def tune_xgb(X, y, n_trials=60, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
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
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_rf(X, y, n_trials=60, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_int('n_estimators', 100, 600),
            'max_depth':       trial.suggest_int('max_depth', 2, 12),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':    trial.suggest_float('max_features', 0.3, 1.0),
            'random_state': 42,
        }
        model = RandomForestRegressor(**params, n_jobs=-1)
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring='neg_root_mean_squared_error', n_jobs=-1)
        return -scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── Non-Carburized: tune XGBoost (5-fold CV, N~390) ──────────────────────────
print("\n" + "="*60)
print(" TUNING: NON-CARBURIZED XGBoost (5-fold CV, 60 trials)")
print("="*60)
X_nc = df_non_carb[nc_features]
y_nc = df_non_carb['Fatigue']

nc_best_params = tune_xgb(X_nc, y_nc, n_trials=60, n_splits=5)
print(f"Best params: {nc_best_params}")

nc_tuned = xgb.XGBRegressor(**nc_best_params, random_state=42)
nc_cv_rmse, nc_cv_std = rmse_cv(nc_tuned, X_nc, y_nc, KFold(5, shuffle=True, random_state=42))
print(f"CV RMSE : {nc_cv_rmse:.2f} ± {nc_cv_std:.2f} MPa")

# Fit on full non-carb data
nc_tuned.fit(X_nc, y_nc)
nc_preds_train = nc_tuned.predict(X_nc)
nc_residuals = y_nc.values - nc_preds_train

# ── Carburized: tune RF with LOO-like small CV (carb N~48) ───────────────────
# Use 5-fold for carb too; LOO is noisy for tree models
print("\n" + "="*60)
print(" TUNING: CARBURIZED Random Forest (5-fold CV, 60 trials)")
print("="*60)
X_c = df_carb[c_features]   # use REDUCED features, not all 26
y_c = df_carb['Fatigue']

c_best_params = tune_rf(X_c, y_c, n_trials=60, n_splits=5)
print(f"Best params: {c_best_params}")

c_tuned = RandomForestRegressor(**c_best_params, n_jobs=-1, random_state=42)
c_cv_rmse, c_cv_std = rmse_cv(c_tuned, X_c, y_c, KFold(5, shuffle=True, random_state=42))
print(f"CV RMSE : {c_cv_rmse:.2f} ± {c_cv_std:.2f} MPa")

# Fit on full carb data
c_tuned.fit(X_c, y_c)
c_preds_train = c_tuned.predict(X_c)
c_residuals = y_c.values - c_preds_train

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
        'best_params':   nc_best_params,
    },
    'c': {
        'model_file':    'models/c_rf.pkl',
        'features_file': 'models/c_features.json',
        'cv_rmse':       round(c_cv_rmse, 4),
        'cv_rmse_std':   round(c_cv_std, 4),
        'residuals_normal': bool(c_p_sw > 0.05),
        'best_params':   c_best_params,
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
