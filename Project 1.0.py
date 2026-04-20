# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: IMPORTS, PREPROCESSING, AND METALLURGICAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
import os
import warnings

os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.matplotlib'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm

sns.set_theme(style='whitegrid', palette='muted')

# 1. Load Data
df_raw = pd.read_csv('data.csv')
df_sub = df_raw.copy()
if 'Sl. No.' in df_sub.columns:
    df_sub = df_sub.drop('Sl. No.', axis=1)

# 2. Engineer Binary Process Flags
df_sub['is_carburized'] = (df_sub['CT'] == 930).astype(int)
df_sub['is_through_hardened'] = (df_sub['THT'] > 30).astype(int)

# 3. Define Base Feature Set (Everything except targets and the constant split flag)
all_features = [col for col in df_sub.columns if col not in ['Fatigue', 'log_Fatigue', 'is_carburized']]

# 4. Split into Mixture of Experts Sub-Populations
df_carb = df_sub[df_sub['is_carburized'] == 1]
df_non_carb = df_sub[df_sub['is_carburized'] == 0]

print(f"Dataset split successfully:")
print(f"  - Non-Carburized Samples : {len(df_non_carb)}")
print(f"  - Carburized Samples     : {len(df_carb)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: VIF -> SHAP FEATURE SELECTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def vif_then_shap_selection(X_full, y_full, group_name, vif_threshold=10.0, top_n_shap=12):
    print(f"\n{'='*60}")
    print(f" FEATURE PIPELINE: {group_name.upper()} ")
    print(f"{'='*60}")

    # --- STEP 1: VIF FILTER ---
    X_vif_calc = X_full.copy()
    X_vif_calc['intercept'] = 1.0 # Required for statsmodels VIF

    dropped_by_vif = []
    while True:
        vif_data = pd.DataFrame({
            "feature": X_vif_calc.columns,
            "VIF": [variance_inflation_factor(X_vif_calc.values, i) for i in range(X_vif_calc.shape[1])]
        })
        vif_data = vif_data[vif_data['feature'] != 'intercept']

        if vif_data['VIF'].max() > vif_threshold:
            worst_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
            dropped_by_vif.append(worst_feature)
            X_vif_calc = X_vif_calc.drop(columns=[worst_feature])
        else:
            break

    vif_survivors = [col for col in X_vif_calc.columns if col != 'intercept']
    print(f"STEP 1: VIF Filter (Threshold = {vif_threshold})")
    print(f"  * Dropped {len(dropped_by_vif)} highly correlated features.")

    # --- STEP 2: SHAP FILTER (The "Scout" Model) ---
    X_shap = X_full[vif_survivors]
    rf_scout = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)
    rf_scout.fit(X_shap, y_full)

    explainer = shap.TreeExplainer(rf_scout)
    shap_values = explainer.shap_values(X_shap)

    shap_importance = pd.DataFrame({
        'Feature': X_shap.columns,
        'SHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values(by='SHAP', ascending=False)

    final_features = shap_importance['Feature'].head(top_n_shap).tolist()
    print(f"STEP 2: SHAP Filter")
    print(f"  * Final Top {len(final_features)} Features: {final_features}")

    return final_features

# Extract the optimal feature sets for both regimes
nc_features = vif_then_shap_selection(
    df_non_carb[all_features], df_non_carb['Fatigue'], "Non-Carburized", top_n_shap=12
)

c_features = vif_then_shap_selection(
    df_carb[all_features], df_carb['Fatigue'], "Carburized", top_n_shap=10
)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: BENCHMARKING EXPERT MODELS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_expert(model, X, y, test_size, model_name, feature_set_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{model_name:<15} | {feature_set_name:<16} | RMSE: {rmse:>5.2f} MPa | R²: {r2:.4f}")
    return model, rmse

print("==========================================================")
print(f"       NON-CARBURIZED REGIME (N={len(df_non_carb)}, Test Size=0.20)      ")
print("==========================================================")
y_nc = df_non_carb['Fatigue']

# XGBoost
xgb_nc_all = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
evaluate_expert(xgb_nc_all, df_non_carb[all_features], y_nc, 0.20, "XGBoost", "All Features")

xgb_nc_red = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
final_nc_model, final_nc_rmse = evaluate_expert(xgb_nc_red, df_non_carb[nc_features], y_nc, 0.20, "XGBoost", "Reduced Features")

# Random Forest
rf_nc_all = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=4, random_state=42)
evaluate_expert(rf_nc_all, df_non_carb[all_features], y_nc, 0.20, "Random Forest", "All Features")

rf_nc_red = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=4, random_state=42)
evaluate_expert(rf_nc_red, df_non_carb[nc_features], y_nc, 0.20, "Random Forest", "Reduced Features")


print("\n==========================================================")
print(f"         CARBURIZED REGIME (N={len(df_carb)}, Test Size=0.15)         ")
print("==========================================================")
y_c = df_carb['Fatigue']

# XGBoost (Shallower depth for small data)
xgb_c_all = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
evaluate_expert(xgb_c_all, df_carb[all_features], y_c, 0.15, "XGBoost", "All Features")

xgb_c_red = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)
evaluate_expert(xgb_c_red, df_carb[c_features], y_c, 0.15, "XGBoost", "Reduced Features")

# Random Forest (Shallower depth for small data)
rf_c_all = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=2, random_state=42)
final_c_model, final_c_rmse = evaluate_expert(rf_c_all, df_carb[all_features], y_c, 0.15, "Random Forest", "All Features")

rf_c_red = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_split=2, random_state=42)
evaluate_expert(rf_c_red, df_carb[c_features], y_c, 0.15, "Random Forest", "Reduced Features")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: v1.0 DEPLOYMENT WRAPPER (PROBABILITY OF FAILURE)
# ─────────────────────────────────────────────────────────────────────────────
def predict_tool_reliability(material_features_dict, applied_stress_mpa):
    """
    Intelligently routes the material to the correct expert model based on processing.
    """
    is_carb = (material_features_dict.get('CT', 0) == 930)
    material_features_dict['is_through_hardened'] = int(material_features_dict.get('THT', 0) > 30)

    input_df = pd.DataFrame([material_features_dict])

    if is_carb:
        # Route to Carburized Expert (Winner: Random Forest, All Features)
        active_model = final_c_model
        active_rmse = final_c_rmse
        input_df = input_df[all_features]
        regime = "Carburized (Random Forest | All Features)"
    else:
        # Route to Non-Carburized Expert (Winner: XGBoost, Reduced Features)
        active_model = final_nc_model
        active_rmse = final_nc_rmse
        input_df = input_df[nc_features]
        regime = "Standard/Through-Hardened (XGBoost | Reduced Features)"

    predicted_fatigue = active_model.predict(input_df)[0]

    # Calculate bounds and Factor of Safety / Probability of Failure
    margin_of_error = 1.96 * active_rmse
    lower_bound = predicted_fatigue - margin_of_error
    upper_bound = predicted_fatigue + margin_of_error
    prob_failure = norm.cdf(applied_stress_mpa, loc=predicted_fatigue, scale=active_rmse)

    print("═"*60)
    print("             v1.0 ALLOY RELIABILITY REPORT             ")
    print("═"*60)
    print(f"Expert Regime Selected     : {regime}")
    print(f"Applied Operational Stress : {applied_stress_mpa:.1f} MPa")
    print("-" * 60)
    print(f"Predicted Fatigue Strength : {predicted_fatigue:.1f} MPa")
    print(f"95% Prediction Interval    : [{lower_bound:.1f} MPa, {upper_bound:.1f} MPa]")
    print(f"Probability of Failure     : {prob_failure * 100:.4f}%")
    print("═"*60)

    return predicted_fatigue, prob_failure

# Test Example
sample_part = {
    'NT': 880, 'THT': 0, 'THt': 0, 'THQCr': 0, 'CT': 30, 'Ct': 0,
    'DT': 30, 'Dt': 0, 'QmT': 30, 'TT': 600, 'Tt': 60, 'TCr': 0,
    'C': 0.45, 'Si': 0.25, 'Mn': 0.70, 'P': 0.015, 'S': 0.01,
    'Ni': 0.05, 'Cr': 1.0, 'Cu': 0.1, 'Mo': 0.2, 'RedRatio': 600,
    'dA': 0.02, 'dB': 0.0, 'dC': 0.0
}
_ = predict_tool_reliability(sample_part, applied_stress_mpa=550)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: TEST SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────

# Example 1: A Non-Carburized part (CT = 30) under high stress
sample_standard = {
    'NT': 880, 'THT': 0, 'THt': 0, 'THQCr': 0, 'CT': 30, 'Ct': 0,
    'DT': 30, 'Dt': 0, 'QmT': 30, 'TT': 600, 'Tt': 60, 'TCr': 0,
    'C': 0.45, 'Si': 0.25, 'Mn': 0.70, 'P': 0.015, 'S': 0.01,
    'Ni': 0.05, 'Cr': 1.0, 'Cu': 0.1, 'Mo': 0.2, 'RedRatio': 600,
    'dA': 0.02, 'dB': 0.0, 'dC': 0.0
}

# The model predicts ~480 MPa for this alloy. Let's apply 400 MPa of stress.
print("\n--- Testing Standard Steel Component ---")
predict_tool_reliability(sample_standard, applied_stress_mpa=571.6)


# Example 2: A Carburized part (CT = 930) under massive stress
sample_carburized = {
    'NT': 880, 'THT': 30, 'THt': 0, 'THQCr': 0, 'CT': 930, 'Ct': 120,
    'DT': 880, 'Dt': 45, 'QmT': 60, 'TT': 160, 'Tt': 120, 'TCr': 0,
    'C': 0.20, 'Si': 0.25, 'Mn': 0.70, 'P': 0.015, 'S': 0.01,
    'Ni': 0.05, 'Cr': 1.0, 'Cu': 0.1, 'Mo': 0.2, 'RedRatio': 600,
    'dA': 0.02, 'dB': 0.0, 'dC': 0.0
}

# The model predicts ~950 MPa for this alloy. Let's apply 880 MPa of stress.
print("\n--- Testing Carburized Steel Component ---")
predict_tool_reliability(sample_carburized, applied_stress_mpa=880)