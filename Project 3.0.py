# ─────────────────────────────────────────────────────────────────────────────
# CELL 1: IMPORTS & LOAD TUNED MODELS FROM PHASE 1
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import warnings

os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.matplotlib'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import norm
from scipy.stats.qmc import LatinHypercube

sns.set_theme(style='whitegrid', palette='muted')

# Load everything persisted in Phase 1
nc_model  = joblib.load('models/nc_xgb.pkl')
c_model   = joblib.load('models/c_rf.pkl')
nc_feats  = json.load(open('models/nc_features.json'))
c_feats   = json.load(open('models/c_features.json'))
all_feats = json.load(open('models/all_features.json'))
meta      = json.load(open('models/metadata.json'))

# Reload training data to compute bounds and kNN guard
df_raw = pd.read_csv('data.csv')
df_sub = df_raw.copy()
if 'Sl. No.' in df_sub.columns:
    df_sub = df_sub.drop('Sl. No.', axis=1)
df_sub['is_carburized']    = (df_sub['CT'] == 930).astype(int)
df_sub['is_through_hardened'] = (df_sub['THT'] > 30).astype(int)

df_carb    = df_sub[df_sub['is_carburized'] == 1].reset_index(drop=True)
df_non_carb = df_sub[df_sub['is_carburized'] == 0].reset_index(drop=True)

print("Models loaded from Phase 1.")
print(f"  NC XGBoost  CV RMSE : {meta['nc']['cv_rmse']:.2f} MPa")
print(f"  C  RF       CV RMSE : {meta['c']['cv_rmse']:.2f} MPa")

# Pre-compute OOF residuals for NC to build an empirical error CDF.
# OOF residuals pass through the same fold structure as the reported CV-RMSE,
# so they represent out-of-sample prediction error honestly.
# NC OOF residuals fail Gaussian normality (Shapiro+KS p<0.001); empirical CDF
# is used instead of norm.cdf for NC PoF.  C residuals pass normality → norm.cdf kept.
_nc_cv = KFold(5, shuffle=True, random_state=42)
_nc_oof_preds = cross_val_predict(
    xgb.XGBRegressor(**meta['nc']['best_params'], random_state=42),
    df_non_carb[nc_feats], df_non_carb['Fatigue'],
    cv=_nc_cv, n_jobs=-1
)
NC_OOF_RESIDUALS = df_non_carb['Fatigue'].values - _nc_oof_preds
print(f"  NC OOF residuals computed (N={len(NC_OOF_RESIDUALS)}, "
      f"std={NC_OOF_RESIDUALS.std():.2f} MPa) — used for empirical PoF.")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 2: COST MODEL
# Alloy element cost proxy (USD per kg based on approximate LME prices).
# Only alloying elements that appear in the dataset and affect cost.
# ─────────────────────────────────────────────────────────────────────────────
ELEMENT_COST_USD_PER_KG = {
    'C':  0.50,   # carbon steel — baseline, cheap
    'Si': 1.20,
    'Mn': 1.80,
    'Ni': 14.0,
    'Cr': 10.0,
    'Cu': 9.0,
    'Mo': 30.0,
    'P':  0.0,    # impurity — no cost benefit
    'S':  0.0,
}

# Empirical PoF using pre-computed OOF residuals (used for NC where Gaussian fails).
# P(fatigue < applied) = P(F_hat + epsilon < applied) = P(epsilon < applied - F_hat)
def empirical_pof(applied_stress: np.ndarray, pred_fatigue: np.ndarray,
                  oof_residuals: np.ndarray) -> np.ndarray:
    gap = applied_stress - pred_fatigue   # shape (n_candidates,)
    # for each candidate, fraction of OOF residuals below that gap
    return np.array([(oof_residuals < g).mean() for g in gap])


# Carburizing processing cost proxy (USD per unit time in minutes)
CARB_PROCESS_COST_PER_MIN = 0.05  # relative scale

def compute_alloy_cost(row: pd.Series) -> float:
    """
    Compositional cost = sum(price_i * wt_fraction_i).
    Processing penalty for long carburizing time (Ct).
    Returns a dimensionless cost score (USD per kg of alloy, approximately).
    """
    comp_cost = sum(
        ELEMENT_COST_USD_PER_KG.get(elem, 0.0) * row.get(elem, 0.0)
        for elem in ELEMENT_COST_USD_PER_KG
    )
    proc_cost = CARB_PROCESS_COST_PER_MIN * row.get('Ct', 0.0)
    return comp_cost + proc_cost


# ─────────────────────────────────────────────────────────────────────────────
# CELL 3: CONVEX-HULL GUARD (kNN distance filter)
# Reject candidates that are far from any training point in feature space.
# Threshold = 99th percentile of training kNN distances (generous enough
# to allow neighbourhood exploration without extrapolating wildly).
# ─────────────────────────────────────────────────────────────────────────────
def build_knn_guard(X_train: pd.DataFrame, k: int = 5, percentile: float = 99.0):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    knn.fit(X_scaled)
    dists, _ = knn.kneighbors(X_scaled)
    threshold = np.percentile(dists.mean(axis=1), percentile)
    return knn, scaler, threshold


def in_data_hull(X_candidates: pd.DataFrame, knn, scaler, threshold) -> np.ndarray:
    X_scaled = scaler.transform(X_candidates)
    dists, _ = knn.kneighbors(X_scaled)
    return dists.mean(axis=1) <= threshold


# Build guards for each regime
knn_nc, scaler_nc, thresh_nc = build_knn_guard(df_non_carb[nc_feats])
knn_c,  scaler_c,  thresh_c  = build_knn_guard(df_carb[c_feats])
print(f"kNN guard thresholds — NC: {thresh_nc:.3f}  C: {thresh_c:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 4: NEIGHBOURHOOD SAMPLER
# Perturbs training points with Gaussian noise (scale = fraction of per-feature
# std). Stays naturally inside the data distribution — much better than LHS
# over the bounding box when data is clustered in a subregion of feature space.
# ─────────────────────────────────────────────────────────────────────────────
def get_feature_bounds(df_regime: pd.DataFrame, features: list) -> dict:
    return {f: (df_regime[f].min(), df_regime[f].max()) for f in features}


def neighborhood_sample_candidates(df_train: pd.DataFrame,
                                   features: list,
                                   n_samples: int,
                                   noise_scale: float = 0.20,
                                   seed: int = 42) -> pd.DataFrame:
    # Sample over regime features PLUS every cost-relevant element column so
    # compute_alloy_cost sees the full composition (not just the regime subset).
    cost_cols = list(ELEMENT_COST_USD_PER_KG.keys()) + ['Ct']
    all_cols  = sorted(set(features) | set(c for c in cost_cols if c in df_train.columns))

    rng   = np.random.default_rng(seed)
    idx   = rng.integers(0, len(df_train), size=n_samples)
    base  = df_train[all_cols].values[idx].copy().astype(float)
    stds  = df_train[all_cols].std().values
    noise = rng.normal(0, noise_scale, size=base.shape) * stds
    candidates = base + noise
    # clip to training bounds to avoid extrapolation
    mins = df_train[all_cols].min().values
    maxs = df_train[all_cols].max().values
    candidates = np.clip(candidates, mins, maxs)
    return pd.DataFrame(candidates, columns=all_cols)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 5: INVERSE DESIGN — ALLOY RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────
def recommend_alloys(
    applied_stress_mpa: float,
    target_fos: float  = 1.5,
    regime: str        = 'non_carb',   # 'non_carb' | 'carb'
    n_samples: int     = 25000,
    top_k: int         = 10,
    max_pof_pct: float = 5.0,
) -> pd.DataFrame:
    """
    Inverse design: given target Factor of Safety and applied stress,
    return top_k alloy configurations that satisfy PoF ≤ max_pof_pct
    ranked by alloy cost (cheapest first), with a Pareto summary.

    FoS = predicted_fatigue / applied_stress  (nominal)
    PoF = P(fatigue < applied_stress) = norm.cdf(applied_stress, μ, σ_cv)
    """
    assert regime in ('non_carb', 'carb'), "regime must be 'non_carb' or 'carb'"

    if regime == 'non_carb':
        model   = nc_model
        feats   = nc_feats
        rmse    = meta['nc']['cv_rmse']
        df_trn  = df_non_carb
        knn, scaler, thresh = knn_nc, scaler_nc, thresh_nc
    else:
        model   = c_model
        feats   = c_feats
        rmse    = meta['c']['cv_rmse']
        df_trn  = df_carb
        knn, scaler, thresh = knn_c, scaler_c, thresh_c

    # ── 1. Neighbourhood sample candidates ──────────────────────────────────
    candidates = neighborhood_sample_candidates(df_trn, feats, n_samples)

    # ── 2. kNN hull filter — discard any that drifted outside data region ───
    in_hull    = in_data_hull(candidates[feats], knn, scaler, thresh)
    candidates = candidates[in_hull].reset_index(drop=True)
    print(f"[{regime}] {in_hull.sum():,} / {n_samples:,} candidates inside data hull.")

    if len(candidates) == 0:
        print("No candidates survived hull filter. Try increasing n_samples.")
        return pd.DataFrame()

    # ── 3. Predict fatigue strength ──────────────────────────────────────────
    candidates['pred_fatigue']  = model.predict(candidates[feats])
    candidates['fos']           = candidates['pred_fatigue'] / applied_stress_mpa
    # NC: OOF residuals fail Gaussian normality → use empirical CDF of OOF residuals.
    # C: OOF residuals pass normality → keep Gaussian norm.cdf.
    if regime == 'non_carb':
        candidates['pof_pct'] = empirical_pof(
            np.full(len(candidates), applied_stress_mpa),
            candidates['pred_fatigue'].values,
            NC_OOF_RESIDUALS
        ) * 100
    else:
        candidates['pof_pct'] = norm.cdf(
            applied_stress_mpa,
            loc=candidates['pred_fatigue'],
            scale=rmse
        ) * 100

    # ── 4. Filter by FoS and PoF ─────────────────────────────────────────────
    required_fatigue = target_fos * applied_stress_mpa
    filtered = candidates[
        (candidates['pred_fatigue'] >= required_fatigue) &
        (candidates['pof_pct']      <= max_pof_pct)
    ].copy()

    print(f"[{regime}] {len(filtered):,} candidates satisfy FoS ≥ {target_fos} "
          f"and PoF ≤ {max_pof_pct}%.")

    if len(filtered) == 0:
        print("No feasible alloys found. Relax target_fos or max_pof_pct.")
        return pd.DataFrame()

    # ── 5. Compute cost ──────────────────────────────────────────────────────
    filtered['cost_score'] = filtered.apply(compute_alloy_cost, axis=1)

    # ── 6. Return top-K cheapest feasible alloys ─────────────────────────────
    top = filtered.nsmallest(top_k, 'cost_score').reset_index(drop=True)
    top.index += 1  # 1-indexed rank
    return top, filtered


# ─────────────────────────────────────────────────────────────────────────────
# CELL 6: PARETO FRONT — COST vs PROBABILITY OF FAILURE
# ─────────────────────────────────────────────────────────────────────────────
def pareto_front(df: pd.DataFrame,
                 obj1: str = 'cost_score',
                 obj2: str = 'pof_pct') -> pd.DataFrame:
    """Return Pareto-optimal rows (minimise both obj1 and obj2)."""
    pts = df[[obj1, obj2]].values
    n   = len(pts)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] <= pts[i, 1] and
                    (pts[j, 0] < pts[i, 0] or pts[j, 1] < pts[i, 1])):
                dominated[i] = True
                break
    return df[~dominated].sort_values(obj1).reset_index(drop=True)


def plot_pareto(filtered: pd.DataFrame,
                top: pd.DataFrame,
                pareto: pd.DataFrame,
                regime: str,
                applied_stress: float,
                target_fos: float):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'Alloy Recommender — {regime.replace("_", "-").title()}\n'
        f'Applied stress = {applied_stress} MPa | Target FoS ≥ {target_fos}',
        fontsize=13, fontweight='bold'
    )

    # ── Pareto scatter ────────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(filtered['cost_score'], filtered['pof_pct'],
               s=10, alpha=0.3, color='steelblue', label='Feasible candidates')
    ax.scatter(pareto['cost_score'],  pareto['pof_pct'],
               s=50, color='darkorange', zorder=5, label='Pareto front')
    ax.scatter(top['cost_score'],     top['pof_pct'],
               s=80, marker='*', color='crimson', zorder=6,
               label=f'Top-{len(top)} cheapest')
    ax.set_xlabel('Cost Score (USD/kg, approx)')
    ax.set_ylabel('Probability of Failure (%)')
    ax.set_title('Cost vs PoF — Pareto Front')
    ax.legend(fontsize=9)

    # ── Top-K: predicted fatigue bar chart ───────────────────────────────────
    ax2 = axes[1]
    ranks = top.index.tolist()
    fatigue_vals = top['pred_fatigue'].values
    cost_vals    = top['cost_score'].values
    bars = ax2.barh(ranks, fatigue_vals, color='steelblue', alpha=0.8)
    ax2.axvline(applied_stress, color='crimson', linestyle='--', lw=1.5,
                label=f'Applied stress ({applied_stress} MPa)')
    ax2.axvline(target_fos * applied_stress, color='darkorange', linestyle='--', lw=1.5,
                label=f'Target fatigue ({target_fos}×stress)')
    for bar, cost in zip(bars, cost_vals):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f'${cost:.2f}', va='center', fontsize=8)
    ax2.set_yticks(ranks)
    ax2.set_yticklabels([f'Rank {r}' for r in ranks])
    ax2.set_xlabel('Predicted Fatigue Strength (MPa)')
    ax2.set_title('Top-K Alloys: Fatigue Strength vs Rank')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()
    fname = f'models/pareto_{regime}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# CELL 7: RUN RECOMMENDER — NON-CARBURIZED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  INVERSE DESIGN: NON-CARBURIZED REGIME")
print("  Input: applied_stress=500 MPa, target_fos=1.5, max_pof=2%")
print("═"*65)

APPLIED_STRESS_NC = 500.0
TARGET_FOS_NC     = 1.5

result_nc = recommend_alloys(
    applied_stress_mpa = APPLIED_STRESS_NC,
    target_fos         = TARGET_FOS_NC,
    regime             = 'non_carb',
    n_samples          = 25000,
    top_k              = 10,
    max_pof_pct        = 2.0,
)

if isinstance(result_nc, tuple):
    top_nc, filtered_nc = result_nc
    pareto_nc = pareto_front(filtered_nc)

    print(f"\nTop-10 alloy recommendations (sorted by cost):\n")
    display_cols = nc_feats + ['pred_fatigue', 'fos', 'pof_pct', 'cost_score']
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.3f}'.format)
    print(top_nc[display_cols].to_string())

    print(f"\nPareto front: {len(pareto_nc)} non-dominated alloys.")
    plot_pareto(filtered_nc, top_nc, pareto_nc,
                'non_carb', APPLIED_STRESS_NC, TARGET_FOS_NC)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 8: RUN RECOMMENDER — CARBURIZED
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*65)
print("  INVERSE DESIGN: CARBURIZED REGIME")
print("  Input: applied_stress=700 MPa, target_fos=1.4, max_pof=2%")
print("═"*65)

APPLIED_STRESS_C = 700.0
TARGET_FOS_C     = 1.4

result_c = recommend_alloys(
    applied_stress_mpa = APPLIED_STRESS_C,
    target_fos         = TARGET_FOS_C,
    regime             = 'carb',
    n_samples          = 25000,
    top_k              = 10,
    max_pof_pct        = 2.0,
)

if isinstance(result_c, tuple):
    top_c, filtered_c = result_c
    pareto_c = pareto_front(filtered_c)

    print(f"\nTop-10 alloy recommendations (sorted by cost):\n")
    display_cols_c = c_feats + ['pred_fatigue', 'fos', 'pof_pct', 'cost_score']
    print(top_c[display_cols_c].to_string())

    print(f"\nPareto front: {len(pareto_c)} non-dominated alloys.")
    plot_pareto(filtered_c, top_c, pareto_c,
                'carb', APPLIED_STRESS_C, TARGET_FOS_C)


# ─────────────────────────────────────────────────────────────────────────────
# CELL 9: COST BREAKDOWN — TOP-1 ALLOY PER REGIME
# ─────────────────────────────────────────────────────────────────────────────
def print_cost_breakdown(row: pd.Series, label: str):
    print(f"\n{'─'*50}")
    print(f" Cost Breakdown: {label}")
    print(f"{'─'*50}")
    total = 0.0
    for elem, price in sorted(ELEMENT_COST_USD_PER_KG.items(), key=lambda x: -x[1]):
        if elem in row and price > 0:
            contribution = price * row[elem]
            total += contribution
            print(f"  {elem:>3}  {row[elem]:.4f} wt%  × ${price:>5}/kg = ${contribution:.4f}")
    if 'Ct' in row:
        proc = CARB_PROCESS_COST_PER_MIN * row['Ct']
        total += proc
        print(f"  Ct   {row['Ct']:.1f} min  × ${CARB_PROCESS_COST_PER_MIN}/min = ${proc:.4f}")
    print(f"  {'─'*38}")
    print(f"  Total cost score : ${total:.4f} / kg")
    print(f"  Pred. fatigue    : {row['pred_fatigue']:.1f} MPa")
    print(f"  FoS              : {row['fos']:.3f}")
    print(f"  PoF              : {row['pof_pct']:.4f}%")


if isinstance(result_nc, tuple):
    print_cost_breakdown(top_nc.iloc[0], "Non-Carburized Rank-1 Alloy")

if isinstance(result_c, tuple):
    print_cost_breakdown(top_c.iloc[0], "Carburized Rank-1 Alloy")
