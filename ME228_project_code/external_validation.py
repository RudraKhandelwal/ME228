"""
External validation harness — three legs:

1. Stratified NIMS holdout — train on 80%, test on 20%.  Compares against CV.
2. matbench_steels compositional cross-dataset — predict relative rank of YS
   for maraging steels using composition-only features.  Ground-truth is YS,
   not fatigue, so we expect a systematic offset; we test rank correlation.
3. Handbook canonical grades — already in crosscheck.py::step_external.
   Re-run here for a single consolidated table.
"""
import json, joblib, warnings, re
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

MATBENCH_FALLBACK = {
    'spearman_rho': -0.04,
    'spearman_p': 0.44,
    'pearson_r': -0.35,
}


def load_nims():
    df = pd.read_csv('data.csv').drop('Sl. No.', axis=1)
    df['is_carburized'] = (df['CT'] == 930).astype(int)
    df['is_through_hardened'] = (df['THT'] > 30).astype(int)
    return df


def parse_matbench_formula(formula: str) -> dict:
    """Parse Fe0.62C0.001Mn0.0005... into wt% dict.  Atom fractions in source;
    convert to weight fractions then multiply by 100 for wt%."""
    atomic_wt = {
        'Fe': 55.85, 'C': 12.01, 'Si': 28.09, 'Mn': 54.94, 'P': 30.97, 'S': 32.07,
        'Ni': 58.69, 'Cr': 52.00, 'Cu': 63.55, 'Mo': 95.96, 'V': 50.94, 'Co': 58.93,
        'Nb': 92.91, 'W': 183.84, 'Al': 26.98, 'Ti': 47.87, 'N': 14.01, 'Ta': 180.95,
        'Zr': 91.22, 'B': 10.81, 'Ce': 140.12,
    }
    pat = re.compile(r'([A-Z][a-z]?)(\d*\.?\d+)')
    atoms = {el: float(v) for el, v in pat.findall(formula)}
    masses = {el: f * atomic_wt.get(el, 50.0) for el, f in atoms.items()}
    total = sum(masses.values())
    return {el: (m / total) * 100.0 for el, m in masses.items()}  # wt%


def step_holdout():
    print("=" * 65)
    print(" STEP 1: STRATIFIED NIMS HOLDOUT (80% train / 20% test)")
    print("=" * 65)
    df = load_nims()
    nc_feats = json.load(open('models/nc_features.json'))
    c_feats  = json.load(open('models/c_features.json'))
    meta     = json.load(open('models/metadata.json'))
    rng = np.random.default_rng(42)

    for regime, df_r, feats, model_cls, params in [
        ('NC', df[df.is_carburized==0].reset_index(drop=True), nc_feats,
         xgb.XGBRegressor, meta['nc']['best_params']),
        ('C',  df[df.is_carburized==1].reset_index(drop=True), c_feats,
         RandomForestRegressor, meta['c']['best_params']),
    ]:
        # stratify on fatigue quintiles to balance the train/test split
        bins = pd.qcut(df_r['Fatigue'], q=min(5, len(df_r)//8), labels=False, duplicates='drop')
        Xtr, Xte, ytr, yte = train_test_split(
            df_r[feats], df_r['Fatigue'],
            test_size=0.20, stratify=bins, random_state=42
        )
        m = model_cls(**params, random_state=42)
        if model_cls is RandomForestRegressor:
            m.set_params(n_jobs=-1)
        m.fit(Xtr, ytr)
        pred = m.predict(Xte)
        rmse = np.sqrt(mean_squared_error(yte, pred))
        r2 = r2_score(yte, pred)
        # CV for comparison
        cv = KFold(5, shuffle=True, random_state=42)
        cv_rmse = -cross_val_score(model_cls(**params, random_state=42, **({'n_jobs':-1} if model_cls is RandomForestRegressor else {})),
                                   df_r[feats], df_r['Fatigue'], cv=cv,
                                   scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        print(f"\n[{regime}]  N_train={len(Xtr)}  N_test={len(Xte)}")
        print(f"  Holdout RMSE = {rmse:.2f} MPa   R² = {r2:.4f}")
        print(f"  CV RMSE     = {cv_rmse:.2f} MPa   ratio = {rmse/cv_rmse:.2f}x")


def step_matbench():
    print("\n" + "=" * 65)
    print(" STEP 2: matbench_steels CROSS-DATASET — composition-only rank test")
    print("=" * 65)
    try:
        from matminer.datasets import load_dataset
    except ImportError:
        print("  matminer unavailable, using cached benchmark:")
        print(f"    Spearman ρ = {MATBENCH_FALLBACK['spearman_rho']:+.3f}  (p={MATBENCH_FALLBACK['spearman_p']:.4f})")
        print(f"    Pearson  r = {MATBENCH_FALLBACK['pearson_r']:+.3f}  (cached prior result)")
        return
    mb = load_dataset('matbench_steels')
    nims = load_nims()

    # Parse compositions
    parsed = mb['composition'].apply(parse_matbench_formula)
    el_set = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Cu', 'Mo']
    mb_comp = pd.DataFrame([{e: row.get(e, 0.0) for e in el_set} for row in parsed])
    mb_comp['YS'] = mb['yield strength'].values

    # No filter: matbench is dominantly maraging steels (different population from NIMS),
    # but composition signal can still rank-correlate.  We test rank, not absolute value.
    mb_in = mb_comp.reset_index(drop=True)
    print(f"\n  matbench_steels: total={len(mb_in)} (no compositional filter; OOD test)")

    # Train a composition-only XGBoost on NIMS NC fatigue (reduced to comp + a default heat-treat row)
    # Simpler approach: train on compositional features alone, predict fatigue, check rank vs YS
    nims_nc = nims[nims.is_carburized == 0].reset_index(drop=True)
    Xc = nims_nc[el_set]
    y = nims_nc['Fatigue']
    m = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(Xc, y)
    pred_fatigue = m.predict(mb_in[el_set])
    # Rank correlation between predicted fatigue and observed YS
    sp_r, sp_p = spearmanr(pred_fatigue, mb_in['YS'])
    pe_r, pe_p = pearsonr(pred_fatigue, mb_in['YS'])
    print(f"\n  Composition-only NIMS model predicts fatigue on matbench grades:")
    print(f"    pred_fatigue: mean={pred_fatigue.mean():.0f}  std={pred_fatigue.std():.0f}  "
          f"range=[{pred_fatigue.min():.0f}, {pred_fatigue.max():.0f}] MPa")
    print(f"    matbench YS : mean={mb_in.YS.mean():.0f}  std={mb_in.YS.std():.0f}  "
          f"range=[{mb_in.YS.min():.0f}, {mb_in.YS.max():.0f}] MPa")
    print(f"    Spearman ρ = {sp_r:+.3f}  (p={sp_p:.4f})")
    print(f"    Pearson  r = {pe_r:+.3f}  (p={pe_p:.4f})")
    print(f"  Interpretation: ρ > 0.3 → composition signal transfers; ρ < 0.1 → no transfer.")


def step_handbook():
    print("\n" + "=" * 65)
    print(" STEP 3: HANDBOOK CANONICAL GRADES (consolidated)")
    print("=" * 65)
    nc_model = joblib.load('models/nc_xgb.pkl')
    c_model  = joblib.load('models/c_rf.pkl')
    nc_feats = json.load(open('models/nc_features.json'))
    c_feats  = json.load(open('models/c_features.json'))
    meta     = json.load(open('models/metadata.json'))

    grades = {
        '1045_normalized':
            (False, dict(NT=870, THT=845, THt=30, THQCr=12, CT=30, Ct=0, DT=30, Dt=0, QmT=30,
                         TT=550, Tt=60, TCr=20, C=0.45, Si=0.25, Mn=0.75, P=0.020, S=0.020,
                         Ni=0.05, Cr=0.10, Cu=0.10, Mo=0.02, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
             (260, 340)),
        '4140_QT':
            (False, dict(NT=870, THT=845, THt=30, THQCr=12, CT=30, Ct=0, DT=30, Dt=0, QmT=30,
                         TT=540, Tt=60, TCr=20, C=0.40, Si=0.25, Mn=0.85, P=0.020, S=0.020,
                         Ni=0.05, Cr=0.95, Cu=0.10, Mo=0.20, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
             (450, 620)),
        '4340_QT':
            (False, dict(NT=870, THT=845, THt=30, THQCr=12, CT=30, Ct=0, DT=30, Dt=0, QmT=30,
                         TT=540, Tt=60, TCr=20, C=0.40, Si=0.25, Mn=0.70, P=0.020, S=0.020,
                         Ni=1.80, Cr=0.80, Cu=0.10, Mo=0.25, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
             (520, 660)),
        '8620_carb':
            (True,  dict(NT=870, THT=30, THt=0, THQCr=0, CT=930, Ct=240, DT=850, Dt=60, QmT=60,
                         TT=170, Tt=60, TCr=20, C=0.20, Si=0.25, Mn=0.80, P=0.020, S=0.020,
                         Ni=0.55, Cr=0.55, Cu=0.10, Mo=0.20, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
             (700, 1000)),
    }
    rows = []
    for name, (is_carb, fd, (lo, hi)) in grades.items():
        fd['is_through_hardened'] = int(fd['THT'] > 30)
        if is_carb:
            X = pd.DataFrame([fd])[c_feats]
            pred = c_model.predict(X)[0]; sigma = meta['c']['cv_rmse']; lab='C'
        else:
            X = pd.DataFrame([fd])[nc_feats]
            pred = nc_model.predict(X)[0]; sigma = meta['nc']['cv_rmse']; lab='NC'
        in_range = lo <= pred <= hi
        off = 0 if in_range else min(abs(pred-lo), abs(pred-hi))
        rows.append((name, lab, pred, sigma, lo, hi, 'OK' if in_range else ('LOW' if pred<lo else 'HIGH'), off))
    print(f"\n  {'Grade':18} {'Reg':4} {'Pred':>7} {'σ':>6} {'Low':>5} {'High':>5} {'Status':6} {'Off':>6}")
    for r in rows:
        print(f"  {r[0]:18} {r[1]:4} {r[2]:7.1f} {r[3]:6.1f} {r[4]:5} {r[5]:5} {r[6]:6} {r[7]:6.1f}")


if __name__ == '__main__':
    step_holdout()
    step_matbench()
    step_handbook()
