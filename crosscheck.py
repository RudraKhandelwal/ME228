# Cross-check script. Runs incrementally via __main__ args.
import json, joblib, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler as SS
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from scipy import stats
from scipy.stats import norm
import xgboost as xgb

# ---------- shared loaders ----------
def load_all():
    df = pd.read_csv('data.csv').drop('Sl. No.', axis=1)
    df['is_carburized'] = (df['CT'] == 930).astype(int)
    df['is_through_hardened'] = (df['THT'] > 30).astype(int)
    nc_feats = json.load(open('models/nc_features.json'))
    c_feats  = json.load(open('models/c_features.json'))
    meta     = json.load(open('models/metadata.json'))
    nc_model = joblib.load('models/nc_xgb.pkl')
    c_model  = joblib.load('models/c_rf.pkl')
    dnc = df[df['is_carburized']==0].reset_index(drop=True)
    dc  = df[df['is_carburized']==1].reset_index(drop=True)
    return df, dnc, dc, nc_feats, c_feats, meta, nc_model, c_model

PRICES = {'C':0.5,'Si':1.2,'Mn':1.8,'Ni':14.0,'Cr':10.0,'Cu':9.0,'Mo':30.0,'P':0.0,'S':0.0}
PROC = 0.05
ALL_PRICED = list(PRICES.keys())

# ---------- helpers ----------
def neighborhood_full(df_train, regime_feats, n=25000, noise=0.20, seed=42):
    rng = np.random.default_rng(seed)
    union = sorted(set(regime_feats) | set(ALL_PRICED) | {'Ct'})
    union = [c for c in union if c in df_train.columns]
    idx = rng.integers(0, len(df_train), size=n)
    base = df_train[union].values[idx].copy().astype(float)
    stds = df_train[union].std().values
    cand = base + rng.normal(0, noise, size=base.shape) * stds
    cand = np.clip(cand, df_train[union].min().values, df_train[union].max().values)
    return pd.DataFrame(cand, columns=union)

def knn_guard(X, k=5, p=99.0):
    s = StandardScaler().fit(X)
    Xs = s.transform(X)
    knn = NearestNeighbors(n_neighbors=k).fit(Xs)
    d, _ = knn.kneighbors(Xs)
    return knn, s, np.percentile(d.mean(axis=1), p)

def cost_full(row):
    return sum(PRICES.get(e,0)*row.get(e,0.0) for e in PRICES) + PROC*row.get('Ct',0.0)

def cost_buggy(row, feats):
    cols = set(feats) | {'Ct'}
    return sum(PRICES.get(e,0)*row.get(e,0.0) for e in PRICES if e in cols) + PROC*(row.get('Ct',0.0) if 'Ct' in cols else 0.0)


# ---------- step 3: cost-bug demo ----------
def step_cost_bug():
    print('=== STEP 3: COST-MODEL BUG DEMO ===')
    _, dnc, dc, nc_feats, c_feats, meta, nc_model, c_model = load_all()

    for name, df_t, model, feats, applied, fos, max_pof, rmse in [
        ('NC', dnc, nc_model, nc_feats, 500.0, 1.5, 2.0, meta['nc']['cv_rmse']),
        ('C',  dc,  c_model,  c_feats,  700.0, 1.4, 2.0, meta['c']['cv_rmse'])]:

        cand = neighborhood_full(df_t, feats)
        knn, sc, th = knn_guard(df_t[feats])
        Xs = sc.transform(cand[feats])
        d, _ = knn.kneighbors(Xs)
        cand = cand[d.mean(axis=1) <= th].reset_index(drop=True)
        cand['pred_fatigue'] = model.predict(cand[feats])
        cand['pof_pct'] = norm.cdf(applied, loc=cand['pred_fatigue'], scale=rmse)*100
        fil = cand[(cand['pred_fatigue']>=fos*applied)&(cand['pof_pct']<=max_pof)].copy()

        if len(fil) == 0:
            print(f'[{name}] no feasible'); continue

        fil['cost_buggy'] = fil.apply(lambda r: cost_buggy(r, feats), axis=1)
        fil['cost_fixed'] = fil.apply(cost_full, axis=1)

        top_b = fil.nsmallest(10, 'cost_buggy').reset_index(drop=True)
        top_f = fil.nsmallest(10, 'cost_fixed').reset_index(drop=True)

        # how different are the lists?
        b_set = {tuple(r.round(3)) for r in top_b[feats].values}
        f_set = {tuple(r.round(3)) for r in top_f[feats].values}
        overlap = len(b_set & f_set)
        print(f'\n[{name}] feasible={len(fil)}, top-10 overlap={overlap}/10')
        b1 = top_b.iloc[0]; f1 = top_f.iloc[0]
        print(f'  Buggy top-1: cost_buggy=${b1.cost_buggy:.3f}  cost_FIXED=${b1.cost_fixed:.3f}  '
              f'Ni={b1.get("Ni",0):.3f}  Cr={b1.get("Cr",0):.3f}  C={b1.get("C",0):.3f}  Mo={b1.get("Mo",0):.3f}  '
              f'fatigue={b1.pred_fatigue:.0f}')
        print(f'  Fixed top-1: cost_buggy=${f1.cost_buggy:.3f}  cost_FIXED=${f1.cost_fixed:.3f}  '
              f'Ni={f1.get("Ni",0):.3f}  Cr={f1.get("Cr",0):.3f}  C={f1.get("C",0):.3f}  Mo={f1.get("Mo",0):.3f}  '
              f'fatigue={f1.pred_fatigue:.0f}')
        gap = b1.cost_fixed - f1.cost_fixed
        print(f'  --> True cost penalty of using buggy top-1: ${gap:.3f}/kg ({100*gap/f1.cost_fixed:.1f}% inflation)')


# ---------- step 4: MoE split sensitivity ----------
def step_moe_split():
    print('\n=== STEP 4: MoE SPLIT SENSITIVITY ===')
    df, dnc, dc, nc_feats, c_feats, meta, _, _ = load_all()
    cv = KFold(5, shuffle=True, random_state=42)
    nc_p = meta['nc']['best_params']; c_p = meta['c']['best_params']

    # current split
    nc_m = xgb.XGBRegressor(**nc_p, random_state=42)
    c_m  = RandomForestRegressor(**c_p, n_jobs=-1, random_state=42)
    nc_rmse = -cross_val_score(nc_m, dnc[nc_feats], dnc['Fatigue'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
    c_rmse  = -cross_val_score(c_m,  dc[c_feats],  dc['Fatigue'],  cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
    pooled_rmse = np.sqrt((len(dnc)*nc_rmse**2 + len(dc)*c_rmse**2)/(len(dnc)+len(dc)))
    print(f'A. Current  CT==930:    NC={nc_rmse:.2f}  C={c_rmse:.2f}  pooled={pooled_rmse:.2f}')

    # global single model
    all_feats = json.load(open('models/all_features.json'))
    g_m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42)
    g_rmse = -cross_val_score(g_m, df[all_feats], df['Fatigue'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
    print(f'B. Global XGB:           {g_rmse:.2f}')

    # alt: split on THT == 30 (untreated/carb on one side, TH on other)
    s1 = df[df['THT'] != 30].reset_index(drop=True)  # through-hardened
    s2 = df[df['THT'] == 30].reset_index(drop=True)  # untreated + carb
    if len(s2) > 30:
        m1 = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42)
        m2 = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        r1 = -cross_val_score(m1, s1[all_feats], s1['Fatigue'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        r2 = -cross_val_score(m2, s2[all_feats], s2['Fatigue'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        pooled = np.sqrt((len(s1)*r1**2 + len(s2)*r2**2)/(len(s1)+len(s2)))
        print(f'C. Split THT==30:        TH={r1:.2f} (N={len(s1)})  notTH={r2:.2f} (N={len(s2)})  pooled={pooled:.2f}')

    # alt: 3-class — untreated / TH / carb
    seg_un = df[(df['THT']==30) & (df['CT']==30)].reset_index(drop=True)
    seg_th = df[(df['THT']!=30) & (df['CT']==30)].reset_index(drop=True)
    seg_cb = df[df['CT']==930].reset_index(drop=True)
    print(f'D. 3-class sizes: untreated={len(seg_un)}, TH={len(seg_th)}, carb={len(seg_cb)}')
    rmses = []
    for nm, s in [('TH', seg_th), ('carb', seg_cb)]:
        if len(s) < 5: continue
        m = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42) if nm=='TH' \
            else RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=-1, random_state=42)
        r = -cross_val_score(m, s[all_feats], s['Fatigue'], cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1).mean()
        rmses.append((nm, r, len(s)))
        print(f'   {nm}: RMSE={r:.2f} (N={len(s)})')
    # untreated too small for CV; skip


# ---------- step 5: model family bench-off ----------
def step_models():
    print('\n=== STEP 5: MODEL FAMILY BENCH-OFF (5-fold CV, current features) ===')
    _, dnc, dc, nc_feats, c_feats, _, _, _ = load_all()
    cv = KFold(5, shuffle=True, random_state=42)
    candidates = {
        'XGB-default':  lambda: xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42),
        'RF-default':   lambda: RandomForestRegressor(n_estimators=400, max_depth=8, n_jobs=-1, random_state=42),
        'GBM-sk':       lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
        'Ridge':        lambda: Pipeline([('s', SS()), ('m', Ridge(alpha=1.0))]),
        'kNN-7':        lambda: Pipeline([('s', SS()), ('m', KNeighborsRegressor(n_neighbors=7))]),
        'GP-Matern':    lambda: Pipeline([('s', SS()), ('m', GaussianProcessRegressor(
                            kernel=Matern(nu=2.5)+WhiteKernel(), normalize_y=True, n_restarts_optimizer=2, random_state=42))]),
    }
    try:
        import lightgbm as lgb
        candidates['LGBM'] = lambda: lgb.LGBMRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, random_state=42, verbose=-1)
    except Exception:
        print('  (lightgbm unavailable, skip)')

    for regime, X, y in [('NC', dnc[nc_feats], dnc['Fatigue']), ('C', dc[c_feats], dc['Fatigue'])]:
        print(f'\nRegime: {regime}  (N={len(X)})')
        for name, mk in candidates.items():
            try:
                m = mk()
                s = cross_val_score(m, X, y, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=-1)
                print(f'  {name:14}  RMSE = {-s.mean():6.2f} ± {s.std():5.2f}')
            except Exception as e:
                print(f'  {name:14}  ERROR: {e.__class__.__name__}: {str(e)[:60]}')


# ---------- step 6: log transform ----------
def step_log_transform():
    print('\n=== STEP 6: LOG-TRANSFORM TARGET ===')
    _, dnc, dc, nc_feats, c_feats, meta, _, _ = load_all()
    cv = KFold(5, shuffle=True, random_state=42)

    for regime, df_t, feats, params, model_cls in [
        ('NC', dnc, nc_feats, meta['nc']['best_params'], xgb.XGBRegressor),
        ('C',  dc,  c_feats,  meta['c']['best_params'],  RandomForestRegressor)]:
        y = df_t['Fatigue'].values
        ylog = np.log1p(y)
        # Original
        m1 = model_cls(**params, random_state=42)
        if model_cls is RandomForestRegressor:
            m1.set_params(n_jobs=-1)
        oof_orig = cross_val_predict(m1, df_t[feats], y, cv=cv, n_jobs=-1)
        rmse_orig = np.sqrt(((y - oof_orig)**2).mean())
        # Log
        m2 = model_cls(**params, random_state=42)
        if model_cls is RandomForestRegressor:
            m2.set_params(n_jobs=-1)
        oof_log = cross_val_predict(m2, df_t[feats], ylog, cv=cv, n_jobs=-1)
        oof_unlog = np.expm1(oof_log)
        rmse_log = np.sqrt(((y - oof_unlog)**2).mean())
        print(f'  {regime}: original RMSE={rmse_orig:.2f}  log-target RMSE (back-transformed)={rmse_log:.2f}')


# ---------- step 7: kNN hull threshold sweep ----------
def step_hull_sweep():
    print('\n=== STEP 7: kNN HULL THRESHOLD SWEEP ===')
    _, dnc, dc, nc_feats, c_feats, meta, nc_model, c_model = load_all()
    for regime, df_t, model, feats, applied, fos, max_pof, rmse in [
        ('NC', dnc, nc_model, nc_feats, 500.0, 1.5, 2.0, meta['nc']['cv_rmse']),
        ('C',  dc,  c_model,  c_feats,  700.0, 1.4, 2.0, meta['c']['cv_rmse'])]:
        cand = neighborhood_full(df_t, feats)
        cand['pred_fatigue'] = model.predict(cand[feats])
        cand['pof_pct'] = norm.cdf(applied, loc=cand['pred_fatigue'], scale=rmse)*100
        cand['cost'] = cand.apply(cost_full, axis=1)
        scaler = StandardScaler().fit(df_t[feats])
        knn = NearestNeighbors(n_neighbors=5).fit(scaler.transform(df_t[feats]))
        Xs = scaler.transform(cand[feats])
        dists, _ = knn.kneighbors(Xs)
        mean_d = dists.mean(axis=1)
        train_d, _ = knn.kneighbors(scaler.transform(df_t[feats]))
        train_md = train_d.mean(axis=1)
        print(f'\n[{regime}] feasible base set (no hull): {((cand.pred_fatigue>=fos*applied)&(cand.pof_pct<=max_pof)).sum()}')
        for p in [80, 90, 95, 99, 99.5]:
            th = np.percentile(train_md, p)
            ok = mean_d <= th
            f = cand[ok & (cand.pred_fatigue>=fos*applied) & (cand.pof_pct<=max_pof)]
            if len(f)>0:
                top1 = f.nsmallest(1, 'cost').iloc[0]
                print(f'  p{p:5}  th={th:.3f}  pass_hull={ok.sum():>5}  feasible={len(f):>5}  '
                      f'top1: cost=${top1.cost:.2f}  fatigue={top1.pred_fatigue:.0f}')


# ---------- step 8: inverse design seed stability ----------
def step_seed_stability():
    print('\n=== STEP 8: INVERSE DESIGN SEED STABILITY ===')
    _, dnc, dc, nc_feats, c_feats, meta, nc_model, c_model = load_all()
    for regime, df_t, model, feats, applied, fos, max_pof, rmse in [
        ('NC', dnc, nc_model, nc_feats, 500.0, 1.5, 2.0, meta['nc']['cv_rmse']),
        ('C',  dc,  c_model,  c_feats,  700.0, 1.4, 2.0, meta['c']['cv_rmse'])]:
        knn, sc, th = knn_guard(df_t[feats])
        top1_costs = []; top1_fatigues = []; top10_signatures = []
        for seed in range(10):
            cand = neighborhood_full(df_t, feats, seed=seed)
            d, _ = knn.kneighbors(sc.transform(cand[feats]))
            cand = cand[d.mean(axis=1) <= th].reset_index(drop=True)
            cand['pred_fatigue'] = model.predict(cand[feats])
            cand['pof_pct'] = norm.cdf(applied, loc=cand['pred_fatigue'], scale=rmse)*100
            cand['cost'] = cand.apply(cost_full, axis=1)
            f = cand[(cand.pred_fatigue>=fos*applied)&(cand.pof_pct<=max_pof)]
            if len(f) == 0: continue
            top10 = f.nsmallest(10, 'cost')
            top1_costs.append(top10.iloc[0].cost)
            top1_fatigues.append(top10.iloc[0].pred_fatigue)
            sig = tuple(sorted(round(v, 2) for v in top10['cost'].values))
            top10_signatures.append(sig)
        print(f'\n[{regime}] over 10 seeds:')
        print(f'  top-1 cost     : mean=${np.mean(top1_costs):.3f}  std=${np.std(top1_costs):.3f}  range=[${min(top1_costs):.3f}, ${max(top1_costs):.3f}]')
        print(f'  top-1 fatigue  : mean={np.mean(top1_fatigues):.1f}  std={np.std(top1_fatigues):.1f}  range=[{min(top1_fatigues):.0f}, {max(top1_fatigues):.0f}]')
        # Jaccard between top-10 cost-sets across seeds
        from itertools import combinations
        jac = []
        sets = [set(round(v,1) for v in s) for s in top10_signatures]
        for a, b in combinations(sets, 2):
            jac.append(len(a&b)/max(1,len(a|b)))
        print(f'  top-10 cost-set Jaccard (pairwise): mean={np.mean(jac):.2f}  min={min(jac):.2f}')


# ---------- step 9: external validation ----------
def step_external():
    print('\n=== STEP 9: EXTERNAL VALIDATION (canonical grades) ===')
    _, dnc, dc, nc_feats, c_feats, meta, nc_model, c_model = load_all()

    # Handbook compositions (mid-range) + plausible heat-treat for AISI grades
    grades = {
        # AISI 1045: med-carbon plain steel, normalized + tempered
        '1045_normalized': dict(NT=870, THT=845, THt=30, THQCr=12, CT=30, Ct=0, DT=30, Dt=0, QmT=30,
                                TT=550, Tt=60, TCr=20, C=0.45, Si=0.25, Mn=0.75, P=0.020, S=0.020,
                                Ni=0.05, Cr=0.10, Cu=0.10, Mo=0.02, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
        # AISI 4140: Cr-Mo, quenched-and-tempered
        '4140_QT':        dict(NT=870, THT=845, THt=30, THQCr=12, CT=30, Ct=0, DT=30, Dt=0, QmT=30,
                                TT=540, Tt=60, TCr=20, C=0.40, Si=0.25, Mn=0.85, P=0.020, S=0.020,
                                Ni=0.05, Cr=0.95, Cu=0.10, Mo=0.20, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
        # AISI 4340: Ni-Cr-Mo, QT — high fatigue strength reference
        '4340_QT':        dict(NT=870, THT=845, THt=30, THQCr=12, CT=30, Ct=0, DT=30, Dt=0, QmT=30,
                                TT=540, Tt=60, TCr=20, C=0.40, Si=0.25, Mn=0.70, P=0.020, S=0.020,
                                Ni=1.80, Cr=0.80, Cu=0.10, Mo=0.25, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
        # AISI 8620 carburized: low-carbon, carburized to surface, tempered
        '8620_carb':      dict(NT=870, THT=30, THt=0, THQCr=0, CT=930, Ct=240, DT=850, Dt=60, QmT=60,
                                TT=170, Tt=60, TCr=20, C=0.20, Si=0.25, Mn=0.80, P=0.020, S=0.020,
                                Ni=0.55, Cr=0.55, Cu=0.10, Mo=0.20, RedRatio=600, dA=0.05, dB=0.0, dC=0.0),
    }
    # Approx published rotating-bending fatigue strength (10^7 cycles, polished), MPa
    # Source: Shigley / Roark / Bannantine handbook ranges
    handbook = {
        '1045_normalized': (260, 340),   # ~0.5 * UTS_normalized
        '4140_QT':         (450, 620),
        '4340_QT':         (520, 660),
        '8620_carb':       (700, 1000),  # carburized
    }

    for name, feats_d in grades.items():
        is_carb = (feats_d['CT'] == 930)
        feats_d['is_through_hardened'] = int(feats_d['THT'] > 30)
        if is_carb:
            X = pd.DataFrame([feats_d])[c_feats]
            pred = c_model.predict(X)[0]; sigma = meta['c']['cv_rmse']
            label = 'C'
        else:
            X = pd.DataFrame([feats_d])[nc_feats]
            pred = nc_model.predict(X)[0]; sigma = meta['nc']['cv_rmse']
            label = 'NC'
        lo, hi = handbook[name]
        in_range = lo <= pred <= hi
        nearest = 0 if in_range else min(abs(pred-lo), abs(pred-hi))
        flag = 'OK ' if in_range else ('LOW' if pred < lo else 'HIGH')
        print(f'  {name:18} [{label}]  pred={pred:6.1f} MPa  ±{sigma:.1f}  '
              f'handbook=[{lo},{hi}]  {flag}  off-by={nearest:.0f}')


# ---------- main ----------
if __name__ == '__main__':
    fn = sys.argv[1] if len(sys.argv) > 1 else 'all'
    funcs = {
        'cost': step_cost_bug,
        'moe':  step_moe_split,
        'models': step_models,
        'log':  step_log_transform,
        'hull': step_hull_sweep,
        'seed': step_seed_stability,
        'ext':  step_external,
    }
    if fn == 'all':
        for f in funcs.values(): f()
    else:
        funcs[fn]()
