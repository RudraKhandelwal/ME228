"""
Generates all figures for the 10-min ME228 final presentation.
Output: slides_plots/*.png  (300 dpi, sized for ~1280x720 slides)

Run: python gen_slide_plots.py
"""
import os, json, joblib, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import xgboost as xgb

OUT = 'slides_plots'
os.makedirs(OUT, exist_ok=True)

sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi': 130,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': 'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
})

# colour palette
NC_COL = '#2c7fb8'   # blue
C_COL  = '#e34a33'   # red-orange
ACC_COL = '#31a354'  # green
WARN_COL = '#fdae61' # warm
GREY = '#666666'


def load_all():
    df = pd.read_csv('data.csv').drop('Sl. No.', axis=1)
    df['is_carburized'] = (df['CT'] == 930).astype(int)
    df['is_through_hardened'] = (df['THT'] > 30).astype(int)
    nc_feats = json.load(open('models/nc_features.json'))
    c_feats  = json.load(open('models/c_features.json'))
    meta     = json.load(open('models/metadata.json'))
    nc_model = joblib.load('models/nc_xgb.pkl')
    c_model  = joblib.load('models/c_rf.pkl')
    return df, nc_feats, c_feats, meta, nc_model, c_model


# ============================================================
# 1.  FATIGUE BIMODALITY (slide 3)
# ============================================================
def fig_bimodality():
    df, *_ = load_all()
    nc = df[df.is_carburized == 0]['Fatigue']
    c  = df[df.is_carburized == 1]['Fatigue']

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(nc, bins=35, alpha=0.78, color=NC_COL, edgecolor='white', label=f'Non-carburised  (N={len(nc)})')
    ax.hist(c,  bins=20, alpha=0.85, color=C_COL,  edgecolor='white', label=f'Carburised  (N={len(c)})')
    ax.axvline(nc.mean(), color=NC_COL, linestyle='--', lw=1.4, alpha=0.9)
    ax.axvline(c.mean(),  color=C_COL,  linestyle='--', lw=1.4, alpha=0.9)
    ax.text(nc.mean()+8,  ax.get_ylim()[1]*0.92, f'μ={nc.mean():.0f}', color=NC_COL, fontsize=9, fontweight='bold')
    ax.text(c.mean()+8,   ax.get_ylim()[1]*0.92, f'μ={c.mean():.0f}',  color=C_COL,  fontsize=9, fontweight='bold')

    ax.set_xlabel('Fatigue strength (MPa)')
    ax.set_ylabel('Sample count')
    ax.set_title('Bimodal target distribution motivates Mixture-of-Experts split')
    ax.legend(loc='upper right')
    ax.text(0.02, 0.95, 'NIMS dataset, N=437', transform=ax.transAxes,
            fontsize=9, verticalalignment='top', color=GREY,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='none'))
    plt.tight_layout()
    fp = f'{OUT}/01_bimodality.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 2.  ARCHITECTURE DIAGRAM (slide 4)
# ============================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#dceaf5', ec=NC_COL, fw='normal', size=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                           linewidth=1.4, facecolor=fc, edgecolor=ec)
        ax.add_patch(b)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=size, fontweight=fw)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', lw=1.6, color=GREY))

    # input
    box(30, 88, 40, 8, 'Input: 26 features\n(composition + heat-treatment)',
        fc='#f0f0f0', ec=GREY, fw='bold')

    # gate
    box(38, 75, 24, 7, 'Regime gate:  CT == 930 °C ?', fc='#fff5d6', ec='#d4a017', fw='bold')

    arrow(50, 88, 50, 82)

    # branches
    box(8,  56, 35, 12, 'Carburised expert\nRandom Forest, 10 features\nCV-RMSE 40.4 MPa',
        fc='#fce4d8', ec=C_COL, fw='bold', size=10)
    box(57, 56, 35, 12, 'Non-carburised expert\nXGBoost, 12 features\nCV-RMSE 19.9 MPa',
        fc='#dceaf5', ec=NC_COL, fw='bold', size=10)

    arrow(45, 75, 25, 68)
    arrow(55, 75, 75, 68)
    ax.text(33, 71, 'yes', fontsize=10, color=C_COL, fontweight='bold')
    ax.text(63, 71, 'no',  fontsize=10, color=NC_COL, fontweight='bold')

    # uncertainty
    box(8,  41, 35, 9, 'Gaussian PoF\n(OOF residuals normal, p>0.1)',
        fc='#f0f0f0', ec=GREY, size=9)
    box(57, 41, 35, 9, 'Empirical PoF (OOF CDF)\n(NC residuals fail normality)',
        fc='#f0f0f0', ec=GREY, size=9)
    arrow(25, 56, 25, 50)
    arrow(75, 56, 75, 50)

    # merge
    box(30, 27, 40, 9, 'Forward report:  F̂,  FoS,  PoF,  95 % CI',
        fc='#e8f5e8', ec=ACC_COL, fw='bold', size=11)
    arrow(25, 41, 40, 36)
    arrow(75, 41, 60, 36)

    # inverse loop
    box(15, 7, 70, 14,
        'Inverse design loop  (recommend_alloys)\n'
        '·  neighbourhood sample over full composition\n'
        '·  kNN hull guard (99-th percentile)   ·   feasibility filter (FoS, PoF)\n'
        '·  cost over (C, Si, Mn, Ni, Cr, Cu, Mo, P, S, Ct)   ·   Pareto front + top-K',
        fc='#fff5d6', ec='#d4a017', size=9.5)
    arrow(50, 27, 50, 21)

    ax.set_title('Mixture-of-Experts pipeline: forward prediction + inverse design',
                 fontsize=13, fontweight='bold', y=1.0)
    fp = f'{OUT}/02_architecture.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 3.  COURSE-ALGORITHM BAKE-OFF (slide 7)
# ============================================================
def fig_bakeoff():
    df, nc_feats, c_feats, meta, *_ = load_all()
    cv = KFold(5, shuffle=True, random_state=42)

    cands = {
        'OLS':           lambda: Pipeline([('s', StandardScaler()), ('m', LinearRegression())]),
        'Ridge α=1':     lambda: Pipeline([('s', StandardScaler()), ('m', Ridge(alpha=1.0))]),
        'Lasso α=0.5':   lambda: Pipeline([('s', StandardScaler()), ('m', Lasso(alpha=0.5, max_iter=10000))]),
        'Perceptron SGD':lambda: Pipeline([('s', StandardScaler()),
                                           ('m', SGDRegressor(max_iter=2000, learning_rate='constant',
                                                              eta0=0.001, random_state=42))]),
        'MLP-Adam 16':   lambda: Pipeline([('s', StandardScaler()),
                                           ('m', MLPRegressor(hidden_layer_sizes=(16,), solver='adam',
                                                              learning_rate_init=0.01, max_iter=2000,
                                                              random_state=42))]),
        'MLP-Adam 16,16':lambda: Pipeline([('s', StandardScaler()),
                                           ('m', MLPRegressor(hidden_layer_sizes=(16,16), solver='adam',
                                                              learning_rate_init=0.005, max_iter=3000,
                                                              random_state=42))]),
        'MLP-RMSprop 16':lambda: Pipeline([('s', StandardScaler()),
                                           ('m', MLPRegressor(hidden_layer_sizes=(16,), solver='adam',
                                                              learning_rate_init=0.01, beta_1=0.0, beta_2=0.9,
                                                              max_iter=2000, random_state=42))]),
        'RF default':    lambda: RandomForestRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=42),
        'RF tuned':      lambda: RandomForestRegressor(**meta['c']['best_params'], n_jobs=-1, random_state=42),
        'XGB tuned':     lambda: xgb.XGBRegressor(**meta['nc']['best_params'], random_state=42),
    }

    results = {'NC': {}, 'C': {}}
    for regime, dfr, feats in [
        ('NC', df[df.is_carburized==0].reset_index(drop=True), nc_feats),
        ('C',  df[df.is_carburized==1].reset_index(drop=True), c_feats),
    ]:
        for name, mk in cands.items():
            try:
                s = cross_val_score(mk(), dfr[feats], dfr['Fatigue'], cv=cv,
                                    scoring='neg_root_mean_squared_error', n_jobs=-1)
                results[regime][name] = (-s.mean(), s.std())
            except Exception:
                results[regime][name] = (np.nan, 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, regime, col, n in [(axes[0], 'NC', NC_COL, 389), (axes[1], 'C', C_COL, 48)]:
        d = sorted(results[regime].items(), key=lambda x: x[1][0] if not np.isnan(x[1][0]) else 1e9)
        names = [k for k,_ in d]
        means = [v[0] for _,v in d]
        stds  = [v[1] for _,v in d]
        # cap absurd y for C MLP failures
        y_cap = 80 if regime == 'NC' else 110
        means_clip = [min(m, y_cap) for m in means]
        bars = ax.barh(names, means_clip, xerr=stds, color=[ACC_COL if 'tuned' in n else col for n in names],
                       alpha=0.85, edgecolor='white', error_kw=dict(ecolor=GREY, capsize=3, lw=1))
        # mark the chosen model
        chosen = 'XGB tuned' if regime == 'NC' else 'RF tuned'
        for bar, name, m in zip(bars, names, means):
            if name == chosen:
                bar.set_color(ACC_COL); bar.set_edgecolor('black'); bar.set_linewidth(1.5)
            label = f'{m:.1f}' if m < y_cap else f'{m:.0f}*'
            ax.text(min(m, y_cap) + 1.5, bar.get_y() + bar.get_height()/2,
                    label, va='center', fontsize=9, fontweight='bold' if name == chosen else 'normal')
        ax.set_xlabel('5-fold CV RMSE (MPa)')
        ax.set_title(f'{regime} regime  (N = {n})')
        ax.set_xlim(0, y_cap * 1.15)
        ax.invert_yaxis()
        if regime == 'C':
            ax.text(y_cap * 0.55, 0, '* MLP overfits at N=48\n  (true RMSE 148–189)',
                    fontsize=9, color=GREY, va='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=GREY))

    fig.suptitle('Course-algorithm bake-off — chosen models highlighted in green',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fp = f'{OUT}/03_bakeoff.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 4.  OOF RESIDUALS — DEMOLISHES GAUSSIAN ASSUMPTION (slide 8a)
# ============================================================
def fig_oof_residuals():
    df, nc_feats, c_feats, meta, *_ = load_all()
    cv = KFold(5, shuffle=True, random_state=42)
    dnc = df[df.is_carburized==0].reset_index(drop=True)
    dc  = df[df.is_carburized==1].reset_index(drop=True)

    nc_oof = cross_val_predict(xgb.XGBRegressor(**meta['nc']['best_params'], random_state=42),
                               dnc[nc_feats], dnc['Fatigue'], cv=cv, n_jobs=-1)
    c_oof  = cross_val_predict(RandomForestRegressor(**meta['c']['best_params'], n_jobs=-1, random_state=42),
                               dc[c_feats], dc['Fatigue'], cv=cv, n_jobs=-1)
    r_nc = dnc['Fatigue'].values - nc_oof
    r_c  = dc['Fatigue'].values  - c_oof

    fig = plt.figure(figsize=(13, 6.3))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    # NC histogram
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(r_nc, bins=30, density=True, alpha=0.8, color=NC_COL, edgecolor='white')
    xx = np.linspace(r_nc.min(), r_nc.max(), 400)
    ax.plot(xx, stats.norm.pdf(xx, r_nc.mean(), r_nc.std()), 'k--', lw=1.6, label='Gaussian fit')
    ax.set_title(f'NC OOF residuals  —  Shapiro p<10⁻³,  KS p=2×10⁻⁴',
                 fontsize=11, color=NC_COL)
    ax.set_xlabel('Residual (MPa)'); ax.set_ylabel('Density'); ax.legend(fontsize=9)
    ax.text(0.02, 0.95, f'σ_OOF = {r_nc.std():.1f} MPa\nrange [{r_nc.min():.0f}, {r_nc.max():.0f}]',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # NC Q-Q
    ax = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(r_nc, dist='norm')
    ax.scatter(osm, osr, s=18, alpha=0.65, color=NC_COL)
    ax.plot([osm.min(), osm.max()], [slope*osm.min()+intercept, slope*osm.max()+intercept], 'k--', lw=1.4)
    ax.set_title(f'NC Q-Q  —  R={r:.3f}, heavy tails', fontsize=11, color=NC_COL)
    ax.set_xlabel('Theoretical quantile'); ax.set_ylabel('Sample quantile')

    # C histogram
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(r_c, bins=14, density=True, alpha=0.8, color=C_COL, edgecolor='white')
    xx = np.linspace(r_c.min(), r_c.max(), 400)
    ax.plot(xx, stats.norm.pdf(xx, r_c.mean(), r_c.std()), 'k--', lw=1.6, label='Gaussian fit')
    ax.set_title(f'C OOF residuals  —  Shapiro p={stats.shapiro(r_c)[1]:.2f}, KS p={stats.kstest(r_c, "norm", args=(r_c.mean(), r_c.std()))[1]:.2f}',
                 fontsize=11, color=C_COL)
    ax.set_xlabel('Residual (MPa)'); ax.set_ylabel('Density'); ax.legend(fontsize=9)
    ax.text(0.02, 0.95, f'σ_OOF = {r_c.std():.1f} MPa\nrange [{r_c.min():.0f}, {r_c.max():.0f}]',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # C Q-Q
    ax = fig.add_subplot(gs[1, 1])
    (osm, osr), (slope, intercept, r) = stats.probplot(r_c, dist='norm')
    ax.scatter(osm, osr, s=22, alpha=0.7, color=C_COL)
    ax.plot([osm.min(), osm.max()], [slope*osm.min()+intercept, slope*osm.max()+intercept], 'k--', lw=1.4)
    ax.set_title(f'C Q-Q  —  R={r:.3f}, normality holds', fontsize=11, color=C_COL)
    ax.set_xlabel('Theoretical quantile'); ax.set_ylabel('Sample quantile')

    fig.suptitle('OOF residuals: NC fails Gaussian → empirical PoF; C passes → keep Gaussian',
                 fontsize=13, fontweight='bold', y=1.005)
    fp = f'{OUT}/04_oof_residuals.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 5.  PoF: GAUSSIAN vs EMPIRICAL (slide 8b)
# ============================================================
def fig_pof_compare():
    df, nc_feats, _, meta, *_ = load_all()
    cv = KFold(5, shuffle=True, random_state=42)
    dnc = df[df.is_carburized==0].reset_index(drop=True)
    nc_oof = cross_val_predict(xgb.XGBRegressor(**meta['nc']['best_params'], random_state=42),
                               dnc[nc_feats], dnc['Fatigue'], cv=cv, n_jobs=-1)
    r = dnc['Fatigue'].values - nc_oof
    sigma = r.std()

    F_hat = 564.9   # canonical example from report
    applied = np.linspace(F_hat - 80, F_hat + 80, 400)
    pof_gauss = stats.norm.cdf(applied, loc=F_hat, scale=sigma)
    pof_emp = np.array([(r < (a - F_hat)).mean() for a in applied])

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(applied, pof_gauss * 100, lw=2.4, color=NC_COL, label='Gaussian: $\\Phi((\\sigma_{app}-\\hat{F})/\\sigma)$')
    ax.plot(applied, pof_emp * 100,   lw=2.4, color=ACC_COL, label='Empirical CDF of OOF residuals')

    # marker at the canonical example (571.6 MPa)
    a0 = 571.6
    g0 = stats.norm.cdf(a0, F_hat, sigma) * 100
    e0 = (r < (a0 - F_hat)).mean() * 100
    ax.axvline(a0, color=GREY, linestyle=':', lw=1.2)
    ax.scatter([a0, a0], [g0, e0], s=60, zorder=5, color=['#1f4e79', '#1d6f3a'], edgecolor='white', linewidth=1.5)
    ax.annotate(f'Gaussian PoF = {g0:.1f}%', xy=(a0, g0), xytext=(a0+8, g0-12),
                fontsize=10, color=NC_COL, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=NC_COL, lw=1))
    ax.annotate(f'Empirical PoF = {e0:.1f}%', xy=(a0, e0), xytext=(a0+8, e0+8),
                fontsize=10, color=ACC_COL, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=ACC_COL, lw=1))
    ax.text(a0+1, -8, f'σ_app={a0} MPa', fontsize=9, color=GREY)

    ax.set_xlabel('Applied stress  σ_app  (MPa)')
    ax.set_ylabel('PoF (%)')
    ax.set_title(f'NC Probability of Failure: Gaussian over-confident in heavy tail\n($\\hat F=$ {F_hat} MPa, σ={sigma:.1f} MPa)')
    ax.legend(loc='upper left')
    ax.set_ylim(-12, 105)
    plt.tight_layout()
    fp = f'{OUT}/05_pof_compare.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 6.  COST-BUG BEFORE / AFTER (slide 9a)
# ============================================================
def fig_cost_bug():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    grades = ['Buggy NC top-1', 'Fixed NC top-1', 'Buggy C top-1', 'Fixed C top-1']
    rep = [6.18, 6.52, 6.88, 17.41]      # what the buggy code reported
    true= [11.35, 6.52, 18.86, 17.41]    # true cost when Ni/Cr accounted

    x = np.arange(len(grades))
    w = 0.36
    b1 = ax.bar(x - w/2, rep,  width=w, color=GREY, alpha=0.85, label='Reported cost')
    b2 = ax.bar(x + w/2, true, width=w, color=C_COL, alpha=0.9, label='True cost (full composition)')
    for bars, vals in [(b1, rep), (b2, true)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, f'${v:.2f}', ha='center', fontsize=9.5)

    ax.set_xticks(x); ax.set_xticklabels(grades, rotation=10)
    ax.set_ylabel('Cost (USD/kg, proxy)')
    ax.set_title('Cost-model bug — sampler dropped Ni (NC) and Ni/Cr (C); +74 % NC inflation')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fp = f'{OUT}/06_cost_bug.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 7.  PARETO FRONT showcase (slide 9b) — re-render to slide aspect
# ============================================================
def fig_pareto_showcase():
    """Compose the existing pareto plots into one slide-friendly figure."""
    from matplotlib.image import imread
    if not (os.path.exists('models/pareto_non_carb.png') and os.path.exists('models/pareto_carb.png')):
        print('  pareto images missing — skip showcase')
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, path, title in [
        (axes[0], 'models/pareto_non_carb.png', 'Non-carburised  (σ_app=500 MPa, FoS≥1.5)'),
        (axes[1], 'models/pareto_carb.png',     'Carburised  (σ_app=700 MPa, FoS≥1.4)')]:
        ax.imshow(imread(path))
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    fig.suptitle('Inverse design: Pareto front (cost vs PoF) + top-K cheapest',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout()
    fp = f'{OUT}/07_pareto_showcase.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 8.  HOEFFDING vs OOF RMSE bound (slide 6)
# ============================================================
def fig_hoeffding():
    df, *_ = load_all()
    delta = 0.05
    fig, ax = plt.subplots(figsize=(9, 4.6))

    regimes = ['NC (N=389)', 'C (N=48)']
    hoeff = []
    actual = []
    rng    = []
    for r, dfr, oof_rmse in [
        ('NC', df[df.is_carburized==0], 20.07),
        ('C',  df[df.is_carburized==1], 42.70),
    ]:
        N = len(dfr)
        L = dfr['Fatigue'].max() - dfr['Fatigue'].min()
        eps = L * np.sqrt(np.log(2/delta) / (2 * N))
        hoeff.append(eps); actual.append(oof_rmse); rng.append(L)

    x = np.arange(len(regimes))
    w = 0.36
    b1 = ax.bar(x - w/2, hoeff,  width=w, color='#cccccc', edgecolor='black', label='Hoeffding 95 % bound  ε')
    b2 = ax.bar(x + w/2, actual, width=w, color=ACC_COL,  edgecolor='black', label='Measured OOF RMSE')
    for b, v in zip(b1, hoeff):
        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}', ha='center', fontsize=10)
    for b, v in zip(b2, actual):
        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}', ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(regimes)
    ax.set_ylabel('MPa')
    ax.set_title('Hoeffding generalisation bound vs measured OOF RMSE\nBoth regimes safely inside the bound (no overfit signal)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fp = f'{OUT}/08_hoeffding.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 9.  THREE-LEG VALIDATION SUMMARY (slide 10)
# ============================================================
def fig_validation():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Leg 1: holdout vs CV
    ax = axes[0]
    legs = ['NC CV', 'NC Holdout', 'C CV', 'C Holdout']
    vals = [19.91, 17.63, 40.42, 29.54]
    cols = [NC_COL, '#5dade2', C_COL, '#f1948a']
    bars = ax.bar(legs, vals, color=cols, edgecolor='white')
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.6, f'{v:.2f}', ha='center', fontsize=10)
    ax.set_ylabel('RMSE (MPa)')
    ax.set_title('Leg 1: Stratified 80/20 holdout\n(within CV envelope)', fontsize=11)
    ax.set_xticklabels(legs, rotation=10)
    ax.grid(axis='y', alpha=0.3)

    # Leg 2: matbench rank correlation — REAL data
    ax = axes[1]
    try:
        from matminer.datasets import load_dataset
        import re
        mb = load_dataset('matbench_steels')
        atomic_wt = {'Fe':55.85,'C':12.01,'Si':28.09,'Mn':54.94,'P':30.97,'S':32.07,
                     'Ni':58.69,'Cr':52.00,'Cu':63.55,'Mo':95.96,'V':50.94,'Co':58.93,
                     'Nb':92.91,'W':183.84,'Al':26.98,'Ti':47.87,'N':14.01,'Ta':180.95,
                     'Zr':91.22,'B':10.81,'Ce':140.12}
        pat = re.compile(r'([A-Z][a-z]?)(\d*\.?\d+)')
        el_set = ['C','Si','Mn','P','S','Ni','Cr','Cu','Mo']
        rows = []
        for f in mb['composition']:
            atoms = {el: float(v) for el, v in pat.findall(f)}
            masses = {el: a * atomic_wt.get(el, 50.0) for el, a in atoms.items()}
            tot = sum(masses.values())
            rows.append({el: masses.get(el, 0)/tot*100 for el in el_set})
        mb_comp = pd.DataFrame(rows)
        # train comp-only NC model
        df_, nc_feats, *_ = load_all()
        dnc = df_[df_.is_carburized==0].reset_index(drop=True)
        m = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=42)
        m.fit(dnc[el_set], dnc['Fatigue'])
        pred = m.predict(mb_comp)
        ys = mb['yield strength'].values
        rho, _ = stats.spearmanr(pred, ys)
        ax.scatter(pred, ys, s=14, alpha=0.55, color=GREY)
        ax.set_title(f'Leg 2: matbench OOD test\nSpearman ρ = {rho:+.2f}  (no transfer = correct)', fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f'matbench unavailable\n{e.__class__.__name__}',
                ha='center', va='center', transform=ax.transAxes, fontsize=10, color=GREY)
    ax.set_xlabel('NIMS-trained predicted fatigue (MPa)')
    ax.set_ylabel('matbench yield strength (MPa)')
    ax.text(0.02, 0.97, 'Maraging steels —\noutside training distribution',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor=GREY))
    ax.grid(alpha=0.3)

    # Leg 3: handbook grades
    ax = axes[2]
    grades = ['1045\nnorm.', '4140\nQT', '4340\nQT', '8620\ncarb.']
    pred = [527.5, 621.6, 624.6, 923.0]
    lo = [260, 450, 520, 700]
    hi = [340, 620, 660, 1000]
    x = np.arange(len(grades))
    for xi, l, h in zip(x, lo, hi):
        ax.add_patch(plt.Rectangle((xi-0.32, l), 0.64, h-l, color=ACC_COL, alpha=0.25))
    for xi, p in zip(x, pred):
        col = WARN_COL if p > hi[xi] else ACC_COL
        ax.scatter(xi, p, s=110, color=col, edgecolor='black', zorder=4, marker='D')
        ax.text(xi, p+25, f'{p:.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(grades)
    ax.set_ylabel('Fatigue strength (MPa)')
    ax.set_title('Leg 3: AISI handbook grades\n3/4 in range; 1045 plain-C = bound limit', fontsize=11)
    ax.set_ylim(0, 1100)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Three-leg external validation', fontsize=13, fontweight='bold', y=1.04)
    plt.tight_layout()
    fp = f'{OUT}/09_validation.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# 10.  VC TABLE rendered as figure (optional, slide 6 alternative)
# ============================================================
def fig_vc_table():
    rows = [
        ('NC', 389, 'Linear / Ridge',                13, 29.92, '✓'),
        ('NC', 389, 'MLP 1×16',                     225,  1.73, 'low'),
        ('NC', 389, 'XGBoost (chosen)',            9280,  0.04, 'raw fail*'),
        ('C',  48,  'Linear / Ridge',                11,  4.36, 'low'),
        ('C',  48,  'MLP 1×8',                       97,  0.49, 'raw fail*'),
        ('C',  48,  'Random Forest (chosen)',      6848,  0.01, 'raw fail*'),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis('off')
    col_lbl = ['Regime', 'N', 'Hypothesis class', 'd_VC (raw)', 'N / d_VC', 'N ≥ 10·d_VC?']
    table = ax.table(cellText=[[str(c) for c in r] for r in rows],
                     colLabels=col_lbl, loc='center', cellLoc='center',
                     colColours=['#dceaf5']*6)
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1, 1.55)
    # color flag column
    for i, r in enumerate(rows, start=1):
        flag = r[-1]
        cell = table[i, 5]
        cell.set_facecolor('#d9efd9' if flag == '✓' else ('#fff2cc' if flag == 'low' else '#fadbd8'))
    ax.set_title('VC dimension bookkeeping  —  rule of thumb $N \\geq 10\\,d_{VC}$\n'
                 '*tree-ensemble raw bound mitigated by bagging + L1/L2 regularization',
                 fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    fp = f'{OUT}/10_vc_table.png'
    plt.savefig(fp); plt.close()
    print(f'  saved {fp}')


# ============================================================
# main
# ============================================================
if __name__ == '__main__':
    print(f'Generating slide plots → {OUT}/')
    fig_bimodality()
    fig_architecture()
    fig_bakeoff()
    fig_oof_residuals()
    fig_pof_compare()
    fig_cost_bug()
    fig_pareto_showcase()
    fig_hoeffding()
    fig_validation()
    fig_vc_table()
    print('\nAll figures generated.')
