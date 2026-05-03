"""
ME228 — Fatigue Strength Predictor & Alloy Recommender
Streamlit GUI — Phase 4
"""
import os, sys, json, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.matplotlib'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st
import joblib
import shap
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ME228 Fatigue Predictor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_everything():
    nc_model  = joblib.load('models/nc_xgb.pkl')
    c_model   = joblib.load('models/c_rf.pkl')
    nc_feats  = json.load(open('models/nc_features.json'))
    c_feats   = json.load(open('models/c_features.json'))
    meta      = json.load(open('models/metadata.json'))

    df_raw = pd.read_csv('data.csv')
    df_sub = df_raw.copy()
    if 'Sl. No.' in df_sub.columns:
        df_sub = df_sub.drop('Sl. No.', axis=1)
    df_sub['is_carburized']       = (df_sub['CT'] == 930).astype(int)
    df_sub['is_through_hardened'] = (df_sub['THT'] > 30).astype(int)
    df_nc = df_sub[df_sub['is_carburized'] == 0].reset_index(drop=True)
    df_c  = df_sub[df_sub['is_carburized'] == 1].reset_index(drop=True)

    # kNN guards
    def build_guard(X_tr, pct=99.0):
        sc = StandardScaler().fit(X_tr)
        Xs = sc.transform(X_tr)
        knn = NearestNeighbors(n_neighbors=5).fit(Xs)
        d, _ = knn.kneighbors(Xs)
        return knn, sc, np.percentile(d.mean(axis=1), pct)

    knn_nc, sc_nc, thr_nc = build_guard(df_nc[nc_feats])
    knn_c,  sc_c,  thr_c  = build_guard(df_c[c_feats])

    return (nc_model, c_model, nc_feats, c_feats, meta,
            df_nc, df_c, knn_nc, sc_nc, thr_nc, knn_c, sc_c, thr_c)


(nc_model, c_model, nc_feats, c_feats, meta,
 df_nc, df_c, knn_nc, sc_nc, thr_nc, knn_c, sc_c, thr_c) = load_everything()

# ── Cost constants ────────────────────────────────────────────────────────────
ELEM_COST = {'C': 0.50, 'Si': 1.20, 'Mn': 1.80,
             'Ni': 14.0, 'Cr': 10.0, 'Cu': 9.0, 'Mo': 30.0,
             'P': 0.0, 'S': 0.0}
CARB_PROC_COST = 0.05

def alloy_cost(row):
    c = sum(ELEM_COST.get(e, 0) * row.get(e, 0) for e in ELEM_COST)
    c += CARB_PROC_COST * row.get('Ct', 0)
    return c

# ── Helpers ───────────────────────────────────────────────────────────────────
def in_hull(X_cand, knn, sc, thr):
    d, _ = knn.kneighbors(sc.transform(X_cand))
    return d.mean(axis=1) <= thr

def neighborhood_sample(df_tr, feats, n=25000, noise=0.20, seed=42):
    rng  = np.random.default_rng(seed)
    idx  = rng.integers(0, len(df_tr), size=n)
    base = df_tr[feats].values[idx].astype(float)
    std  = df_tr[feats].std().values
    cands = np.clip(base + rng.normal(0, noise, base.shape) * std,
                    df_tr[feats].min().values, df_tr[feats].max().values)
    return pd.DataFrame(cands, columns=feats)

def pareto(df, c1='cost_score', c2='pof_pct'):
    pts = df[[c1, c2]].values
    dom = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j and (pts[j,0] <= pts[i,0] and pts[j,1] <= pts[i,1] and
                           (pts[j,0] < pts[i,0] or pts[j,1] < pts[i,1])):
                dom[i] = True; break
    return df[~dom].sort_values(c1).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/b/b8/IIT_Bombay_Logo.svg/200px-IIT_Bombay_Logo.svg.png",
                 width=80)
st.sidebar.title("ME228 — Fatigue Predictor")
st.sidebar.markdown("**NIMS Steel Fatigue Dataset**  \nAgrawal et al. (2014)")
st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "Navigate",
    ["Forward Prediction", "Inverse Design (Alloy Recommender)"],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    f"NC XGBoost CV RMSE: **{meta['nc']['cv_rmse']:.1f} MPa**  \n"
    f"C RF CV RMSE: **{meta['c']['cv_rmse']:.1f} MPa**"
)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORWARD PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
if tab_choice == "Forward Prediction":
    st.title("⚙️ Forward Prediction — Fatigue Strength & Reliability")
    st.markdown(
        "Enter alloy composition and heat-treatment parameters. "
        "The model routes the input to the correct regime-specific expert and "
        "returns predicted fatigue strength, 95% prediction interval, and "
        "probability of failure."
    )

    regime = st.radio("Heat Treatment Regime", ["Non-Carburized", "Carburized"],
                      horizontal=True)
    is_carb = (regime == "Carburized")

    st.markdown("### 🔩 Composition (wt %)")
    c1, c2, c3, c4 = st.columns(4)
    C_  = c1.slider("C",   0.10, 0.70, 0.40, 0.01)
    Si_ = c2.slider("Si",  0.10, 2.10, 0.25, 0.01)
    Mn_ = c3.slider("Mn",  0.30, 1.70, 0.70, 0.01)
    Cr_ = c4.slider("Cr",  0.00, 1.20, 0.70, 0.01)

    c5, c6, c7, c8 = st.columns(4)
    Mo_ = c5.slider("Mo",  0.00, 0.25, 0.00, 0.01)
    Ni_ = c6.slider("Ni",  0.00, 3.50, 0.05, 0.01)
    Cu_ = c7.slider("Cu",  0.00, 0.30, 0.05, 0.01)
    P_  = c8.slider("P",   0.002, 0.032, 0.015, 0.001, format="%.3f")

    c9, c10 = st.columns([1, 3])
    S_  = c9.slider("S",   0.002, 0.032, 0.015, 0.001, format="%.3f")

    st.markdown("### 🔥 Heat Treatment")
    h1, h2, h3, h4 = st.columns(4)
    NT_  = h1.slider("NT (°C)",  800, 980, 880, 5)
    THT_ = h2.slider("THT (°C)", 0, 870, 0 if not is_carb else 30, 5)
    THt_ = h3.slider("THt (min)",0, 120, 0, 5)
    TT_  = h4.slider("TT (°C)",  30, 700, 600 if not is_carb else 160, 5)

    h5, h6, h7, h8 = st.columns(4)
    Tt_     = h5.slider("Tt (min)", 0, 180, 60, 5)
    DT_     = h6.slider("DT (°C)", 30, 900, 30, 5)
    Dt_     = h7.slider("Dt (min)",0, 120, 0, 5)
    QmT_    = h8.slider("QmT (°C)",30, 200, 30 if not is_carb else 60, 5)

    h9, h10, h11, h12 = st.columns(4)
    CT_     = h9.slider("CT (°C)", 30, 930, 30 if not is_carb else 930, step=900 if not is_carb else 900)
    Ct_     = h10.slider("Ct (min)", 0, 600, 0 if not is_carb else 120, 10)
    TCr_    = h11.slider("TCr", 0, 10, 0, 1)
    THQCr_  = h12.slider("THQCr", 0, 10, 0, 1)

    st.markdown("### 📐 Specimen / Process")
    s1, s2, s3, s4 = st.columns(4)
    RedRatio_ = s1.slider("RedRatio", 200, 5600, 600, 50)
    dA_       = s2.slider("dA",  0.00, 0.15, 0.04, 0.005, format="%.3f")
    dB_       = s3.slider("dB",  0.00, 0.10, 0.00, 0.005, format="%.3f")
    dC_       = s4.slider("dC",  0.00, 0.10, 0.00, 0.005, format="%.3f")

    applied_stress = st.number_input(
        "Applied Operational Stress (MPa)", min_value=50.0, max_value=2000.0,
        value=500.0, step=10.0
    )

    feats_row = {
        'NT': NT_, 'THT': THT_, 'THt': THt_, 'THQCr': THQCr_,
        'CT': CT_, 'Ct': Ct_, 'DT': DT_, 'Dt': Dt_, 'QmT': QmT_,
        'TT': TT_, 'Tt': Tt_, 'TCr': TCr_,
        'C': C_, 'Si': Si_, 'Mn': Mn_, 'P': P_, 'S': S_,
        'Ni': Ni_, 'Cr': Cr_, 'Cu': Cu_, 'Mo': Mo_,
        'RedRatio': RedRatio_, 'dA': dA_, 'dB': dB_, 'dC': dC_,
        'is_through_hardened': int(THT_ > 30),
    }

    if st.button("🔮 Predict Fatigue Strength", type="primary"):
        if is_carb:
            model, feats, rmse, label = c_model, c_feats, meta['c']['cv_rmse'], "Carburized (RF)"
        else:
            model, feats, rmse, label = nc_model, nc_feats, meta['nc']['cv_rmse'], "Non-Carburized (XGBoost)"

        X_in   = pd.DataFrame([feats_row])[feats]
        pred   = float(model.predict(X_in)[0])
        margin = 1.96 * rmse
        pof    = norm.cdf(applied_stress, loc=pred, scale=rmse) * 100
        fos    = pred / applied_stress

        # ── Result cards ──────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📊 Reliability Report")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted Fatigue", f"{pred:.1f} MPa")
        m2.metric("Factor of Safety",  f"{fos:.3f}",
                  delta="Safe" if fos >= 1.2 else "⚠ Low",
                  delta_color="normal" if fos >= 1.2 else "inverse")
        m3.metric("Prob. of Failure",  f"{pof:.4f}%",
                  delta="Critical" if pof > 5 else "OK",
                  delta_color="inverse" if pof > 5 else "normal")
        m4.metric("Regime", label)

        # ── Prediction interval plot ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Gaussian PoF curve
        ax = axes[0]
        x  = np.linspace(pred - 4*rmse, pred + 4*rmse, 400)
        ax.plot(x, norm.pdf(x, pred, rmse), color='steelblue', lw=2)
        ax.axvline(applied_stress, color='crimson', linestyle='--', lw=1.5,
                   label=f'Applied stress ({applied_stress:.0f} MPa)')
        ax.axvline(pred, color='steelblue', linestyle=':', lw=1.5,
                   label=f'Predicted ({pred:.1f} MPa)')
        ax.fill_between(x, norm.pdf(x, pred, rmse),
                        where=(x <= applied_stress), alpha=0.25, color='crimson',
                        label=f'PoF area ({pof:.4f}%)')
        lo, hi = pred - margin, pred + margin
        ax.axvspan(lo, hi, alpha=0.08, color='steelblue', label='95% PI')
        ax.set_xlabel('Fatigue Strength (MPa)')
        ax.set_ylabel('Probability Density')
        ax.set_title('Predicted Fatigue Distribution vs Applied Stress')
        ax.legend(fontsize=8)

        # SHAP waterfall
        ax2 = axes[1]
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_in)
        sv_arr = sv[0] if isinstance(sv, list) else sv.flatten()
        feat_names = list(X_in.columns)
        base_val = float(explainer.expected_value
                         if not isinstance(explainer.expected_value, (list, np.ndarray))
                         else explainer.expected_value[0])

        idx_sorted = np.argsort(np.abs(sv_arr))[-10:]
        vals  = sv_arr[idx_sorted]
        names = [feat_names[i] for i in idx_sorted]
        colors = ['#d73027' if v > 0 else '#4575b4' for v in vals]
        ax2.barh(names, vals, color=colors)
        ax2.axvline(0, color='black', lw=0.8)
        ax2.set_xlabel('SHAP Value (impact on prediction)')
        ax2.set_title(f'Top-10 Feature Contributions\n(base = {base_val:.1f} MPa)')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — INVERSE DESIGN
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.title("🔬 Inverse Design — Alloy Recommender")
    st.markdown(
        "Specify your operating conditions and desired reliability target. "
        "The engine samples alloy candidates within the training data hull, "
        "filters by Factor of Safety and Probability of Failure, then ranks "
        "by alloy cost."
    )

    col_a, col_b, col_c, col_d = st.columns(4)
    inv_regime  = col_a.radio("Regime", ["Non-Carburized", "Carburized"])
    inv_stress  = col_b.number_input("Applied Stress (MPa)", 100.0, 1500.0, 500.0, 10.0)
    inv_fos     = col_c.slider("Target FoS ≥", 1.1, 2.5, 1.5, 0.05)
    inv_maxpof  = col_d.slider("Max PoF (%)", 0.1, 20.0, 2.0, 0.5)

    col_e, col_f = st.columns(2)
    inv_topk    = col_e.slider("Show Top-K alloys", 5, 20, 10)
    inv_n       = col_f.select_slider("Sampling size",
                                      options=[10000, 25000, 50000, 100000], value=25000)

    if st.button("🚀 Find Optimal Alloys", type="primary"):
        regime_key = 'non_carb' if inv_regime == "Non-Carburized" else 'carb'
        model  = nc_model  if regime_key == 'non_carb' else c_model
        feats  = nc_feats  if regime_key == 'non_carb' else c_feats
        rmse   = meta['nc']['cv_rmse'] if regime_key == 'non_carb' else meta['c']['cv_rmse']
        df_tr  = df_nc     if regime_key == 'non_carb' else df_c
        knn_, sc_, thr_ = (knn_nc, sc_nc, thr_nc) if regime_key == 'non_carb' else (knn_c, sc_c, thr_c)

        with st.spinner("Sampling and evaluating candidates…"):
            cands = neighborhood_sample(df_tr, feats, n=inv_n)
            mask  = in_hull(cands, knn_, sc_, thr_)
            cands = cands[mask].reset_index(drop=True)
            n_hull = len(cands)

            cands['pred_fatigue'] = model.predict(cands[feats])
            cands['fos']          = cands['pred_fatigue'] / inv_stress
            cands['pof_pct']      = norm.cdf(inv_stress,
                                             loc=cands['pred_fatigue'],
                                             scale=rmse) * 100
            feasible = cands[
                (cands['pred_fatigue'] >= inv_fos * inv_stress) &
                (cands['pof_pct']      <= inv_maxpof)
            ].copy()
            feasible['cost_score'] = feasible.apply(alloy_cost, axis=1)

        st.info(
            f"{n_hull:,} candidates inside data hull  |  "
            f"**{len(feasible):,}** satisfy FoS ≥ {inv_fos} & PoF ≤ {inv_maxpof}%"
        )

        if len(feasible) == 0:
            st.error("No feasible alloys found. Relax FoS or increase max PoF.")
            st.stop()

        top = feasible.nsmallest(inv_topk, 'cost_score').reset_index(drop=True)
        top.index += 1
        pareto_df = pareto(feasible)

        # ── Pareto + bar chart ────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 5))
        gs  = gridspec.GridSpec(1, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0])
        ax1.scatter(feasible['cost_score'], feasible['pof_pct'],
                    s=8, alpha=0.25, color='steelblue', label='Feasible')
        ax1.scatter(pareto_df['cost_score'], pareto_df['pof_pct'],
                    s=50, color='darkorange', zorder=5, label='Pareto front')
        ax1.scatter(top['cost_score'], top['pof_pct'],
                    s=90, marker='*', color='crimson', zorder=6,
                    label=f'Top-{inv_topk}')
        ax1.set_xlabel('Cost Score (USD/kg, approx)')
        ax1.set_ylabel('Probability of Failure (%)')
        ax1.set_title('Cost vs PoF — Pareto Front')
        ax1.legend(fontsize=9)

        ax2 = fig.add_subplot(gs[1])
        ranks = top.index.tolist()
        bars  = ax2.barh(ranks, top['pred_fatigue'], color='steelblue', alpha=0.8)
        ax2.axvline(inv_stress, color='crimson', linestyle='--', lw=1.5,
                    label=f'Applied stress ({inv_stress:.0f} MPa)')
        ax2.axvline(inv_fos * inv_stress, color='darkorange', linestyle='--', lw=1.5,
                    label=f'Target ({inv_fos}×σ)')
        for bar, cost in zip(bars, top['cost_score']):
            ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                     f'${cost:.2f}', va='center', fontsize=8)
        ax2.set_yticks(ranks)
        ax2.set_yticklabels([f'#{r}' for r in ranks])
        ax2.set_xlabel('Predicted Fatigue (MPa)')
        ax2.set_title('Top-K Alloys by Rank')
        ax2.legend(fontsize=9)
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Results table ─────────────────────────────────────────────────────
        st.markdown("### 📋 Top Alloy Configurations")
        disp_cols = feats + ['pred_fatigue', 'fos', 'pof_pct', 'cost_score']
        fmt = {'pred_fatigue': '{:.1f}', 'fos': '{:.3f}',
               'pof_pct': '{:.4f}', 'cost_score': '{:.3f}'}
        fmt.update({f: '{:.4f}' for f in feats})
        display_df = top[disp_cols].copy()
        for col, fmt_str in fmt.items():
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda v: fmt_str.format(v) if pd.notna(v) else v)
        st.dataframe(display_df, use_container_width=True)

        # ── SHAP waterfall for selected rank ──────────────────────────────────
        st.markdown("### 🔍 SHAP Feature Explanation")
        sel_rank = st.selectbox("Select alloy rank for SHAP explanation",
                                options=top.index.tolist(), index=0)
        sel_row  = top.loc[sel_rank, feats]
        X_sel    = pd.DataFrame([sel_row])

        explainer = shap.TreeExplainer(model)
        sv        = explainer.shap_values(X_sel)
        sv_arr    = sv[0] if isinstance(sv, list) else sv.flatten()
        base_val  = float(explainer.expected_value
                          if not isinstance(explainer.expected_value, (list, np.ndarray))
                          else explainer.expected_value[0])

        feat_names = list(X_sel.columns)
        idx_s = np.argsort(np.abs(sv_arr))
        vals  = sv_arr[idx_s]
        names = [feat_names[i] for i in idx_s]
        colors = ['#d73027' if v > 0 else '#4575b4' for v in vals]

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.barh(names, vals, color=colors)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel('SHAP Value (MPa impact)')
        ax.set_title(
            f'Rank #{sel_rank} — SHAP Contributions\n'
            f'Base={base_val:.1f}  Pred={top.loc[sel_rank, "pred_fatigue"]:.1f} MPa  '
            f'FoS={top.loc[sel_rank, "fos"]:.3f}  Cost=${top.loc[sel_rank, "cost_score"]:.2f}'
        )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── Cost breakdown for rank 1 ─────────────────────────────────────────
        st.markdown("### 💰 Cost Breakdown — Rank #1")
        best = top.iloc[0]
        breakdown = []
        for elem, price in sorted(ELEM_COST.items(), key=lambda x: -x[1]):
            if price > 0 and elem in best:
                breakdown.append({
                    'Element': elem,
                    'wt%': round(float(best[elem]), 4),
                    'Price (USD/kg)': price,
                    'Contribution (USD/kg)': round(price * float(best[elem]), 4),
                })
        if 'Ct' in best:
            breakdown.append({
                'Element': 'Carb. time (Ct)',
                'wt%': round(float(best['Ct']), 1),
                'Price (USD/kg)': CARB_PROC_COST,
                'Contribution (USD/kg)': round(CARB_PROC_COST * float(best['Ct']), 4),
            })
        st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)
