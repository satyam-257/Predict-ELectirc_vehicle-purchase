"""
ensemble_frontier_v9.py — Frontier Synergy V9 Architecture
Integrating Option C Models:
1. goodpjw2008's GLM-Margin GBDT Stack (Public LB = 0.94675, OOF = 0.946638)
2. Our V7 Apex Frontier H-Blend Leader (Public LB = 0.94668)
3. Our 9-Model 10-Fold Leak-Free Super-Stacker (CV = 0.946606)
4. Our New Local 25-Key Triple-TE Residual LightGBM (Option C, OOF = 0.946364)
5. Our New Local 25-Key Triple-TE Residual XGBoost on CUDA (Option C, OOF = 0.946367)
6. RealMLP Tabular Deep Neural Network (KS-Divergence = 0.1452)
7. Continuous Heuljax DGP Dither for Zero-Tie Lexicographical Ordering
8. 4 Pure Ground-Truth Generator Laws (100.000% pure physics)
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr
import os

print("=" * 80)
print(" S6E9: FRONTIER SYNERGY V9 (INTEGRATING LOCAL OPTION C RESIDUAL MODELS)")
print("=" * 80)

# 1. Load Test Data
test = pd.read_csv('data/test.csv')
test_ids = test['id'].values
n_test = len(test_ids)
print(f"Loaded test set: {n_test:,} rows")

# 2. Extract Ground-Truth Generator Laws
income = test['Annual_Income_USD'].values
commute = test['Daily_Commute_km'].values
no_subsidy = (test['Subsidy_Available'] == 'No').values
c1_mask = (test['Environmental_Concern_Level'] == 1).values
anx = test['Range_Anxiety_Level'].isin(['Medium', 'High']).values

r_always = income >= 170537
r_never_inc = (income >= 31004) & (income <= 41970)
r_never_comm = commute >= 83
r_never_30k = (income == 30000) & no_subsidy & (c1_mask | anx)
r_never_all = r_never_inc | r_never_comm | r_never_30k

def apply_laws_and_zerotie(raw_score, continuous_dither):
    dither = (continuous_dither - np.mean(continuous_dither)) / (np.std(continuous_dither) + 1e-12) * 1e-9
    r = rankdata(raw_score + dither) / len(raw_score)
    r[r_always] += 2.0
    r[r_never_all] -= 2.0
    return rankdata(r, method="ordinal") / len(r)

def to_rank(x):
    return (rankdata(x) - 0.5) / len(x)

def to_probit(x):
    return norm.ppf(np.clip(to_rank(x), 1e-7, 1 - 1e-7))

# 3. Load Predictions
print("\n[Loading Component Predictions]")
p_pjw_stack = pd.read_csv('kernels_out/lr_margin_gbdt/submission.csv')['Will_Buy_EV'].values
p_v7_apex = pd.read_csv('submission_apex_frontier_hblend_v7_opt.csv')['Will_Buy_EV'].values
p_super_9m = np.load('preds/test_super_stacker_9model.npy')

# Our newly trained Option C models
our_res_lgb = np.load('preds/test_glm_margin_triple_te_lgbm.npy')
our_res_xgb = np.load('preds/test_glm_margin_xgb.npy')

p_realmlp = pd.read_csv('kernels_out/realmlp/submission.csv')['Will_Buy_EV'].values
dither_signal = np.load('preds/test_heuljax_xgb.npy')

print("  1. PJW GLM-Margin GBDT Stack (LB: 0.94675, OOF: 0.946638)")
print("  2. Our V7 Apex Frontier H-Blend (LB: 0.94668, Rank #81)")
print("  3. Our 9-Model 10-Fold Super-Stacker (CV: 0.946606)")
print("  4. Our Option C Residual LightGBM (OOF: 0.946364)")
print("  5. Our Option C Residual XGBoost (OOF: 0.946367)")
print("  6. RealMLP Tabular Deep NN (KS-Div: 0.1452)")

# ------------------------------------------------------------------------------
# V9 CANDIDATE 1: Apex Synergistic H-Blend V9 (with Option C Dual Residuals)
# ------------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Generating V9 Candidate 1: Apex Synergistic H-Blend V9")
# Matrix with 6 diverse models
matrix_v9 = np.column_stack([p_pjw_stack, p_v7_apex, p_super_9m, our_res_lgb, our_res_xgb, p_realmlp])
main_wts = np.array([0.52, 0.26, 0.08, 0.05, 0.05, 0.04])
correct_wts = np.array([+0.025, +0.015, 0.0, 0.0, -0.015, -0.025])

def h_blend(mat, mw, cw, asc_w=0.30, desc_w=0.70):
    def side(rev):
        order = np.argsort(-mat if rev else mat, axis=1)
        out = np.zeros(mat.shape[0], dtype=float)
        for rank_idx in range(mat.shape[1]):
            col_idx = order[:, rank_idx]
            new_weights = mw[col_idx] + cw[rank_idx]
            out += mat[np.arange(mat.shape[0]), col_idx] * new_weights
        return out
    return side(False) * asc_w + side(True) * desc_w

c1_raw = h_blend(matrix_v9, main_wts, correct_wts)
c1_final = apply_laws_and_zerotie(c1_raw, dither_signal)
sub1_path = 'submission_apex_frontier_v9_hblend_opt.csv'
pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c1_final}).to_csv(sub1_path, index=False)
print(f"  -> Successfully Saved {sub1_path}")

# ------------------------------------------------------------------------------
# V9 CANDIDATE 2: Probit Latent Nexus V9 (with Option C Dual Residuals)
# ------------------------------------------------------------------------------
print("\n" + "-" * 60)
print("Generating V9 Candidate 2: Probit Latent Nexus V9")
c2_probit = (
    0.54 * to_probit(p_pjw_stack) +
    0.26 * to_probit(p_v7_apex) +
    0.08 * to_probit(p_super_9m) +
    0.04 * to_probit(our_res_lgb) +
    0.04 * to_probit(our_res_xgb) +
    0.04 * to_probit(p_realmlp)
)
c2_final = apply_laws_and_zerotie(c2_probit, dither_signal)
sub2_path = 'submission_probit_latent_nexus_v9_opt.csv'
pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c2_final}).to_csv(sub2_path, index=False)
print(f"  -> Successfully Saved {sub2_path}")

print("\n" + "=" * 80)
print("FRONTIER SYNERGY V9 CANDIDATES READY FOR SUBMISSION!")
print("=" * 80)
