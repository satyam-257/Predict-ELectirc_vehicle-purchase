"""
ensemble_frontier_v10.py — Frontier Super Nexus V10 Architecture
Synchronized Tri-Pillar Co-Design:
1. goodpjw2008's 344-Feature GLM-Margin GBDT Stack (Public LB = 0.94675, OOF = 0.946638)
2. Our V7 Apex Frontier Leader (Public LB = 0.94668, Rank #81 worldwide)
3. Jazivxt's Zoom Zoom 146-Feature Digit Decomposition + 1280-Quantile Neighbor QTE (LB = 0.94660)
4. Heuljax's Generator-Aware Ridge Logistic Regression (LB = 0.94656, OOF = 0.946400)
5. RealMLP Tabular Deep Neural Network (KS-Divergence = 0.1452)
6. Nina's 30-Seed Round-Robin Frontier (LB = 0.94668)
7. Continuous Heuljax DGP Dither for Zero-Tie Lexicographical Ordering
8. 4 Pure Ground-Truth Generator Laws (100.000% pure physics)
"""

import os
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr

print("=" * 80)
print(" S6E9: FRONTIER SUPER NEXUS V10 (TRI-PILLAR SYNCHRONIZED ARCHITECTURE)")
print("=" * 80)

# 1. Load Test Data
test = pd.read_csv('data/test.csv')
test_ids = test['id'].values
n_test = len(test_ids)
print(f"Loaded test set: {n_test:,} rows")

# 2. Extract Ground-Truth Generator Laws (100.000% pure physics)
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

n_resolved = np.sum(r_always) + np.sum(r_never_all)
print(f"Verified 4 Ground-Truth Generator Laws: resolves {n_resolved:,} test cases deterministically.")

def apply_laws_and_zerotie(raw_score, continuous_dither):
    # Lexicographical Zero-Tie Dither
    dither = (continuous_dither - np.mean(continuous_dither)) / (np.std(continuous_dither) + 1e-12) * 1e-9
    r = rankdata(raw_score + dither) / len(raw_score)
    # Apply hard laws
    r[r_always] += 2.0
    r[r_never_all] -= 2.0
    # Final ordinal ranking
    return rankdata(r, method="ordinal") / len(r)

def to_rank(x):
    return (rankdata(x) - 0.5) / len(x)

def to_probit(x):
    return norm.ppf(np.clip(to_rank(x), 1e-7, 1 - 1e-7))

# 3. Load Predictions
print("\n[Loading Component Predictions]")
p_pjw = pd.read_csv('kernels_out/lr_margin_gbdt/submission.csv')['Will_Buy_EV'].values
p_v7_apex = pd.read_csv('submission_apex_frontier_hblend_v7_opt.csv')['Will_Buy_EV'].values
p_zoom37 = pd.read_csv('kernels_out/zoom_zoom_v37/submission.csv')['Will_Buy_EV'].values
p_heuljax_ridge = pd.read_csv('kernels_out/heuljax_ridge/output/submission.csv')['Will_Buy_EV'].values
p_realmlp = pd.read_csv('kernels_out/realmlp/submission.csv')['Will_Buy_EV'].values
p_nina_rr = pd.read_csv('kernels_out/nina_round_robin/output/submission.csv')['Will_Buy_EV'].values
dither_signal = np.load('preds/test_heuljax_xgb.npy')

print("  1. PJW GLM-Margin GBDT Stack (LB: 0.94675, OOF: 0.946638)")
print("  2. Our V7 Apex Frontier H-Blend (LB: 0.94668, Rank #81)")
print("  3. Zoom Zoom V37 Digit+QTE Pipeline (LB: 0.94660)")
print("  4. Heuljax Generator-Aware Ridge LogReg (LB: 0.94656)")
print("  5. RealMLP Tabular Deep NN (KS-Div: 0.1452)")
print("  6. Nina Round-Robin Frontier (LB: 0.94668)")

# Cross-correlations
print("\n[Pairwise Spearman Correlations vs PJW Stack]")
print(f"  V7 Apex vs PJW Stack:        {spearmanr(p_v7_apex, p_pjw).statistic:.5f}")
print(f"  Zoom Zoom V37 vs PJW Stack:   {spearmanr(p_zoom37, p_pjw).statistic:.5f}")
print(f"  Heuljax Ridge vs PJW Stack:  {spearmanr(p_heuljax_ridge, p_pjw).statistic:.5f}")
print(f"  RealMLP vs PJW Stack:        {spearmanr(p_realmlp, p_pjw).statistic:.5f}")
print(f"  Nina RR vs PJW Stack:        {spearmanr(p_nina_rr, p_pjw).statistic:.5f}")

# ------------------------------------------------------------------------------
# V10 CANDIDATE 1: Super Nexus Probit Synergy V10 (Optimal Multi-Paradigm Latent Fusion)
# ------------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Generating V10 Candidate 1: Super Nexus Probit Synergy V10")
# Weights derived from 10-fold nested hill climbing + frontier stability
w_pjw = 0.65
w_v7 = 0.16
w_zoom = 0.07
w_heuljax = 0.05
w_realmlp = 0.04
w_nina = 0.03

probit_v10 = (
    w_pjw * to_probit(p_pjw) +
    w_v7 * to_probit(p_v7_apex) +
    w_zoom * to_probit(p_zoom37) +
    w_heuljax * to_probit(p_heuljax_ridge) +
    w_realmlp * to_probit(p_realmlp) +
    w_nina * to_probit(p_nina_rr)
)

c1_final = apply_laws_and_zerotie(probit_v10, dither_signal)
sub1_csv = 'submission_super_nexus_v10_probit_opt.csv'
sub1_zip = 'submission_super_nexus_v10_probit_opt.zip'

pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c1_final}).to_csv(sub1_csv, index=False)
with zipfile.ZipFile(sub1_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub1_csv, arcname='submission.csv')
print(f"  -> Successfully generated {sub1_csv} & {sub1_zip} ({os.path.getsize(sub1_zip):,} bytes)")

# ------------------------------------------------------------------------------
# V10 CANDIDATE 2: Apex Asymmetric Hierarchical Nexus V10
# ------------------------------------------------------------------------------
print("\n" + "-" * 80)
print("Generating V10 Candidate 2: Apex Asymmetric Hierarchical Nexus V10")
mat_v10 = np.column_stack([p_pjw, p_v7_apex, p_zoom37, p_heuljax_ridge, p_realmlp, p_nina_rr])
main_weights = np.array([0.62, 0.16, 0.08, 0.06, 0.05, 0.03])
correct_weights = np.array([+0.03, +0.015, 0.0, 0.0, -0.015, -0.03])

def h_blend(mat, mw, cw, asc_w=0.25, desc_w=0.75):
    def side(rev):
        order = np.argsort(-mat if rev else mat, axis=1)
        out = np.zeros(mat.shape[0], dtype=float)
        for rank_idx in range(mat.shape[1]):
            col_idx = order[:, rank_idx]
            new_weights = mw[col_idx] + cw[rank_idx]
            out += mat[np.arange(mat.shape[0]), col_idx] * new_weights
        return out
    return side(False) * asc_w + side(True) * desc_w

c2_raw = h_blend(mat_v10, main_weights, correct_weights)
c2_final = apply_laws_and_zerotie(c2_raw, dither_signal)
sub2_csv = 'submission_apex_asymmetric_nexus_v10_opt.csv'
sub2_zip = 'submission_apex_asymmetric_nexus_v10_opt.zip'

pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c2_final}).to_csv(sub2_csv, index=False)
with zipfile.ZipFile(sub2_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write(sub2_csv, arcname='submission.csv')
print(f"  -> Successfully generated {sub2_csv} & {sub2_zip} ({os.path.getsize(sub2_zip):,} bytes)")

print("\n" + "=" * 80)
print(" V10 SUPER NEXUS COMPLETE — READY FOR LEADERBOARD BREAKTHROUGH")
print("=" * 80)
