import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm
import hashlib
import os

print("="*70)
print("S6E9: FRONTIER FUSION & GROUND-TRUTH GENERATOR LAW ENSEMBLE ENGINE")
print("="*70)

# 1. Load Test Metadata
test = pd.read_csv('data/test.csv')
test_ids = test['id'].values
n_test = len(test_ids)
print(f"Loaded test set: {n_test:,} rows")

# 2. Extract Deterministic Generator Rule Masks
income = test['Annual_Income_USD'].values
commute = test['Daily_Commute_km'].values
no_subsidy = (test['Subsidy_Available'] == 'No').values
c1 = (test['Environmental_Concern_Level'] == 1).values
anx = test['Range_Anxiety_Level'].isin(['Medium', 'High']).values

r_always = income >= 170537
r_never_inc = (income >= 31004) & (income <= 41970)
r_never_comm = commute >= 83
r_never_30k = (income == 30000) & no_subsidy & (c1 | anx)
r_never_all = r_never_inc | r_never_comm | r_never_30k

print(f"\n[Ground-Truth Generator Laws Verification]")
print(f"  Rule 1 (income >= 170,537 -> Always Buys):       {r_always.sum():6,d} test rows")
print(f"  Rule 2 (income 31,004-41,970 -> Never Buys):     {r_never_inc.sum():6,d} test rows")
print(f"  Rule 3 (commute >= 83 km -> Never Buys):         {r_never_comm.sum():6,d} test rows")
print(f"  Rule 4 (30k, no sub, c1/anxious -> Never Buys):  {r_never_30k.sum():6,d} test rows")
print(f"  --> Total Test Rows Governed by Pure Physics:    {(r_always | r_never_all).sum():6,d} test rows")

def apply_generator_laws(score_vector):
    """
    Applies the 4 100.000% pure generator boundary conditions.
    Shifts verified buyers to absolute top and verified non-buyers to absolute bottom,
    preserving the internal ranking among those rows.
    """
    r = rankdata(score_vector) / len(score_vector)
    r[r_always] += 2.0
    r[r_never_all] -= 2.0
    return rankdata(r) / len(r)

# 3. Load Model Predictions
print(f"\n[Loading Component Predictions]")
models = {}

# A. Frontier Models
models['GrandPrix_48Engine'] = pd.read_csv('kernels_out/lucifer_grandprix/submission.csv')['Will_Buy_EV'].values
models['ZoomZoom_V37_Blend'] = pd.read_csv('kernels_out/zoom_zoom_v37/blend.csv')['Will_Buy_EV'].values
models['Nina_V7_HBlend'] = pd.read_csv('kernels_out/nina_hblend/submission.csv')['Will_Buy_EV'].values
models['Miickey_94657'] = pd.read_csv('kernels_out/miickey/submission_exact_public_94657.csv')['Will_Buy_EV'].values

# B. Locally-Trained Independent 10-Fold Models
models['Our_Sub2_8Model_Stacker'] = pd.read_csv('submission_sub2_full_8model_stacker.csv')['Will_Buy_EV'].values
models['Our_Sub5_Robust_Rank'] = pd.read_csv('submission_sub5_robust_rank_ensemble.csv')['Will_Buy_EV'].values
models['Our_Heuljax_10Fold_XGB'] = np.load('preds/test_heuljax_xgb.npy')
models['Our_BlamerX_10Fold_XGB'] = np.load('preds/test_blamerx_xgb.npy')
models['RealMLP_DeepNN'] = pd.read_csv('kernels_out/realmlp/submission.csv')['Will_Buy_EV'].values

for name, preds in models.items():
    print(f"  {name:26s}: shape={preds.shape}, range=[{preds.min():.5f}, {preds.max():.5f}]")

# 4. Probit and Normalized Rank Representations
def to_rank(x):
    return (rankdata(x) - 0.5) / len(x)

def to_probit(x):
    r = np.clip(to_rank(x), 1e-7, 1 - 1e-7)
    return norm.ppf(r)

ranks = {k: to_rank(v) for k, v in models.items()}
probits = {k: to_probit(v) for k, v in models.items()}

# 5. Build Submissions

# ------------------------------------------------------------------------------
# Candidate 1: Frontier Pure Optimal (Grand Prix + Zoom Zoom V37 + Nina V7) + Rules
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("BUILDING CANDIDATE 1: Frontier Pure Optimal + Generator Laws")
print("="*70)
c1_raw = (
    0.48 * ranks['GrandPrix_48Engine'] +
    0.48 * ranks['ZoomZoom_V37_Blend'] +
    0.04 * ranks['Nina_V7_HBlend']
)
c1_final = apply_generator_laws(c1_raw)
df_c1 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c1_final})
c1_path = 'submission_frontier_pure_optimal.csv'
df_c1.to_csv(c1_path, index=False)
print(f"Saved: {c1_path} (Range: [{c1_final.min():.6f}, {c1_final.max():.6f}])")

# ------------------------------------------------------------------------------
# Candidate 2: Synergy Frontier + Independent 8-Model Stacker + RealMLP Hedge + Rules
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("BUILDING CANDIDATE 2: Synergy Frontier + 8-Model Stacker + RealMLP + Generator Laws")
print("="*70)
# Hammad Farooq 88/10/2 Hedge rule: Inject independent diversity to protect against public overfitting
c2_frontier = 0.50 * ranks['GrandPrix_48Engine'] + 0.50 * ranks['ZoomZoom_V37_Blend']
c2_raw = (
    0.88 * c2_frontier +
    0.10 * ranks['Our_Sub2_8Model_Stacker'] +
    0.02 * ranks['RealMLP_DeepNN']
)
c2_final = apply_generator_laws(c2_raw)
df_c2 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c2_final})
c2_path = 'submission_synergy_frontier_stacker.csv'
df_c2.to_csv(c2_path, index=False)
print(f"Saved: {c2_path} (Range: [{c2_final.min():.6f}, {c2_final.max():.6f}])")

# ------------------------------------------------------------------------------
# Candidate 3: Probit Grand Master Fusion (Gaussian Latent Scale Fusion) + Rules
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("BUILDING CANDIDATE 3: Probit Gaussian Latent Scale Fusion + Generator Laws")
print("="*70)
# Aggregates across the continuous unbounded probit latent space
c3_latent = (
    0.42 * probits['GrandPrix_48Engine'] +
    0.42 * probits['ZoomZoom_V37_Blend'] +
    0.10 * probits['Our_Sub2_8Model_Stacker'] +
    0.04 * probits['Our_Heuljax_10Fold_XGB'] +
    0.02 * probits['RealMLP_DeepNN']
)
c3_final = apply_generator_laws(c3_latent)
df_c3 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c3_final})
c3_path = 'submission_probit_latent_fusion.csv'
df_c3.to_csv(c3_path, index=False)
print(f"Saved: {c3_path} (Range: [{c3_final.min():.6f}, {c3_final.max():.6f}])")

# ------------------------------------------------------------------------------
# Candidate 4: Multi-Resolution Frontier Master (Heuljax DGP + Zoom Zoom + Grand Prix) + Rules
# ------------------------------------------------------------------------------
print("\n" + "="*70)
print("BUILDING CANDIDATE 4: Multi-Resolution Master (DGP Heuljax + Blamerx + Frontier) + Generator Laws")
print("="*70)
c4_raw = (
    0.40 * ranks['GrandPrix_48Engine'] +
    0.40 * ranks['ZoomZoom_V37_Blend'] +
    0.10 * ranks['Our_Sub5_Robust_Rank'] +
    0.06 * ranks['Our_Heuljax_10Fold_XGB'] +
    0.04 * ranks['Our_BlamerX_10Fold_XGB']
)
c4_final = apply_generator_laws(c4_raw)
df_c4 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': c4_final})
c4_path = 'submission_multiresolution_frontier_master.csv'
df_c4.to_csv(c4_path, index=False)
print(f"Saved: {c4_path} (Range: [{c4_final.min():.6f}, {c4_final.max():.6f}])")

print("\n" + "="*70)
print("ALL 4 SUBMISSION CANDIDATES GENERATED SUCCESSFULLY!")
print("="*70)
