import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm

print("="*70)
print("S6E9: ADVANCED HYPERPARAMETER TUNING & DIVERSE ENSEMBLE ENGINE")
print("="*70)

# 1. Load Test Set
test = pd.read_csv('data/test.csv')
test_ids = test['id'].values
n_test = len(test_ids)
print(f"Loaded test set: {n_test:,} rows")

# 2. Extract Ground Truth Generator Rules
income = pd.to_numeric(test['Annual_Income_USD'], errors='coerce').values
commute = pd.to_numeric(test['Daily_Commute_km'], errors='coerce').values
subsidy_no = (test['Subsidy_Available'].astype(str) == 'No').values
concern_1 = (pd.to_numeric(test['Environmental_Concern_Level'], errors='coerce') == 1.0).values
anx_high = test['Range_Anxiety_Level'].isin(['Medium', 'High']).values

r_always = income >= 170537.0
r_never_inc = (income >= 31004.0) & (income <= 41970.0)
r_never_comm = commute >= 83.0
r_never_30k = (income == 30000.0) & subsidy_no & (concern_1 | anx_high)
r_never_all = r_never_inc | r_never_comm | r_never_30k

print(f"Verified Physics: {r_always.sum():,} buyers, {r_never_all.sum():,} non-buyers (total {(r_always | r_never_all).sum():,} rows)")

# 3. Helpers
def rk01(x):
    return (rankdata(x) - 0.5) / len(x)

def lexsort_zero_ties(primary, secondary):
    order = np.lexsort((secondary, primary))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return (ranks - 0.5) / len(ranks)

# 4. Load Predictions
print("\n[Loading Models]")
gp = pd.read_csv('kernels_out/lucifer_grandprix/submission.csv')['Will_Buy_EV'].values
zz = pd.read_csv('kernels_out/zoom_zoom_v37/blend.csv')['Will_Buy_EV'].values
sub2 = pd.read_csv('submission_sub2_full_8model_stacker.csv')['Will_Buy_EV'].values
sub5 = pd.read_csv('submission_sub5_robust_rank_ensemble.csv')['Will_Buy_EV'].values
heul = np.load('preds/test_heuljax_xgb.npy')
rmlp = pd.read_csv('kernels_out/realmlp/submission.csv')['Will_Buy_EV'].values

r_gp = rk01(gp)
r_zz = rk01(zz)
r_sub2 = rk01(sub2)
r_sub5 = rk01(sub5)
r_heul = rk01(heul)
r_rmlp = rk01(rmlp)

r_frontier = 0.50 * r_gp + 0.50 * r_zz

# ==============================================================================
# CANDIDATE 5: Dynamic Confidence-Gated Synergy (Focusing diversity on contested rows)
# ==============================================================================
print("\n" + "="*70)
print("BUILDING CANDIDATE 5: Dynamic Confidence-Gated Synergy")
print("="*70)
# Contested zone: middle 70% of distribution
contested = (r_frontier >= 0.15) & (r_frontier <= 0.85)

gated_score = r_frontier.copy()
# In contested zone, blend 65% Frontier + 25% Sub2 8-Model Stacker + 10% RealMLP Deep NN
gated_score[contested] = (
    0.65 * r_frontier[contested] +
    0.25 * r_sub2[contested] +
    0.10 * r_rmlp[contested]
)
# Apply physics shifts
gated_score[r_always] += 10.0
gated_score[r_never_all] -= 10.0

final_c5 = lexsort_zero_ties(gated_score, r_heul)
df_c5 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': final_c5})
c5_path = 'submission_dynamic_gated_synergy_opt.csv'
df_c5.to_csv(c5_path, index=False, float_format='%.8f')
print(f"Saved: {c5_path} | Unique values: {len(np.unique(final_c5)):,}/{n_test:,} (Zero Ties)")

# ==============================================================================
# CANDIDATE 6: High-Diversity Orthogonal Hedge (60% Frontier + 25% Sub2 + 10% Heuljax + 5% RealMLP)
# ==============================================================================
print("\n" + "="*70)
print("BUILDING CANDIDATE 6: High-Diversity Orthogonal Hedge")
print("="*70)
div_score = (
    0.60 * r_frontier +
    0.25 * r_sub2 +
    0.10 * r_heul +
    0.05 * r_rmlp
)
div_score[r_always] += 10.0
div_score[r_never_all] -= 10.0

final_c6 = lexsort_zero_ties(div_score, r_zz)
df_c6 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': final_c6})
c6_path = 'submission_high_diversity_hedge_opt.csv'
df_c6.to_csv(c6_path, index=False, float_format='%.8f')
print(f"Saved: {c6_path} | Unique values: {len(np.unique(final_c6)):,}/{n_test:,} (Zero Ties)")

# ==============================================================================
# CANDIDATE 7: Non-Linear Probit Convex Stacker with Temperature Scaling
# ==============================================================================
print("\n" + "="*70)
print("BUILDING CANDIDATE 7: Probit Convex Stacker with Temperature Scaling")
print("="*70)
# Convert to continuous probit
eps = 1e-7
z_gp = norm.ppf(np.clip(r_gp, eps, 1-eps))
z_zz = norm.ppf(np.clip(r_zz, eps, 1-eps))
z_sub2 = norm.ppf(np.clip(r_sub2, eps, 1-eps))
z_sub5 = norm.ppf(np.clip(r_sub5, eps, 1-eps))
z_rmlp = norm.ppf(np.clip(r_rmlp, eps, 1-eps))

z_blend = (
    0.35 * z_gp +
    0.35 * z_zz +
    0.15 * z_sub2 +
    0.10 * z_sub5 +
    0.05 * z_rmlp
)
# Temperature scaling (T = 0.95 steepens margin sharpness)
z_blend = z_blend / 0.95

z_blend[r_always] += 10.0
z_blend[r_never_all] -= 10.0

final_c7 = lexsort_zero_ties(z_blend, r_heul)
df_c7 = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': final_c7})
c7_path = 'submission_probit_temp_scaled_opt.csv'
df_c7.to_csv(c7_path, index=False, float_format='%.8f')
print(f"Saved: {c7_path} | Unique values: {len(np.unique(final_c7)):,}/{n_test:,} (Zero Ties)")

print("\n" + "="*70)
print("ALL 3 ADVANCED TUNED CANDIDATES GENERATED SUCCESSFULLY!")
print("="*70)
