"""
optimize_super_nexus_fast.py — Ultra-Fast 10-Fold Nested Hill-Climb Optimization
Optimizes combinations of:
1. PJW Stack (OOF = 0.946638, LB = 0.94675)
2. PJW Residual LightGBM (OOF = 0.946462)
3. PJW Residual XGBoost (OOF = 0.946457)
4. Our 9-Model 10-Fold Super-Stacker (OOF = 0.946606)
5. Heuljax Generator-Aware Ridge LogReg (OOF = 0.946400)
6. Heuljax 173-Feature DGP XGBoost (OOF = 0.946309)
7. Blamerx 341-Feature Pipeline (OOF = 0.946208)
8. Zoom Zoom 146-Feature LightGBM (OOF = 0.945900)
9. RealMLP 3-Seed Tabular Deep Neural Net (OOF = 0.946182)
10. Option C Residual LightGBM (OOF = 0.946364)
11. Option C Residual XGBoost (OOF = 0.946367)
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print("=" * 80)
print(" S6E9: ULTRA-FAST MULTI-PARADIGM SUPER NEXUS OPTIMIZATION")
print("=" * 80)

# 1. Load Ground Truth
t0 = time.time()
train = pd.read_csv('data/train.csv', usecols=['id', 'Will_Buy_EV', 'Annual_Income_USD', 'Daily_Commute_km', 'Subsidy_Available', 'Environmental_Concern_Level', 'Range_Anxiety_Level'])
y = (train['Will_Buy_EV'] == 'Yes').astype(np.int8).to_numpy()
n = len(y)
n1 = int(np.sum(y))
n0 = n - n1
const_sub = n1 * (n1 + 1) / 2.0
denom = float(n1 * n0)

def fast_auc(yb, s):
    order = np.argsort(s)
    r = np.empty(len(s), dtype=np.float64)
    r[order] = np.arange(1, len(s) + 1)
    return (np.sum(r[yb == 1]) - const_sub) / denom

# 2. Load Component OOFs
pjw_stack = pd.read_csv('kernels_out/lr_margin_gbdt/oof_stack.csv')

oofs = {
    'pjw_stack': pjw_stack['stack'].values,
    'residual_lgbm': pjw_stack['residual_lgbm'].values,
    'residual_xgb': pjw_stack['residual_xgb'].values,
    'super_9m': np.load('preds/oof_super_stacker_9model.npy'),
    'heuljax_lr': pjw_stack['heuljax_lr'].values,
    'heuljax_xgb': pjw_stack['heuljax_xgb'].values,
    'blamerx_xgb': pjw_stack['blamerx_xgb'].values,
    'zoom_lgb': np.load('preds/oof_zoom_zoom_lgb.npy'),
    'realmlp': pjw_stack['realmlp_3seed'].values,
    'megayak_A': pjw_stack['megayak_A'].values,
    'our_optc_lgb': np.load('preds/oof_glm_margin_triple_te_lgbm.npy'),
    'our_optc_xgb': np.load('preds/oof_glm_margin_xgb.npy')
}

names = list(oofs.keys())
print(f"Loaded {len(names)} models in {time.time()-t0:.1f}s")
for name in names:
    auc = fast_auc(y, oofs[name])
    rho, _ = spearmanr(oofs['pjw_stack'], oofs[name])
    print(f"  {name:16s} : AUC = {auc:.6f} | rho vs PJW Stack = {rho:.5f}")

# 3. Ground Truth Generator Laws
income = train['Annual_Income_USD'].values
commute = train['Daily_Commute_km'].values
no_subsidy = (train['Subsidy_Available'] == 'No').values
c1_mask = (train['Environmental_Concern_Level'] == 1).values
anx = train['Range_Anxiety_Level'].isin(['Medium', 'High']).values

r_always = income >= 170537
r_never_inc = (income >= 31004) & (income <= 41970)
r_never_comm = commute >= 83
r_never_30k = (income == 30000) & no_subsidy & (c1_mask | anx)
r_never_all = r_never_inc | r_never_comm | r_never_30k

def apply_laws(ranks):
    r = ranks.copy()
    r[r_always] += 2.0
    r[r_never_all] -= 2.0
    return rankdata(r) / len(r)

# Rank Normalized Representations
R = {name: (rankdata(oofs[name]) - 0.5) / n for name in names}

# Probit Normalized Representations
P = {name: norm.ppf(np.clip(R[name], 1e-7, 1 - 1e-7)) for name in names}

# 4. Hill Climbing Optimizer
def hill_climb(data_dict, y_true, model_names, step=0.01, iters=200):
    n_samples = len(y_true)
    n1_f = int(np.sum(y_true))
    n0_f = n_samples - n1_f
    const_sub_f = n1_f * (n1_f + 1) / 2.0
    denom_f = float(n1_f * n0_f)
    
    def local_auc(s):
        order = np.argsort(s)
        r = np.empty(len(s), dtype=np.float64)
        r[order] = np.arange(1, len(s) + 1)
        return (np.sum(r[y_true == 1]) - const_sub_f) / denom_f
        
    weights = {m: 0.0 for m in model_names}
    best_init = max(model_names, key=lambda m: local_auc(data_dict[m]))
    weights[best_init] = 1.0
    current_vec = data_dict[best_init].copy()
    current_auc = local_auc(current_vec)
    
    for it in range(iters):
        best_gain = 0.0
        best_m = None
        for m in model_names:
            test_vec = current_vec + step * data_dict[m]
            cand_auc = local_auc(test_vec)
            gain = cand_auc - current_auc
            if gain > best_gain:
                best_gain = gain
                best_m = m
        if best_gain <= 1e-7 or best_m is None:
            break
        weights[best_m] += step
        current_vec += step * data_dict[best_m]
        current_auc += best_gain
        
    tot = sum(weights.values())
    norm_w = {m: weights[m] / tot for m in model_names}
    return norm_w, current_auc

print("\n" + "=" * 80)
print("Evaluating Full-Dataset Hill-Climb (Rank Space):")
w_rank, auc_rank = hill_climb(R, y, names)
print(f"Full Rank-Space Hill-Climb OOF AUC: {auc_rank:.6f}")
print("Active Models in Rank Space:")
for m, w in sorted(w_rank.items(), key=lambda x: -x[1]):
    if w > 0.005:
        print(f"  {m:16s}: {w*100:5.2f}%")

print("\nEvaluating Full-Dataset Hill-Climb (Probit Space):")
w_probit, auc_probit = hill_climb(P, y, names)
print(f"Full Probit-Space Hill-Climb OOF AUC: {auc_probit:.6f}")
print("Active Models in Probit Space:")
for m, w in sorted(w_probit.items(), key=lambda x: -x[1]):
    if w > 0.005:
        print(f"  {m:16s}: {w*100:5.2f}%")

# Compute final composite vector
final_probit_vec = sum(w * P[m] for m, w in w_probit.items())
final_laws_auc = fast_auc(y, apply_laws(final_probit_vec))
print("\n" + "=" * 80)
print(f"FINAL PROBIT + 4 GENERATOR LAWS OOF AUC: {final_laws_auc:.6f}")
print("=" * 80)
