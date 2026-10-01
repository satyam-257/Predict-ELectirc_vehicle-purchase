"""
optimize_super_nexus.py — Multi-Paradigm Super Nexus Optimization
Tests nested 10-fold cross-validated ensembles across:
1. PJW Stack (OOF = 0.946638, LB = 0.94675)
2. PJW Residual LightGBM (OOF = 0.946462)
3. PJW Residual XGBoost (OOF = 0.946457)
4. Our 9-Model 10-Fold Leak-Free Super-Stacker (OOF = 0.946606)
5. Heuljax Generator-Aware Ridge LogReg (OOF = 0.946400)
6. Heuljax 173-Feature DGP XGBoost (OOF = 0.946309)
7. Blamerx 341-Feature Pipeline (OOF = 0.946208)
8. Zoom Zoom 146-Feature LightGBM (OOF = 0.945900)
9. RealMLP 3-Seed Tabular Deep Neural Net (OOF = 0.946182)
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

print("=" * 80)
print(" S6E9: SUPER NEXUS MULTI-PARADIGM ENSEMBLE OPTIMIZATION")
print("=" * 80)

# 1. Load Ground Truth
train = pd.read_csv('data/train.csv', usecols=['id', 'Will_Buy_EV', 'Annual_Income_USD', 'Daily_Commute_km', 'Subsidy_Available', 'Environmental_Concern_Level', 'Range_Anxiety_Level'])
y = (train['Will_Buy_EV'] == 'Yes').astype(np.int8).to_numpy()
n = len(y)

# 2. Load Component OOFs
pjw_stack = pd.read_csv('kernels_out/lr_margin_gbdt/oof_stack.csv')
pjw_mine = pd.read_csv('kernels_out/lr_margin_gbdt/oof_mine.csv')

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
    'megayak_A': pjw_stack['megayak_A'].values
}

names = list(oofs.keys())
print("\n[Loaded Candidate Models & Baseline OOF AUCs]")
for name in names:
    auc = roc_auc_score(y, oofs[name])
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

# 4. Rank Matrices
R_matrix = np.column_stack([rankdata(oofs[name]) / n for name in names])

# Probit Matrix
P_matrix = np.column_stack([norm.ppf(np.clip(R_matrix[:, i], 1e-7, 1 - 1e-7)) for i in range(len(names))])

# 5. Nested 10-Fold Hill-Climb & SLSQP Optimization
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def eval_slsqp_cv(X_mat):
    cv_oof = np.zeros(n)
    fold_weights = []
    
    for tr_idx, va_idx in skf.split(X_mat, y):
        X_tr, y_tr = X_mat[tr_idx], y[tr_idx]
        X_va = X_mat[va_idx]
        
        def loss(w):
            w = np.array(w)
            w = w / np.sum(w)
            pred = X_tr @ w
            return -roc_auc_score(y_tr, pred)
            
        init_w = np.ones(X_mat.shape[1]) / X_mat.shape[1]
        bounds = [(0, 1) for _ in range(X_mat.shape[1])]
        res = minimize(loss, init_w, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        best_w = res.x / np.sum(res.x)
        
        cv_oof[va_idx] = X_va @ best_w
        fold_weights.append(best_w)
        
    mean_w = np.mean(fold_weights, axis=0)
    base_auc = roc_auc_score(y, cv_oof)
    laws_auc = roc_auc_score(y, apply_laws(cv_oof))
    return base_auc, laws_auc, mean_w

print("\n" + "-" * 80)
print("Evaluating 10-Fold Nested CV Rank-Space Optimization:")
auc_rank_raw, auc_rank_laws, w_rank = eval_slsqp_cv(R_matrix)
print(f"  Rank-Space 10-Fold CV AUC (Raw):  {auc_rank_raw:.6f}")
print(f"  Rank-Space 10-Fold CV AUC (+Laws): {auc_rank_laws:.6f} (+{auc_rank_laws - auc_rank_raw:+.6f})")

print("\nEvaluating 10-Fold Nested CV Probit-Space Optimization:")
auc_probit_raw, auc_probit_laws, w_probit = eval_slsqp_cv(P_matrix)
print(f"  Probit-Space 10-Fold CV AUC (Raw):  {auc_probit_raw:.6f}")
print(f"  Probit-Space 10-Fold CV AUC (+Laws): {auc_probit_laws:.6f} (+{auc_probit_laws - auc_probit_raw:+.6f})")

print("\nLearned Model Weights (Probit-Space):")
for name, w in zip(names, w_probit):
    if w > 0.005:
        print(f"  {name:16s}: {w*100:5.2f}%")
