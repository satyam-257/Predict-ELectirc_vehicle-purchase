import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
from scipy.stats import rankdata
from scipy.optimize import minimize

# 1. Load Ground Truth
train = pd.read_csv('data/train.csv', usecols=['id', 'Will_Buy_EV'])
y_true = train['Will_Buy_EV'].values
n_train = len(y_true)

test = pd.read_csv('data/test.csv', usecols=['id'])
test_ids = test['id'].values
n_test = len(test_ids)

print(f"Loaded ground truth: {n_train} train rows, {n_test} test rows.")

# 2. Load all 8 verified models
models = {}

# 1. Heuljax XGB
models['Heuljax_XGB'] = {
    'oof': np.load('preds/oof_heuljax_xgb.npy'),
    'test': np.load('preds/test_heuljax_xgb.npy')
}

# 2. BlamerX XGB
models['BlamerX_XGB'] = {
    'oof': np.load('preds/oof_blamerx_xgb.npy'),
    'test': np.load('preds/test_blamerx_xgb.npy')
}

# 3. Zoom Zoom LGBM
models['Zoom_LGBM'] = {
    'oof': np.load('preds/oof_zoom_zoom_lgb.npy'),
    'test': np.load('preds/test_zoom_zoom_lgb.npy')
}

# 4. RealMLP (Neural Net)
df_oof_rmlp = pd.read_csv('kernels_out/realmlp/oof_preds.csv')
df_sub_rmlp = pd.read_csv('kernels_out/realmlp/submission.csv')
models['RealMLP_NN'] = {
    'oof': df_oof_rmlp['Will_Buy_EV'].values,
    'test': df_sub_rmlp['Will_Buy_EV'].values
}

# 5. CatBoost Naji
df_oof_cat = pd.read_csv('kernels_out/catboost_naji/oof_CATBOOST.csv')
df_sub_cat = pd.read_csv('kernels_out/catboost_naji/submission_CATBOOST.csv')
models['CatBoost_Naji'] = {
    'oof': df_oof_cat.iloc[:, 1].values,
    'test': df_sub_cat.iloc[:, 1].values
}

# 6. XGBoost Naji
df_oof_xgbn = pd.read_csv('kernels_out/xgboost_naji/oof_XGBOOST.csv')
df_sub_xgbn = pd.read_csv('kernels_out/xgboost_naji/submission_XGBOOST.csv')
models['XGBoost_Naji'] = {
    'oof': df_oof_xgbn.iloc[:, 1].values,
    'test': df_sub_xgbn.iloc[:, 1].values
}

# 7. Aryan HistGBM
df_oof_aryan = pd.read_csv('kernels_out/aryan_histgbm/oof_proba.csv')
df_test_aryan = pd.read_csv('kernels_out/aryan_histgbm/test_proba.csv')
models['Aryan_HistGBM'] = {
    'oof': df_oof_aryan.iloc[:, 1].values,
    'test': df_test_aryan.iloc[:, 1].values
}

# 8. Mizushima LGBM
df_oof_miz = pd.read_csv('kernels_out/mizushima_lgbm/oof_predictions.csv')
df_sub_miz = pd.read_csv('kernels_out/mizushima_lgbm/submission.csv')
models['Mizushima_LGBM'] = {
    'oof': df_oof_miz['Will_Buy_EV'].values,
    'test': df_sub_miz['Will_Buy_EV'].values
}

print("\n--- Verified 8 Candidate Models ---")
for name, d in models.items():
    auc = roc_auc_score(y_true, d['oof'])
    print(f"  {name:16s}: OOF AUC = {auc:.6f}")

# Previous record in our repo
prev_best_oof = 0.501 * logit(np.clip(models['Heuljax_XGB']['oof'], 1e-6, 1-1e-6)) + \
                0.349 * logit(np.clip(models['BlamerX_XGB']['oof'], 1e-6, 1-1e-6)) + \
                0.150 * logit(np.clip(models['Zoom_LGBM']['oof'], 1e-6, 1-1e-6))
print(f"\nPrevious Baseline Best (Heuljax 50.1% + Blamerx 34.9% + Zoom 15.0%) OOF AUC = {roc_auc_score(y_true, prev_best_oof):.6f}")

# Fast Vectorized AUC helper
def fast_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

eps = 1e-6
def to_logit(p):
    return logit(np.clip(p, eps, 1-eps))

# ==============================================================================
# SUBMISSION 1: Multi-Paradigm Elite Quad (XGBoost + CatBoost + LightGBM + RealMLP)
# Diverse representation from all 4 foundational model families!
# ==============================================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Multi-Paradigm Elite Quad (Heuljax_XGB, CatBoost, Zoom_LGBM, RealMLP)")
print("="*60)

quad_keys = ['Heuljax_XGB', 'CatBoost_Naji', 'Zoom_LGBM', 'RealMLP_NN']
X_quad = np.column_stack([to_logit(models[k]['oof']) for k in quad_keys])
T_quad = np.column_stack([to_logit(models[k]['test']) for k in quad_keys])

def loss_quad(w):
    w = np.array(w)
    w = w / np.sum(w)
    pred = X_quad @ w
    return -fast_auc(y_true, pred)

res_quad = minimize(loss_quad, [0.45, 0.25, 0.15, 0.15], method='Nelder-Mead', options={'maxiter': 300})
best_w_quad = res_quad.x / np.sum(res_quad.x)
oof_sub1 = X_quad @ best_w_quad
auc_sub1 = fast_auc(y_true, oof_sub1)
print(f"Optimal Quad Weights:")
for k, w in zip(quad_keys, best_w_quad):
    print(f"  {k:16s}: {w*100:.2f}%")
print(f"--> Multi-Paradigm Elite Quad OOF AUC = {auc_sub1:.6f}")

test_sub1_prob = expit(T_quad @ best_w_quad)
sub1_df = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_sub1_prob})
sub1_df.to_csv('submission_sub1_multi_paradigm_quad.csv', index=False)
print("Saved: submission_sub1_multi_paradigm_quad.csv")

# ==============================================================================
# SUBMISSION 2: Full 8-Model Constrained Convex Logit Stacker
# ==============================================================================
print("\n" + "="*60)
print("EXPERIMENT 2: Full 8-Model Constrained Convex Logit Stacker")
print("="*60)

all_keys = list(models.keys())
X_all = np.column_stack([to_logit(models[k]['oof']) for k in all_keys])
T_all = np.column_stack([to_logit(models[k]['test']) for k in all_keys])

def loss_all(w):
    w = np.array(w)
    w = np.clip(w, 0, None)
    if np.sum(w) == 0:
        return 0
    w = w / np.sum(w)
    pred = X_all @ w
    return -fast_auc(y_true, pred)

init_w = np.array([0.30, 0.25, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05])
res_all = minimize(loss_all, init_w, method='Powell', options={'maxiter': 500})
best_w_all = np.clip(res_all.x, 0, None)
best_w_all = best_w_all / np.sum(best_w_all)
oof_sub2 = X_all @ best_w_all
auc_sub2 = fast_auc(y_true, oof_sub2)
print("Optimal 8-Model Weights:")
for k, w in zip(all_keys, best_w_all):
    print(f"  {k:16s}: {w*100:.2f}%")
print(f"--> Full 8-Model Constrained Convex Logit Stacker OOF AUC = {auc_sub2:.6f}")

test_sub2_prob = expit(T_all @ best_w_all)
sub2_df = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_sub2_prob})
sub2_df.to_csv('submission_sub2_full_8model_stacker.csv', index=False)
print("Saved: submission_sub2_full_8model_stacker.csv")

# ==============================================================================
# SUBMISSION 3: Tilii-Inspired CDF/Quantile Orthogonality Blend
# Quantile-aligned RealMLP into Heuljax tree distribution + BlamerX + CatBoost
# ==============================================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Tilii-Inspired CDF / Quantile Orthogonality Blend")
print("="*60)

sorted_heul_oof = np.sort(models['Heuljax_XGB']['oof'])
quantiles_train = np.linspace(0, 1, n_train)
mapped_realmlp_oof = np.interp(rankdata(models['RealMLP_NN']['oof'])/n_train, quantiles_train, sorted_heul_oof)

sorted_heul_test = np.sort(models['Heuljax_XGB']['test'])
quantiles_test = np.linspace(0, 1, n_test)
mapped_realmlp_test = np.interp(rankdata(models['RealMLP_NN']['test'])/n_test, quantiles_test, sorted_heul_test)

cdf_keys = ['Heuljax_XGB', 'BlamerX_XGB', 'CatBoost_Naji', 'Zoom_LGBM']
X_cdf = np.column_stack([to_logit(models[k]['oof']) for k in cdf_keys] + [to_logit(mapped_realmlp_oof)])
T_cdf = np.column_stack([to_logit(models[k]['test']) for k in cdf_keys] + [to_logit(mapped_realmlp_test)])

def loss_cdf(w):
    w = np.array(w)
    w = np.clip(w, 0, None)
    if np.sum(w) == 0: return 0
    w = w / np.sum(w)
    pred = X_cdf @ w
    return -fast_auc(y_true, pred)

res_cdf = minimize(loss_cdf, [0.35, 0.25, 0.15, 0.10, 0.15], method='Nelder-Mead', options={'maxiter': 300})
best_w_cdf = np.clip(res_cdf.x, 0, None)
best_w_cdf = best_w_cdf / np.sum(best_w_cdf)
oof_sub3 = X_cdf @ best_w_cdf
auc_sub3 = fast_auc(y_true, oof_sub3)
print("Optimal CDF-Calibrated Weights:")
for k, w in zip(cdf_keys + ['Mapped_RealMLP'], best_w_cdf):
    print(f"  {k:16s}: {w*100:.2f}%")
print(f"--> Tilii CDF-Calibrated Blend OOF AUC = {auc_sub3:.6f}")

test_sub3_prob = expit(T_cdf @ best_w_cdf)
sub3_df = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_sub3_prob})
sub3_df.to_csv('submission_sub3_tilii_cdf_calibrated.csv', index=False)
print("Saved: submission_sub3_tilii_cdf_calibrated.csv")

# ==============================================================================
# SUBMISSION 4: High-Order Tree Champions + RealMLP Synergy (Heuljax + Blamerx + CatBoost + RealMLP)
# ==============================================================================
print("\n" + "="*60)
print("EXPERIMENT 4: Top Tier 5-Way Logit (Heuljax, Blamerx, CatBoost, Aryan_HistGBM, RealMLP)")
print("="*60)

sub4_keys = ['Heuljax_XGB', 'BlamerX_XGB', 'CatBoost_Naji', 'Aryan_HistGBM', 'RealMLP_NN']
X_sub4 = np.column_stack([to_logit(models[k]['oof']) for k in sub4_keys])
T_sub4 = np.column_stack([to_logit(models[k]['test']) for k in sub4_keys])

def loss_sub4(w):
    w = np.array(w)
    w = np.clip(w, 0, None)
    if np.sum(w) == 0: return 0
    w = w / np.sum(w)
    pred = X_sub4 @ w
    return -fast_auc(y_true, pred)

res_sub4 = minimize(loss_sub4, [0.35, 0.30, 0.15, 0.10, 0.10], method='Nelder-Mead', options={'maxiter': 300})
best_w_sub4 = np.clip(res_sub4.x, 0, None)
best_w_sub4 = best_w_sub4 / np.sum(best_w_sub4)
oof_sub4 = X_sub4 @ best_w_sub4
auc_sub4 = fast_auc(y_true, oof_sub4)
print("Optimal Top-5 Weights:")
for k, w in zip(sub4_keys, best_w_sub4):
    print(f"  {k:16s}: {w*100:.2f}%")
print(f"--> Top-5 Synergy Logit Blend OOF AUC = {auc_sub4:.6f}")

test_sub4_prob = expit(T_sub4 @ best_w_sub4)
sub4_df = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_sub4_prob})
sub4_df.to_csv('submission_sub4_top5_synergy_logit.csv', index=False)
print("Saved: submission_sub4_top5_synergy_logit.csv")

# ==============================================================================
# SUBMISSION 5: Robust Rank-Average Ensemble across Top 5 Distinct Pipelines
# ==============================================================================
print("\n" + "="*60)
print("EXPERIMENT 5: Robust Normalized Rank-Average Ensemble")
print("="*60)

R_oof = np.column_stack([rankdata(models[k]['oof']) / n_train for k in sub4_keys])
R_test = np.column_stack([rankdata(models[k]['test']) / n_test for k in sub4_keys])

def loss_rank(w):
    w = np.array(w)
    w = np.clip(w, 0, None)
    if np.sum(w) == 0: return 0
    w = w / np.sum(w)
    pred = R_oof @ w
    return -fast_auc(y_true, pred)

res_rank = minimize(loss_rank, [0.35, 0.30, 0.15, 0.10, 0.10], method='Nelder-Mead', options={'maxiter': 300})
best_w_rank = np.clip(res_rank.x, 0, None)
best_w_rank = best_w_rank / np.sum(best_w_rank)
oof_sub5 = R_oof @ best_w_rank
auc_sub5 = fast_auc(y_true, oof_sub5)
print("Optimal Rank Weights:")
for k, w in zip(sub4_keys, best_w_rank):
    print(f"  {k:16s}: {w*100:.2f}%")
print(f"--> Robust Normalized Rank-Average OOF AUC = {auc_sub5:.6f}")

test_sub5_prob = R_test @ best_w_rank
sub5_df = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_sub5_prob})
sub5_df.to_csv('submission_sub5_robust_rank_ensemble.csv', index=False)
print("Saved: submission_sub5_robust_rank_ensemble.csv")

print("\n" + "="*60)
print("SUMMARY OF ALL 5 NEW SUBMISSION CANDIDATES")
print("="*60)
print(f"Baseline (Previous Best Submission):           OOF AUC = {roc_auc_score(y_true, prev_best_oof):.6f}")
print(f"Sub 1 (Multi-Paradigm Elite Quad):             OOF AUC = {auc_sub1:.6f}")
print(f"Sub 2 (Full 8-Model Constrained Convex Stacker): OOF AUC = {auc_sub2:.6f}")
print(f"Sub 3 (Tilii CDF-Calibrated Blend):            OOF AUC = {auc_sub3:.6f}")
print(f"Sub 4 (Top-5 Synergy Logit Blend):             OOF AUC = {auc_sub4:.6f}")
print(f"Sub 5 (Robust Normalized Rank Ensemble):       OOF AUC = {auc_sub5:.6f}")
