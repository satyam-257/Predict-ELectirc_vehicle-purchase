"""
Evaluates Ensembling Zoom Zoom LightGBM with Heuljax XGBoost and Blamerx XGBoost
Computes:
1. Pearson & Spearman rank correlations
2. Pairwise rank-average AUC grid search
3. Logit-space Nelder-Mead optimization
4. Generates optimal submission files for public LB testing
"""

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

# Load targets
train = pd.read_csv('data/train.csv', usecols=['Will_Buy_EV'])
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values if train['Will_Buy_EV'].dtype == object else train['Will_Buy_EV'].values.astype(int)

# Load OOFs
oof_zoom = np.load('preds/oof_zoom_zoom_lgb.npy')
oof_heuljax = np.load('preds/oof_heuljax_xgb.npy')
oof_blamerx = np.load('preds/oof_blamerx_xgb.npy')

test_zoom = np.load('preds/test_zoom_zoom_lgb.npy')
test_heuljax = np.load('preds/test_heuljax_xgb.npy')
test_blamerx = np.load('preds/test_blamerx_xgb.npy')
test_df = pd.read_csv('data/test.csv', usecols=['id'])

print("=== Individual Model OOF AUC ===")
print(f"Zoom Zoom LightGBM : {roc_auc_score(y, oof_zoom):.6f}")
print(f"Heuljax XGBoost    : {roc_auc_score(y, oof_heuljax):.6f}")
print(f"Blamerx XGBoost    : {roc_auc_score(y, oof_blamerx):.6f}")

def rank01(x):
    return (pd.Series(x).rank().values - 0.5) / len(x)

r_zoom = rank01(oof_zoom)
r_heuljax = rank01(oof_heuljax)
r_blamerx = rank01(oof_blamerx)

print("\n=== Spearman Rank Correlations ===")
print(f"Corr(Zoom, Heuljax) : {np.corrcoef(r_zoom, r_heuljax)[0, 1]:.5f}")
print(f"Corr(Zoom, Blamerx) : {np.corrcoef(r_blamerx, r_zoom)[0, 1]:.5f}")
print(f"Corr(Heuljax, Blamerx): {np.corrcoef(r_heuljax, r_blamerx)[0, 1]:.5f}")

# 1. 2-Model Rank Blend: Heuljax + Zoom
print("\n--- Grid Search: Heuljax + Zoom (Rank Blend) ---")
best_w_hz = 0
best_auc_hz = 0
for w in np.linspace(0.0, 0.4, 41):
    blend = (1 - w) * r_heuljax + w * r_zoom
    auc = roc_auc_score(y, blend)
    if auc > best_auc_hz:
        best_auc_hz = auc
        best_w_hz = w
print(f"Best: {1-best_w_hz:.2f} Heuljax + {best_w_hz:.2f} Zoom -> AUC: {best_auc_hz:.6f}")

# 2. 3-Model Rank Blend: Heuljax + Blamerx + Zoom
print("\n--- Grid Search: Heuljax (base 0.56) + Blamerx (base 0.44) + Zoom (Rank Blend) ---")
best_w_3 = (0, 0, 0)
best_auc_3 = 0
for w_zoom in np.linspace(0.0, 0.25, 26):
    w_rem = 1.0 - w_zoom
    for ratio_h in np.linspace(0.45, 0.65, 21):
        w_h = w_rem * ratio_h
        w_b = w_rem * (1.0 - ratio_h)
        blend = w_h * r_heuljax + w_b * r_blamerx + w_zoom * r_zoom
        auc = roc_auc_score(y, blend)
        if auc > best_auc_3:
            best_auc_3 = auc
            best_w_3 = (w_h, w_b, w_zoom)

print(f"Best 3-Way Rank Blend: {best_w_3[0]:.3f} Heuljax + {best_w_3[1]:.3f} Blamerx + {best_w_3[2]:.3f} Zoom -> AUC: {best_auc_3:.6f}")

# 3. Logit-Space Hierarchical Calibration Blend
print("\n--- Logit-Space Calibration Blend ---")
eps = 1e-6
l_zoom = logit(np.clip(oof_zoom, eps, 1 - eps))
l_heuljax = logit(np.clip(oof_heuljax, eps, 1 - eps))
l_blamerx = logit(np.clip(oof_blamerx, eps, 1 - eps))

best_logit_w = (0, 0, 0)
best_logit_auc = 0
for w_zoom in np.linspace(0.0, 0.20, 21):
    w_rem = 1.0 - w_zoom
    for ratio_h in np.linspace(0.48, 0.62, 15):
        w_h = w_rem * ratio_h
        w_b = w_rem * (1.0 - ratio_h)
        blend_logit = expit(w_h * l_heuljax + w_b * l_blamerx + w_zoom * l_zoom)
        auc = roc_auc_score(y, blend_logit)
        if auc > best_logit_auc:
            best_logit_auc = auc
            best_logit_w = (w_h, w_b, w_zoom)

print(f"Best Logit-Space Blend: {best_logit_w[0]:.3f} Heuljax + {best_logit_w[1]:.3f} Blamerx + {best_logit_w[2]:.3f} Zoom -> AUC: {best_logit_auc:.6f}")

# Generate submission for best rank blend
test_r_zoom = rank01(test_zoom)
test_r_heuljax = rank01(test_heuljax)
test_r_blamerx = rank01(test_blamerx)

sub_rank = pd.DataFrame({
    'id': test_df['id'],
    'Will_Buy_EV': best_w_3[0] * test_r_heuljax + best_w_3[1] * test_r_blamerx + best_w_3[2] * test_r_zoom
})
sub_rank.to_csv('submission_zoom_grand_rank_ensemble.csv', index=False)
print("Saved submission_zoom_grand_rank_ensemble.csv")

# Generate submission for best logit blend
test_l_zoom = logit(np.clip(test_zoom, eps, 1 - eps))
test_l_heuljax = logit(np.clip(test_heuljax, eps, 1 - eps))
test_l_blamerx = logit(np.clip(test_blamerx, eps, 1 - eps))

sub_logit = pd.DataFrame({
    'id': test_df['id'],
    'Will_Buy_EV': expit(best_logit_w[0] * test_l_heuljax + best_logit_w[1] * test_l_blamerx + best_logit_w[2] * test_l_zoom)
})
sub_logit.to_csv('submission_zoom_grand_logit_ensemble.csv', index=False)
print("Saved submission_zoom_grand_logit_ensemble.csv")
