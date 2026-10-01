import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier

# 1. Load Ground Truth
train = pd.read_csv('data/train.csv', usecols=['id', 'Will_Buy_EV'])
y_true = train['Will_Buy_EV'].values

# 2. Load OOF Predictions
oofs = {
    'heuljax': np.load('preds/oof_heuljax_xgb.npy'),
    'blamerx': np.load('preds/oof_blamerx_xgb.npy'),
    'zoom_lgb': np.load('preds/oof_zoom_zoom_lgb.npy'),
    'catboost': pd.read_csv('kernels_out/catboost_naji/oof_CATBOOST.csv').iloc[:, 1].values,
    'realmlp': pd.read_csv('kernels_out/realmlp/oof_preds.csv')['Will_Buy_EV'].values,
    'xgb_naji': pd.read_csv('kernels_out/xgboost_naji/oof_XGBOOST.csv').iloc[:, 1].values
}

names = list(oofs.keys())
print("Loaded models:", names)
for k, v in oofs.items():
    print(f"  {k:12s}: AUC = {roc_auc_score(y_true, v):.6f}")

# 3. Baseline previous best
prev_best_blend = 0.501 * logit(np.clip(oofs['heuljax'], 1e-6, 1-1e-6)) + \
                  0.349 * logit(np.clip(oofs['blamerx'], 1e-6, 1-1e-6)) + \
                  0.150 * logit(np.clip(oofs['zoom_lgb'], 1e-6, 1-1e-6))
print(f"\nPrevious Best (Heuljax + Blamerx + Zoom) Logit AUC = {roc_auc_score(y_true, prev_best_blend):.6f}")

# 4. Out-of-Fold Cross-Validated Stacking / Blending to avoid meta-leakage
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Test A: Logit-Space Convex Optimization
eps = 1e-6
logit_matrix = np.column_stack([logit(np.clip(oofs[k], eps, 1-eps)) for k in names])
rank_matrix = np.column_stack([rankdata(oofs[k]) / len(y_true) for k in names])
prob_matrix = np.column_stack([oofs[k] for k in names])

def optimize_weights_logit(X_train, y_train, X_val):
    n_models = X_train.shape[1]
    
    def loss(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = X_train @ w
        return -roc_auc_score(y_train, pred)
        
    init_w = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    res = minimize(loss, init_w, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    best_w = res.x / np.sum(res.x)
    return X_val @ best_w, best_w

# Evaluate CV of Logit Blending
cv_oof_logit = np.zeros(len(y_true))
cv_weights = []

for train_idx, val_idx in skf.split(logit_matrix, y_true):
    val_pred, w = optimize_weights_logit(logit_matrix[train_idx], y_true[train_idx], logit_matrix[val_idx])
    cv_oof_logit[val_idx] = val_pred
    cv_weights.append(w)

mean_weights = np.mean(cv_weights, axis=0)
cv_logit_auc = roc_auc_score(y_true, cv_oof_logit)
print(f"\n10-Fold CV Logit-Blend (All 6 Models) AUC = {cv_logit_auc:.6f}")
print("Mean Optimal Weights:")
for name, w in zip(names, mean_weights):
    print(f"  {name:12s}: {w*100:.2f}%")

# Test B: Adding RealMLP to Top-3 (Heuljax + Blamerx + Zoom + RealMLP)
subset_4 = ['heuljax', 'blamerx', 'zoom_lgb', 'realmlp']
X_sub4 = np.column_stack([logit(np.clip(oofs[k], eps, 1-eps)) for k in subset_4])
cv_oof_sub4 = np.zeros(len(y_true))
sub4_weights = []
for train_idx, val_idx in skf.split(X_sub4, y_true):
    val_pred, w = optimize_weights_logit(X_sub4[train_idx], y_true[train_idx], X_sub4[val_idx])
    cv_oof_sub4[val_idx] = val_pred
    sub4_weights.append(w)
print(f"\n10-Fold CV 4-Way Logit (Heuljax + Blamerx + Zoom + RealMLP) AUC = {roc_auc_score(y_true, cv_oof_sub4):.6f}")
for name, w in zip(subset_4, np.mean(sub4_weights, axis=0)):
    print(f"  {name:12s}: {w*100:.2f}%")

# Test C: Adding CatBoost to Top-3 (Heuljax + Blamerx + Zoom + CatBoost)
subset_cat = ['heuljax', 'blamerx', 'zoom_lgb', 'catboost']
X_subcat = np.column_stack([logit(np.clip(oofs[k], eps, 1-eps)) for k in subset_cat])
cv_oof_cat = np.zeros(len(y_true))
subcat_weights = []
for train_idx, val_idx in skf.split(X_subcat, y_true):
    val_pred, w = optimize_weights_logit(X_subcat[train_idx], y_true[train_idx], X_subcat[val_idx])
    cv_oof_cat[val_idx] = val_pred
    subcat_weights.append(w)
print(f"\n10-Fold CV 4-Way Logit (Heuljax + Blamerx + Zoom + CatBoost) AUC = {roc_auc_score(y_true, cv_oof_cat):.6f}")
for name, w in zip(subset_cat, np.mean(subcat_weights, axis=0)):
    print(f"  {name:12s}: {w*100:.2f}%")

# Test D: Top-5 (Heuljax + Blamerx + Zoom + RealMLP + CatBoost)
subset_5 = ['heuljax', 'blamerx', 'zoom_lgb', 'realmlp', 'catboost']
X_sub5 = np.column_stack([logit(np.clip(oofs[k], eps, 1-eps)) for k in subset_5])
cv_oof_sub5 = np.zeros(len(y_true))
sub5_weights = []
for train_idx, val_idx in skf.split(X_sub5, y_true):
    val_pred, w = optimize_weights_logit(X_sub5[train_idx], y_true[train_idx], X_sub5[val_idx])
    cv_oof_sub5[val_idx] = val_pred
    sub5_weights.append(w)
print(f"\n10-Fold CV 5-Way Logit (Heuljax + Blamerx + Zoom + RealMLP + CatBoost) AUC = {roc_auc_score(y_true, cv_oof_sub5):.6f}")
for name, w in zip(subset_5, np.mean(sub5_weights, axis=0)):
    print(f"  {name:12s}: {w*100:.2f}%")

# Test E: Rank Averaging across subset_5
def optimize_weights_rank(X_train, y_train, X_val):
    n_models = X_train.shape[1]
    def loss(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred = X_train @ w
        return -roc_auc_score(y_train, pred)
    init_w = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    res = minimize(loss, init_w, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    best_w = res.x / np.sum(res.x)
    return X_val @ best_w, best_w

X_rank5 = np.column_stack([rankdata(oofs[k]) / len(y_true) for k in subset_5])
cv_oof_rank5 = np.zeros(len(y_true))
for train_idx, val_idx in skf.split(X_rank5, y_true):
    val_pred, _ = optimize_weights_rank(X_rank5[train_idx], y_true[train_idx], X_rank5[val_idx])
    cv_oof_rank5[val_idx] = val_pred
print(f"\n10-Fold CV 5-Way Rank Blend AUC = {roc_auc_score(y_true, cv_oof_rank5):.6f}")

# Test F: Power / Rank-Logit Hybrid (CDF Calibration)
# Tilii noted CDF matching: transform RealMLP to match the empirical CDF / quantiles of Heuljax
print("\n--- Testing Empirical CDF Quantile Mapping (Tilii diversity hypothesis) ---")
from scipy.interpolate import interp1d
sorted_heuljax = np.sort(oofs['heuljax'])
quantiles = np.linspace(0, 1, len(sorted_heuljax))
# Map RealMLP probabilities into Heuljax probability scale via rank quantiles
mapped_realmlp = np.interp(rankdata(oofs['realmlp'])/len(y_true), quantiles, sorted_heuljax)
print(f"Mapped RealMLP AUC: {roc_auc_score(y_true, mapped_realmlp):.6f}")

# Blend mapped RealMLP with Heuljax & Blamerx
mapped_subset = [oofs['heuljax'], oofs['blamerx'], oofs['zoom_lgb'], mapped_realmlp]
X_mapped = np.column_stack([logit(np.clip(m, eps, 1-eps)) for m in mapped_subset])
cv_oof_mapped = np.zeros(len(y_true))
for train_idx, val_idx in skf.split(X_mapped, y_true):
    val_pred, _ = optimize_weights_logit(X_mapped[train_idx], y_true[train_idx], X_mapped[val_idx])
    cv_oof_mapped[val_idx] = val_pred
print(f"10-Fold CV Quantile-Mapped 4-Way Blend AUC = {roc_auc_score(y_true, cv_oof_mapped):.6f}")
