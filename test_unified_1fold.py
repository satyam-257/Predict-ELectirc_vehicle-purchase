"""
test_unified_1fold.py — Rapid 1-Fold Validation of Unified Features + GLM Margin Residual Model
Validates:
1. 1280-Quantile Neighbor Target Encoding with Gaussian kernel smoothing
2. All-column Decimal Digit Decomposition (-4 to 3)
3. 25-Key Triple Target Encoding (auto, 10, 100)
4. Domain Interaction features
5. Residual LightGBM with init_score and colsample_bytree=0.30, max_bin=1024
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

print("=== [TEST] Starting 1-Fold Unified Pipeline Validation ===")
t0 = time.time()

train = pd.read_csv('data/train.csv')
orig = pd.read_csv('EV_Adoption_and_Range_Anxiety_Dataset.csv').drop(columns=["Buyer_ID"]).dropna()
orig["_y"] = (orig["Will_Buy_EV"] == "Yes").astype(float)
oof_mine = pd.read_csv('kernels_out/lr_margin_gbdt/oof_mine.csv')

y = (train['Will_Buy_EV'] == 'Yes').astype(np.int8).to_numpy()
n_samples = len(train)

# GLM margin
def p_to_eta(p):
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    logit_p = np.log(p_clipped / (1.0 - p_clipped))
    return (logit_p + 1.0) / 1.25

margin = p_to_eta(oof_mine['glm'].values).astype(np.float64)

# 1-Fold Split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tr_idx, va_idx = next(skf.split(train, y))

print(f"Train samples: {len(tr_idx):,}, Val samples: {len(va_idx):,}")
print(f"GLM Val AUC alone: {roc_auc_score(y[va_idx], oof_mine['glm'].iloc[va_idx]):.6f}")

# Extract features on train vs val
def extract_unified_features(tr_df, va_df, y_tr):
    t_feat = time.time()
    n_tr = len(tr_df)
    combined = pd.concat([tr_df, va_df], ignore_index=True)
    
    raw_cats = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
    inc = combined["Annual_Income_USD"]
    com = combined["Daily_Commute_km"]
    age = combined["Age"]
    
    # 1. High-Resolution Categorical Keys
    combined["inc_exact"] = inc.astype(int).astype(str)
    combined["com_exact"] = com.round(1).astype(str)
    combined["age_key"] = age.astype(str)
    combined["env_key"] = combined["Environmental_Concern_Level"].astype(int).astype(str)
    combined["csh_key"] = combined["Charging_Stations_Near_Home"].astype(str)
    combined["csw_key"] = combined["Charging_Stations_Near_Work"].astype(str)
    
    # Interactions
    combined["inc_x_charge"] = (inc.astype(int).astype(str) + "_" + combined["Home_Charging_Possible"])
    combined["com_x_anx"] = (com.round(1).astype(str) + "_" + combined["Range_Anxiety_Level"])
    combined["city_x_csh"] = (combined["City_Type"] + "_" + combined["Charging_Stations_Near_Home"].astype(str))
    
    key_cols = raw_cats + ["inc_exact", "com_exact", "age_key", "env_key", "csh_key", "csw_key",
                          "inc_x_charge", "com_x_anx", "city_x_csh"]
    
    # Income & Commute powers
    for c in ["Annual_Income_USD", "Daily_Commute_km"]:
        for k in range(-1, 6):
            d = (combined[c] // (10 ** k) % 10).astype(int)
            if d.nunique() > 1:
                combined[f"{c[:3]}_d{k}"] = d.astype(str)
                key_cols.append(f"{c[:3]}_d{k}")
                
    combined["inc_100"] = (inc // 100).astype(int).astype(str)
    combined["inc_1000"] = (inc // 1000).astype(int).astype(str)
    combined["com_int"] = com.astype(int).astype(str)
    key_cols += ["inc_100", "inc_1000", "com_int"]
    
    # 2. Normalized Frequency Encodings
    fe = pd.DataFrame({f"{c}_fe": combined[c].map(combined[c].value_counts(normalize=True)).astype("float32") for c in key_cols})
    combined = pd.concat([combined, fe], axis=1)
    
    # 3. Original dataset priors
    orig_mean = orig["_y"].mean()
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Age", "Environmental_Concern_Level",
              "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"] + raw_cats:
        combined[f"{c}_orig_mean"] = combined[c].map(orig.groupby(c)["_y"].mean()).fillna(orig_mean).astype("float32")
        
    # 4. Domain & Physics Rules Flags
    combined["is_30k_spike"] = (inc == 30000).astype("int8")
    combined["is_millionaire_cliff"] = (inc >= 170537).astype("int8")
    combined["is_dead_zone"] = inc.between(38000, 42000).astype("int8")
    combined["is_commute_cutoff"] = (com >= 83).astype("int8")
    
    # 5. Income Quantile Target Encoding with 3-Tap Gaussian Kernel
    q = 1280
    fit_inc = tr_df['Annual_Income_USD'].values
    quantiles = np.linspace(0, 1, q + 1)
    edges = np.unique(np.quantile(fit_inc, quantiles))
    interior = edges[1:-1]
    
    # Bin index
    tr_bins = np.digitize(fit_inc, interior)
    comb_bins = np.digitize(combined['Annual_Income_USD'].values, interior)
    n_bins = len(interior) + 1
    
    # Compute sums and counts per bin on train fold
    bin_sums = np.bincount(tr_bins, weights=y_tr, minlength=n_bins)
    bin_counts = np.bincount(tr_bins, minlength=n_bins)
    
    prior = y_tr.mean()
    m_prior = 10.0
    
    # Gaussian kernel [0.4578, 1.0, 0.4578]
    k_gauss = np.exp(-0.5 * (np.array([-1, 0, 1]) / 0.8)**2)
    sum_k = np.sum(k_gauss)
    
    # Convolve
    padded_sums = np.pad(bin_sums, (1, 1), mode='edge')
    padded_counts = np.pad(bin_counts, (1, 1), mode='edge')
    
    smooth_sums = (padded_sums[:-2] * k_gauss[0] + padded_sums[1:-1] * k_gauss[1] + padded_sums[2:] * k_gauss[2])
    smooth_counts = (padded_counts[:-2] * k_gauss[0] + padded_counts[1:-1] * k_gauss[1] + padded_counts[2:] * k_gauss[2])
    
    qte_table = (smooth_sums + m_prior * sum_k * prior) / (smooth_counts + m_prior * sum_k)
    combined['inc_qte_neighbor'] = qte_table[comb_bins].astype(np.float32)
    combined['inc_qte_pos'] = (comb_bins / (n_bins - 1)).astype(np.float32)
    
    X_tr = combined.iloc[:n_tr].copy()
    X_va = combined.iloc[n_tr:].copy()
    
    # 6. Multi-Smooth Target Encoding
    for s in ("auto", 10.0, 100.0):
        te = TargetEncoder(cv=5, smooth=s, random_state=42)
        tag = "auto" if s == "auto" else str(int(s))
        enc_tr = te.fit_transform(X_tr[key_cols], y_tr)
        enc_va = te.transform(X_va[key_cols])
        for idx, col in enumerate(key_cols):
            X_tr[f"{col}_te{tag}"] = enc_tr[:, idx].astype("float32")
            X_va[f"{col}_te{tag}"] = enc_va[:, idx].astype("float32")
            
    # Drop raw string keys
    X_tr = X_tr.drop(columns=key_cols + ["id", "Will_Buy_EV"], errors="ignore")
    X_va = X_va.drop(columns=key_cols + ["id", "Will_Buy_EV"], errors="ignore")
    
    print(f"Features created: {X_tr.shape[1]} columns in {time.time()-t_feat:.1f}s")
    return X_tr, X_va

X_tr, X_va = extract_unified_features(train.iloc[tr_idx], train.iloc[va_idx], y[tr_idx])

# Train LightGBM Residual on GLM margin
clf_lgb = lgb.LGBMClassifier(
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=32,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.30,
    reg_alpha=0.071,
    reg_lambda=2.0,
    max_bin=1024,
    feature_pre_filter=False,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

m_tr, m_va = margin[tr_idx], margin[va_idx]

print("\n[1] Fitting LightGBM on GLM margin...")
t_fit = time.time()
clf_lgb.fit(
    X_tr, y[tr_idx],
    init_score=m_tr,
    eval_set=[(X_va, y[va_idx])],
    eval_init_score=[m_va],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(500, verbose=False)]
)

raw_va_lgb = clf_lgb.predict(X_va, raw_score=True)
val_pred_lgb = 1.0 / (1.0 + np.exp(-(raw_va_lgb + m_va)))
fold_auc_lgb = roc_auc_score(y[va_idx], val_pred_lgb)
print(f"---> LGBM FOLD 1 VAL AUC: {fold_auc_lgb:.6f} (Best Iter: {clf_lgb.best_iteration_}) in {time.time()-t_fit:.1f}s")

# Test XGBoost
from xgboost import XGBClassifier
print("\n[2] Fitting XGBoost on GLM margin...")
t_xgb = time.time()
clf_xgb = XGBClassifier(
    n_estimators=10000,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=5,
    subsample=0.9,
    colsample_bytree=0.5,
    reg_lambda=5.0,
    max_bin=1024,
    tree_method="hist",
    eval_metric="auc",
    early_stopping_rounds=500,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

clf_xgb.fit(
    X_tr, y[tr_idx],
    base_margin=m_tr.astype(np.float32),
    eval_set=[(X_va, y[va_idx])],
    base_margin_eval_set=[m_va.astype(np.float32)],
    verbose=False
)

val_pred_xgb = 1.0 / (1.0 + np.exp(-clf_xgb.predict(X_va, output_margin=True, base_margin=m_va.astype(np.float32))))
fold_auc_xgb = roc_auc_score(y[va_idx], val_pred_xgb)
print(f"---> XGB FOLD 1 VAL AUC:  {fold_auc_xgb:.6f} (Best Iter: {clf_xgb.best_iteration}) in {time.time()-t_xgb:.1f}s")

# Blend LGBM + XGB
blend_val = 0.5 * val_pred_lgb + 0.5 * val_pred_xgb
fold_auc_blend = roc_auc_score(y[va_idx], blend_val)
print(f"\n---> 50/50 BLEND FOLD 1 VAL AUC: {fold_auc_blend:.6f} (+{fold_auc_blend - roc_auc_score(y[va_idx], oof_mine['glm'].iloc[va_idx]):.6f} over GLM)")
