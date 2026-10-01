"""
Production 5-Fold Cross-Validation Implementation of the Kaggle Notebook:
'Zoom Zoom R Edition' by jazivxt (Playground Series S6E9 - EV Adoption Prediction)

Engineered Features (146 features):
1. 13 Categorical Mapped Inputs
2. 104 Decimal Digit Decompositions (10^-4 to 10^3 for all 13 columns)
3. 13 Normalized Frequency Encoded Columns
4. 13 Inner-KFold Smoothed Target Encoded Columns (prior smoothing m=10)
5. 3 Income Quantile Target Encodings (q=1280, position, center, and Gaussian neighbor convolution)

Hyperparameters & Training:
- LightGBM with max_depth=5, num_leaves=247, min_data_in_leaf=10
- feature_fraction=0.3029 (aggressive decorrelation)
- bagging_fraction=0.8128, bagging_freq=1
- lambda_l1=0.0709, lambda_l2=2.033, max_bin=1024, lr=0.02
"""

import time
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 51
N_FOLDS = 5
INNER_K = 3
TARGET = 'Will_Buy_EV'
ID_COL = 'id'

print(f"=== Starting 5-Fold Zoom Zoom LightGBM Pipeline (Seed={SEED}) ===", flush=True)

t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

y = (train_df[TARGET] == 'Yes').astype(int).values if train_df[TARGET].dtype == object else train_df[TARGET].values.astype(int)
test_ids = test_df[ID_COL].values

cat_maps = {
    'Gender': {'Female': 0, 'Male': 1, 'Other': 2},
    'City_Type': {'Rural': 0, 'Suburban': 1, 'Urban': 2},
    'Current_Car_Type': {'Hatchback': 0, 'Sedan': 1, 'SUV': 2, 'Truck': 3},
    'Home_Charging_Possible': {'No': 0, 'Yes': 1},
    'Subsidy_Available': {'No': 0, 'Yes': 1},
    'Range_Anxiety_Level': {'High': 0, 'Low': 1, 'Medium': 2}
}

predictors = [c for c in train_df.columns if c not in (ID_COL, TARGET)]

def map_categories(df):
    out = df[predictors].copy()
    for col, m in cat_maps.items():
        out[col] = out[col].map(m).fillna(-1).astype(int)
    return out

def get_strong_local_features(train_frame, train_y, eval_frame, inner_k=3, seed=42):
    raw_train = map_categories(train_frame)
    raw_eval = map_categories(eval_frame)
    
    feats_train = {}
    feats_eval = {}
    
    # 1. Raw Mapped + Digit Decompositions + Frequency Encoding
    for col in predictors:
        feats_train[col] = raw_train[col].values.astype(np.float32)
        feats_eval[col] = raw_eval[col].values.astype(np.float32)
        
        a = raw_train[col].values.astype(np.float64)
        b = raw_eval[col].values.astype(np.float64)
        
        for power in range(-4, 4):
            scale = 10.0 ** power
            nm = f'{col}_digit{power}'
            feats_train[nm] = (np.floor(np.round(a / scale, 6)) % 10).astype(np.float32)
            feats_eval[nm] = (np.floor(np.round(b / scale, 6)) % 10).astype(np.float32)
            
        ka = train_frame[col].astype(str)
        kb = eval_frame[col].astype(str)
        freq_map = ka.value_counts(normalize=True)
        feats_train[f'{col}_freq'] = ka.map(freq_map).fillna(0.0).values.astype(np.float32)
        feats_eval[f'{col}_freq'] = kb.map(freq_map).fillna(0.0).values.astype(np.float32)
        
    # 2. Inner-KFold Stratified Target Encoding
    skf = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    inner_folds = list(skf.split(train_frame, train_y))
    
    for col in predictors:
        ka = train_frame[col].astype(str).values
        kb = eval_frame[col].astype(str).values
        te_train = np.empty(len(train_y), dtype=np.float32)
        
        prior_full = train_y.mean()
        for fit_idx, held_idx in inner_folds:
            fit_series = pd.Series(train_y[fit_idx], index=ka[fit_idx])
            grp = fit_series.groupby(level=0)
            sums = grp.sum()
            counts = grp.count()
            prior_fold = train_y[fit_idx].mean()
            rates = ((sums + 10.0 * prior_fold) / (counts + 10.0)).to_dict()
            held_vals = ka[held_idx]
            te_train[held_idx] = [rates.get(v, prior_fold) for v in held_vals]
            
        full_series = pd.Series(train_y, index=ka)
        grp_full = full_series.groupby(level=0)
        sums_full = grp_full.sum()
        counts_full = grp_full.count()
        rates_full = ((sums_full + 10.0 * prior_full) / (counts_full + 10.0)).to_dict()
        feats_train[f'{col}_te'] = te_train
        feats_eval[f'{col}_te'] = np.array([rates_full.get(v, prior_full) for v in kb], dtype=np.float32)
        
    # 3. Income Quantile Target Encoding (1280 quantiles + 3-tap Gaussian Kernel)
    q = 1280
    fit_income = train_frame['Annual_Income_USD'].values
    new_income = eval_frame['Annual_Income_USD'].values
    quantiles = np.linspace(0, 1, q + 1)
    edges = np.unique(np.quantile(fit_income, quantiles))
    interior = edges[1:-1]
    n_bins = len(edges)
    
    codes = np.searchsorted(interior, fit_income, side='left')
    new_codes = np.searchsorted(interior, new_income, side='left')
    
    position = (codes / max(q - 1, 1)).astype(np.float32)
    new_position = (new_codes / max(q - 1, 1)).astype(np.float32)
    
    kernel = np.exp(-0.5 * (np.array([-1, 0, 1]) / 0.8)**2)
    k0, k1, k2 = kernel[0], kernel[1], kernel[2]
    kernel_sum = kernel.sum()
    
    central_oof = np.full(len(train_y), train_y.mean(), dtype=np.float32)
    neighbor_oof = np.full(len(train_y), train_y.mean(), dtype=np.float32)
    
    for fit_idx, held_idx in inner_folds:
        f_codes = codes[fit_idx]
        f_y = train_y[fit_idx]
        prior_f = f_y.mean()
        
        sums = np.bincount(f_codes, weights=f_y, minlength=n_bins)
        counts = np.bincount(f_codes, minlength=n_bins)
        
        center = (sums + 10.0 * prior_f) / (counts + 10.0)
        
        ns = np.zeros_like(sums)
        ns[1:] += k0 * sums[:-1]
        ns += k1 * sums
        ns[:-1] += k2 * sums[1:]
        
        nc = np.zeros_like(counts, dtype=float)
        nc[1:] += k0 * counts[:-1]
        nc += k1 * counts
        nc[:-1] += k2 * counts[1:]
        
        neighbor = (ns + 10.0 * kernel_sum * prior_f) / (nc + 10.0 * kernel_sum)
        
        h_codes = codes[held_idx]
        central_oof[held_idx] = center[h_codes]
        neighbor_oof[held_idx] = neighbor[h_codes]
        
    full_prior = train_y.mean()
    sums_full = np.bincount(codes, weights=train_y, minlength=n_bins)
    counts_full = np.bincount(codes, minlength=n_bins)
    center_full = (sums_full + 10.0 * full_prior) / (counts_full + 10.0)
    
    ns_full = np.zeros_like(sums_full)
    ns_full[1:] += k0 * sums_full[:-1]
    ns_full += k1 * sums_full
    ns_full[:-1] += k2 * sums_full[1:]
    
    nc_full = np.zeros_like(counts_full, dtype=float)
    nc_full[1:] += k0 * counts_full[:-1]
    nc_full += k1 * counts_full
    nc_full[:-1] += k2 * counts_full[1:]
    
    neighbor_full = (ns_full + 10.0 * kernel_sum * full_prior) / (nc_full + 10.0 * kernel_sum)
    
    feats_train['income_q1280_pos'] = position
    feats_train['income_q1280_center'] = central_oof
    feats_train['income_q1280_neighbor'] = neighbor_oof
    
    feats_eval['income_q1280_pos'] = new_position
    feats_eval['income_q1280_center'] = center_full[new_codes].astype(np.float32)
    feats_eval['income_q1280_neighbor'] = neighbor_full[new_codes].astype(np.float32)
    
    return pd.DataFrame(feats_train), pd.DataFrame(feats_eval)

# Pre-extract full test features against 100% train data for test prediction
print("Generating full test features against train set...", flush=True)
t_test_f = time.time()
_, df_test_full = get_strong_local_features(train_df, y, test_df, inner_k=INNER_K, seed=SEED)
print(f"Full test features generated in {time.time()-t_test_f:.2f}s. Shape: {df_test_full.shape}", flush=True)

# LightGBM Parameters from Zoom Zoom
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02,
    'max_depth': 5,
    'num_leaves': 247,
    'min_data_in_leaf': 10,
    'feature_fraction': 0.3029300829885024,
    'bagging_fraction': 0.812763123433567,
    'bagging_freq': 1,
    'lambda_l1': 0.07094285437903122,
    'lambda_l2': 2.0330390977032425,
    'max_bin': 1024,
    'feature_pre_filter': False,
    'verbosity': -1,
    'seed': 60,
    'n_jobs': -1
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(y), dtype=np.float64)
test_preds = np.zeros(len(test_df), dtype=np.float64)
fold_scores = []

for fold, (fit_idx, val_idx) in enumerate(skf.split(train_df, y)):
    f_start = time.time()
    print(f"\n--- Fold {fold+1}/{N_FOLDS} (Train={len(fit_idx)}, Val={len(val_idx)}) ---", flush=True)
    
    train_fold = train_df.iloc[fit_idx].reset_index(drop=True)
    val_fold = train_df.iloc[val_idx].reset_index(drop=True)
    y_fit = y[fit_idx]
    y_val = y[val_idx]
    
    x_fit, x_val = get_strong_local_features(train_fold, y_fit, val_fold, inner_k=INNER_K, seed=SEED + fold + 1)
    
    d_fit = lgb.Dataset(x_fit, label=y_fit)
    d_val = lgb.Dataset(x_val, label=y_val, reference=d_fit)
    
    model = lgb.train(
        lgb_params,
        d_fit,
        num_boost_round=10000,
        valid_sets=[d_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=500, verbose=False),
            lgb.log_evaluation(period=400)
        ]
    )
    
    val_pred = model.predict(x_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    fold_auc = roc_auc_score(y_val, val_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold+1} Best Iteration: {model.best_iteration} | AUC: {fold_auc:.6f} | Elapsed: {time.time()-f_start:.1f}s", flush=True)
    
    # Predict on test
    test_preds += model.predict(df_test_full, num_iteration=model.best_iteration) / N_FOLDS

overall_auc = roc_auc_score(y, oof_preds)
print(f"\n{'='*55}", flush=True)
print(f"POOLED 5-FOLD CV AUC: {overall_auc:.6f} (Mean Fold: {np.mean(fold_scores):.6f})", flush=True)
print(f"Total time: {time.time()-t0:.1f}s", flush=True)
print(f"{'='*55}", flush=True)

# Save artifacts
os.makedirs('preds', exist_ok=True)
np.save('preds/oof_zoom_zoom_lgb.npy', oof_preds)
np.save('preds/test_zoom_zoom_lgb.npy', test_preds)
print("Saved preds/oof_zoom_zoom_lgb.npy and preds/test_zoom_zoom_lgb.npy", flush=True)

# Save standalone submission
sub = pd.DataFrame({ID_COL: test_ids, TARGET: test_preds})
sub.to_csv('submission_zoom_zoom_lgb.csv', index=False)
print("Saved submission_zoom_zoom_lgb.csv", flush=True)
