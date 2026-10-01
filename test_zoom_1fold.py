import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

t0 = time.time()
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

TARGET = 'Will_Buy_EV'
y = (train_df[TARGET] == 'Yes').astype(int).values if train_df[TARGET].dtype == object else train_df[TARGET].values.astype(int)

cat_maps = {
    'Gender': {'Female': 0, 'Male': 1, 'Other': 2},
    'City_Type': {'Rural': 0, 'Suburban': 1, 'Urban': 2},
    'Current_Car_Type': {'Hatchback': 0, 'Sedan': 1, 'SUV': 2, 'Truck': 3},
    'Home_Charging_Possible': {'No': 0, 'Yes': 1},
    'Subsidy_Available': {'No': 0, 'Yes': 1},
    'Range_Anxiety_Level': {'High': 0, 'Low': 1, 'Medium': 2}
}

predictors = [c for c in train_df.columns if c not in ('id', TARGET)]

def map_categories(df):
    out = df[predictors].copy()
    for col, m in cat_maps.items():
        out[col] = out[col].map(m).fillna(-1).astype(int)
    return out

def get_strong_local_features(train_frame, train_y, valid_frame, inner_k=3, seed=42):
    raw_train = map_categories(train_frame)
    raw_valid = map_categories(valid_frame)
    
    feats_train = {}
    feats_valid = {}
    
    for col in predictors:
        feats_train[col] = raw_train[col].values.astype(np.float32)
        feats_valid[col] = raw_valid[col].values.astype(np.float32)
        
        a = raw_train[col].values.astype(np.float64)
        b = raw_valid[col].values.astype(np.float64)
        
        for power in range(-4, 4):
            scale = 10.0 ** power
            nm = f'{col}_digit{power}'
            feats_train[nm] = (np.floor(np.round(a / scale, 6)) % 10).astype(np.float32)
            feats_valid[nm] = (np.floor(np.round(b / scale, 6)) % 10).astype(np.float32)
            
        ka = train_frame[col].astype(str)
        kb = valid_frame[col].astype(str)
        freq_map = ka.value_counts(normalize=True)
        feats_train[f'{col}_freq'] = ka.map(freq_map).fillna(0.0).values.astype(np.float32)
        feats_valid[f'{col}_freq'] = kb.map(freq_map).fillna(0.0).values.astype(np.float32)
        
    skf = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    inner_folds = list(skf.split(train_frame, train_y))
    
    for col in predictors:
        ka = train_frame[col].astype(str).values
        kb = valid_frame[col].astype(str).values
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
        feats_valid[f'{col}_te'] = np.array([rates_full.get(v, prior_full) for v in kb], dtype=np.float32)
        
    # Income QTE
    q = 1280
    fit_income = train_frame['Annual_Income_USD'].values
    new_income = valid_frame['Annual_Income_USD'].values
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
    
    feats_valid['income_q1280_pos'] = new_position
    feats_valid['income_q1280_center'] = center_full[new_codes].astype(np.float32)
    feats_valid['income_q1280_neighbor'] = neighbor_full[new_codes].astype(np.float32)
    
    return pd.DataFrame(feats_train), pd.DataFrame(feats_valid)

# Seed 51 5-fold split (same as R notebook)
SEED = 51
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(skf.split(train_df, y))
fit_idx, valid_idx = folds[0][0], folds[0][1]
# Note: folds[0][1] is validation fold 1

print(f'Train fold: {len(fit_idx)}, Val fold: {len(valid_idx)}')
train_fold = train_df.iloc[fit_idx].reset_index(drop=True)
val_fold = train_df.iloc[valid_idx].reset_index(drop=True)
y_fit = y[fit_idx]
y_val = y[valid_idx]

print('Extracting features...')
t_f0 = time.time()
x_fit, x_val = get_strong_local_features(train_fold, y_fit, val_fold, inner_k=3, seed=SEED+1)
print(f'Features extracted in {time.time()-t_f0:.2f}s. Shape: {x_fit.shape}')

params = {
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

d_fit = lgb.Dataset(x_fit, label=y_fit)
d_val = lgb.Dataset(x_val, label=y_val, reference=d_fit)

print('Training LightGBM...')
t_tr = time.time()
evals_result = {}
model = lgb.train(
    params,
    d_fit,
    num_boost_round=10000,
    valid_sets=[d_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=500, verbose=100),
        lgb.log_evaluation(period=200),
        lgb.record_evaluation(evals_result)
    ]
)

val_preds = model.predict(x_val, num_iteration=model.best_iteration)
val_auc = roc_auc_score(y_val, val_preds)
print(f'Done in {time.time()-t_tr:.2f}s! Best Iteration: {model.best_iteration}, Val AUC: {val_auc:.6f}')
