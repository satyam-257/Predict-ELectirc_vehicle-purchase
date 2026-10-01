"""
train_glm_margin_gbdt.py — Full 25-Key Triple Target-Encoded GLM-Margin GBDT
Option C Implementation:
Boosts LightGBM on top of the 344-feature GLM logit margin (init_score),
using 25 distinct categorical & numerical keys target-encoded at 3 smoothing levels
(auto, 10, 100), frequency encodings, and original data buyer rates.
"""

import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

print("=" * 80)
print(" S6E9: FULL 25-KEY TRIPLE-TE GLM-MARGIN GBDT (OPTION C)")
print("=" * 80)

SEED = 42
FOLDS = 5
LR = 0.02
TARGET, ID = "Will_Buy_EV", "id"

# 1. Load Data
print("\n[1/5] Loading datasets...")
t0 = time.time()
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
orig = pd.read_csv('EV_Adoption_and_Range_Anxiety_Dataset.csv').drop(columns=["Buyer_ID"]).dropna()

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
orig["_y"] = (orig[TARGET] == "Yes").astype(float)
ntr, nte = len(train), len(test)

print(f"Loaded train ({ntr:,}) and test ({nte:,}) in {time.time()-t0:.1f}s")

# 2. Load GLM Margin Basis
print("\n[2/5] Loading precomputed GLM logit margins...")
oof_mine = pd.read_csv('kernels_out/lr_margin_gbdt/oof_mine.csv')
test_mine = pd.read_csv('kernels_out/lr_margin_gbdt/test_mine.csv')

def p_to_eta(p):
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    logit_p = np.log(p_clipped / (1.0 - p_clipped))
    return (logit_p + 1.0) / 1.25

margin_train = p_to_eta(oof_mine['glm'].values).astype(np.float64)
margin_test = p_to_eta(test_mine['glm'].values).astype(np.float64)

# 3. Build 25 Keys & Feature Representation
print("\n[3/5] Constructing 25 high-resolution keys & frequency signals...")
t1 = time.time()

def build_features(train, test, orig):
    n_train = len(train)
    df = pd.concat([train, test], ignore_index=True).drop(columns=["Number_of_Cars_Owned"])
    raw_cats = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
    inc, com = df["Annual_Income_USD"], df["Daily_Commute_km"]
    df["inc_exact"] = inc.astype(int).astype(str)
    df["com_exact"] = com.round(1).astype(str)
    df["age_key"] = df["Age"].astype(str)
    df["env_key"] = df["Environmental_Concern_Level"].astype(int).astype(str)
    df["csh_key"] = df["Charging_Stations_Near_Home"].astype(str)
    df["csw_key"] = df["Charging_Stations_Near_Work"].astype(str)
    key_cols = raw_cats + ["inc_exact", "com_exact", "age_key", "env_key", "csh_key", "csw_key"]
    
    for c in ["Annual_Income_USD", "Daily_Commute_km"]:
        for k in range(-1, 6):
            d = (df[c] // (10 ** k) % 10).astype(int)
            if d.nunique() > 1:
                df[f"{c[:3]}_d{k}"] = d.astype(str)
                key_cols.append(f"{c[:3]}_d{k}")
                
    df["inc_100"] = (inc // 100).astype(int).astype(str)
    df["inc_1000"] = (inc // 1000).astype(int).astype(str)
    df["com_int"] = com.astype(int).astype(str)
    key_cols += ["inc_100", "inc_1000", "com_int"]
    
    fe = pd.DataFrame({f"{c}_fe": df[c].map(df[c].value_counts(normalize=True)).astype("float32") for c in key_cols})
    df = pd.concat([df, fe], axis=1)
    
    orig_mean = orig["_y"].mean()
    for c in ["Annual_Income_USD", "Daily_Commute_km", "Age", "Environmental_Concern_Level",
              "Charging_Stations_Near_Home", "Charging_Stations_Near_Work"] + raw_cats:
        df[f"{c}_orig_mean"] = df[c].map(orig.groupby(c)["_y"].mean()).fillna(orig_mean).astype("float32")
        
    df["is_30k_spike"] = (inc == 30000).astype("int8")
    df["is_millionaire_cliff"] = (inc >= 170537).astype("int8")
    df["is_dead_zone"] = inc.between(38000, 42000).astype("int8")
    
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy(), key_cols

TE_SMOOTHS = ("auto", 10.0, 100.0)

def target_encode(X_tr, y_tr, X_others, key_cols):
    outs = [X_tr.copy()] + [X.copy() for X in X_others]
    for s in TE_SMOOTHS:
        te = TargetEncoder(cv=FOLDS, smooth=s, random_state=SEED)
        enc = [te.fit_transform(X_tr[key_cols], y_tr)] + [te.transform(X[key_cols]) for X in X_others]
        tag = "auto" if s == "auto" else str(int(s))
        for Xo, e in zip(outs, enc):
            Xo[[f"{c}_te{tag}" for c in key_cols]] = e.astype("float32")
    return [Xo.drop(columns=key_cols) for Xo in outs]

tr_f, te_f, key_cols = build_features(train.drop(columns=[TARGET]), test, orig)
features = [c for c in te_f.columns if c != ID]
XB, XB_test = tr_f[features], te_f[features]
print(f"Constructed {len(features)} columns before TE, {len(key_cols)} keys x {len(TE_SMOOTHS)} smoothings in {time.time()-t1:.1f}s")

# 4. Train 5-Fold Residual LightGBM with init_score
print("\n[4/5] Training 5-Fold LightGBM initialized from GLM Margin...")
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
splits = list(skf.split(np.zeros(ntr), y))

gbdt_oof = np.zeros(ntr, dtype=np.float64)
gbdt_test = np.zeros(nte, dtype=np.float64)

for k, (tr_idx, va_idx) in enumerate(splits):
    t_fold = time.time()
    m_tr, m_va, m_te = margin_train[tr_idx], margin_train[va_idx], margin_test
    
    t_te = time.time()
    X_tr, X_va, X_te = target_encode(XB.iloc[tr_idx], y[tr_idx], [XB.iloc[va_idx], XB_test], key_cols)
    te_elapsed = time.time() - t_te
    
    clf = lgb.LGBMClassifier(
        n_estimators=10000,
        learning_rate=LR,
        max_depth=5,
        num_leaves=32,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.3,
        reg_alpha=0.071,
        reg_lambda=2.0,
        max_bin=1024,
        feature_pre_filter=False,
        random_state=SEED + k,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(
        X_tr, y[tr_idx],
        init_score=m_tr,
        eval_set=[(X_va, y[va_idx])],
        eval_init_score=[m_va],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(500, verbose=False)]
    )
    
    raw_va = clf.predict(X_va, raw_score=True)
    val_pred = 1.0 / (1.0 + np.exp(-(raw_va + m_va)))
    gbdt_oof[va_idx] = val_pred
    
    raw_te = clf.predict(X_te, raw_score=True)
    test_fold_pred = 1.0 / (1.0 + np.exp(-(raw_te + m_te)))
    gbdt_test += test_fold_pred / FOLDS
    
    fold_auc = roc_auc_score(y[va_idx], val_pred)
    trees = clf.best_iteration_
    print(f"  Fold {k+1}/{FOLDS} — Trees: {trees:5d} | Val AUC: {fold_auc:.6f} | Elapsed: {time.time()-t_fold:.1f}s (TE: {te_elapsed:.1f}s)", flush=True)

overall_auc = roc_auc_score(y, gbdt_oof)
print("\n" + "=" * 80)
print(f"5-FOLD RESIDUAL LIGHTGBM POOLED OOF AUC: {overall_auc:.6f} (GLM alone: {roc_auc_score(y, oof_mine['glm']):.6f})")
print("=" * 80)

# Save predictions
np.save('preds/oof_glm_margin_triple_te_lgbm.npy', gbdt_oof)
np.save('preds/test_glm_margin_triple_te_lgbm.npy', gbdt_test)

sub_path = 'submission_glm_margin_triple_te_lgbm.csv'
pd.DataFrame({'id': test[ID].values, 'Will_Buy_EV': gbdt_test}).to_csv(sub_path, index=False)
print(f"Successfully saved {sub_path} and numpy arrays in preds/")
