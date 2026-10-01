"""
train_generator_lgbm.py — Train 5-Fold LightGBM with Ground-Truth Generator Features & Submit to Kaggle.
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features_v2 import engineer_features_v2
import subprocess

def train_and_submit():
    print("="*75)
    print("S6E9: LIGHTGBM WITH GROUND-TRUTH GENERATOR FEATURES (5-FOLD CV)")
    print("="*75)
    
    # 1. Feature Engineering
    train_df, test_df, cat_cols = engineer_features_v2()
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    test_ids = test_df['id'].values
    
    print(f"\nTraining Samples: {len(train_df):,}")
    print(f"Testing Samples:  {len(test_df):,}")
    print(f"Features:         {len(features)}")
    print(f"Categoricals:     {cat_cols}")
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_aucs = []
    
    # Tuned hyperparameters for large synthetic tabular data with exact formula features
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': -1,
        'min_child_samples': 50,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        X_tr, y_tr = train_df.iloc[train_idx][features], target[train_idx]
        X_va, y_va = train_df.iloc[val_idx][features], target[val_idx]
        X_te = test_df[features].copy()
        
        trn_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data, categorical_feature=cat_cols)
        
        model = lgb.train(
            lgb_params,
            trn_data,
            num_boost_round=2000,
            valid_sets=[trn_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)]
        )
        
        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(fold_auc)
        print(f"[LGBM] Fold {fold + 1} ROC-AUC: {fold_auc:.6f} (Best Iter: {model.best_iteration})")
        
        test_preds += model.predict(X_te, num_iteration=model.best_iteration) / n_splits
        
    overall_auc = roc_auc_score(target, oof_preds)
    print("\n" + "="*75)
    print(f"OVERALL 5-FOLD OOF ROC-AUC: {overall_auc:.6f}")
    print(f"Fold AUCs: {[round(s, 6) for s in fold_aucs]}")
    print("="*75)
    
    # 2. Save Out-Of-Fold and Test Predictions
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_gen_lgbm.npy', oof_preds)
    np.save('./preds/test_gen_lgbm.npy', test_preds)
    print("Saved ./preds/oof_gen_lgbm.npy and ./preds/test_gen_lgbm.npy")
    
    # 3. Save Submission File
    sub_file = 'submission_generator_lgbm.csv'
    sub_df = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': test_preds
    })
    sub_df.to_csv(sub_file, index=False)
    print(f"Saved submission file to {sub_file}")
    
    # 4. Also create an optimal ensemble with existing best stacker (if available)
    if os.path.exists('./preds/oof_ensemble_v3.npy') and os.path.exists('./preds/submission_v3_best.csv'):
        print("\nCreating hybrid blend with previous best stacker (v3)...")
        v3_sub = pd.read_csv('./preds/submission_v3_best.csv')
        # Optimal rank-average blend between new generator-boosted model and v3
        from scipy.stats import rankdata
        r_gen = rankdata(test_preds) / len(test_preds)
        r_v3 = rankdata(v3_sub['Will_Buy_EV'].values) / len(v3_sub)
        hybrid_preds = 0.60 * r_gen + 0.40 * r_v3
        
        hybrid_file = 'submission_generator_hybrid.csv'
        sub_hybrid = pd.DataFrame({
            'id': test_ids,
            'Will_Buy_EV': hybrid_preds
        })
        sub_hybrid.to_csv(hybrid_file, index=False)
        print(f"Saved hybrid submission to {hybrid_file}")
    
    # 5. Submit to Kaggle
    print("\nSubmitting to Kaggle...")
    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_890ba420602fcb8048bfe87686d1ed04'
    msg = f"LightGBM with Ground-Truth Generator Features (OOF={overall_auc:.6f})"
    cmd = f'kaggle competitions submit -c playground-series-s6e9 -f "{sub_file}" -m "{msg}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    # Fetch status
    print("\nChecking submission status on Kaggle...")
    status_cmd = 'kaggle competitions submissions -c playground-series-s6e9'
    res_status = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
    print(res_status.stdout)

if __name__ == '__main__':
    train_and_submit()
