"""
train_generator_xgb.py — Train 5-Fold XGBoost with Ground-Truth Generator Features & Submit Ensemble
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features_v2 import engineer_features_v2
from scipy.stats import rankdata
import subprocess

def train_and_submit_xgb():
    print("="*75)
    print("S6E9: XGBOOST WITH GROUND-TRUTH GENERATOR FEATURES (5-FOLD CV)")
    print("="*75)
    
    train_df, test_df, cat_cols = engineer_features_v2()
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    test_ids = test_df['id'].values
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_aucs = []
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    dtest = xgb.DMatrix(test_df[features], enable_categorical=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        X_tr, y_tr = train_df.iloc[train_idx][features], target[train_idx]
        X_va, y_va = train_df.iloc[val_idx][features], target[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=1500,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=200
        )
        
        val_pred = model.predict(dval)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_va, val_pred)
        fold_aucs.append(fold_auc)
        print(f"[XGB] Fold {fold + 1} ROC-AUC: {fold_auc:.6f} (Best Iter: {model.best_iteration})")
        
        test_preds += model.predict(dtest) / n_splits
        
    overall_auc = roc_auc_score(target, oof_preds)
    print("\n" + "="*75)
    print(f"XGB OVERALL 5-FOLD OOF ROC-AUC: {overall_auc:.6f}")
    print(f"Fold AUCs: {[round(s, 6) for s in fold_aucs]}")
    print("="*75)
    
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_gen_xgb.npy', oof_preds)
    np.save('./preds/test_gen_xgb.npy', test_preds)
    print("Saved ./preds/oof_gen_xgb.npy and ./preds/test_gen_xgb.npy")
    
    # 3. Create Multi-Model GBDT Ensemble: LightGBM + XGBoost with Generator Features
    if os.path.exists('./preds/test_gen_lgbm.npy') and os.path.exists('./preds/oof_gen_lgbm.npy'):
        oof_lgb = np.load('./preds/oof_gen_lgbm.npy')
        test_lgb = np.load('./preds/test_gen_lgbm.npy')
        
        r_lgb_oof = rankdata(oof_lgb) / len(oof_lgb)
        r_xgb_oof = rankdata(oof_preds) / len(oof_preds)
        
        # Optimize blend weight
        best_w, best_auc = 0.5, 0.0
        for w in np.linspace(0.2, 0.8, 13):
            b_auc = roc_auc_score(target, w * r_lgb_oof + (1 - w) * r_xgb_oof)
            if b_auc > best_auc:
                best_auc = b_auc
                best_w = w
        print(f"\n[ENSEMBLE] Optimal Weights: {best_w:.2f} LGB + {1-best_w:.2f} XGB | Blended OOF AUC: {best_auc:.6f}")
        
        r_lgb_test = rankdata(test_lgb) / len(test_lgb)
        r_xgb_test = rankdata(test_preds) / len(test_preds)
        blend_test = best_w * r_lgb_test + (1 - best_w) * r_xgb_test
        
        ens_file = 'submission_generator_ensemble_lgb_xgb.csv'
        pd.DataFrame({'id': test_ids, 'Will_Buy_EV': blend_test}).to_csv(ens_file, index=False)
        print(f"Saved {ens_file}")
        
        # Submit the ensemble
        print("\nSubmitting Ensemble to Kaggle...")
        os.environ['KAGGLE_API_TOKEN'] = 'KGAT_890ba420602fcb8048bfe87686d1ed04'
        msg = f"Generator Ensemble: LGBM ({best_w:.2f}) + XGB ({1-best_w:.2f}) (OOF={best_auc:.6f})"
        cmd = f'kaggle competitions submit -c playground-series-s6e9 -f "{ens_file}" -m "{msg}"'
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        
        # Check Kaggle submissions
        status_cmd = 'kaggle competitions submissions -c playground-series-s6e9'
        res_status = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
        print(res_status.stdout)

if __name__ == '__main__':
    train_and_submit_xgb()
