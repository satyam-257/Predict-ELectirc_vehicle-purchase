"""
Full pipeline: train all 3 models with unified features + fold-safe TE, then ensemble & submit.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from features import engineer_features
from target_encoder import add_fold_safe_target_encoding
import os
import subprocess

def run_full_pipeline():
    print("="*75)
    print("FULL PIPELINE: UNIFIED FEATURES + FOLD-SAFE TE + 3-MODEL ENSEMBLE")
    print("="*75)
    
    train_df, test_df, cat_cols = engineer_features()
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    test_ids = test_df['id'].values
    
    print(f"\nTotal Training Samples: {len(train_df):,}")
    print(f"Total Test Samples:     {len(test_df):,}")
    print(f"Total Feature Count:    {len(features)}")
    
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    te_target_cols = ['City_Subsidy_Interaction', 'Sub_Anx_Group', 'City_Car_Interaction']
    
    # ─────────────────────────────────────────────────────────
    # 1. LIGHTGBM
    # ─────────────────────────────────────────────────────────
    print("\n" + "#"*60)
    print("TRAINING LIGHTGBM...")
    print("#"*60)
    
    lgb_oof = np.zeros(len(train_df))
    lgb_test = np.zeros(len(test_df))
    lgb_fold_aucs = []
    
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
        X_tr = train_df.iloc[train_idx][features]
        y_tr = target[train_idx]
        X_va = train_df.iloc[val_idx][features]
        y_va = target[val_idx]
        X_te = test_df[features].copy()
        
        X_tr, X_va, X_te = add_fold_safe_target_encoding(
            X_tr, y_tr, X_va, X_te,
            te_cols=te_target_cols, smoothing=20.0, random_state=42 + fold
        )
        
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data, categorical_feature=cat_cols)
        
        model = lgb.train(
            lgb_params, train_data, num_boost_round=3000,
            valid_sets=[train_data, val_data], valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)]
        )
        
        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        lgb_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        lgb_fold_aucs.append(fold_auc)
        print(f"[LGBM] Fold {fold+1}/{n_splits} | AUC: {fold_auc:.6f} | Iter: {model.best_iteration}")
        lgb_test += model.predict(X_te, num_iteration=model.best_iteration) / n_splits
    
    lgb_auc = roc_auc_score(target, lgb_oof)
    print(f"\n--> LIGHTGBM OOF ROC-AUC: {lgb_auc:.6f}")
    
    # ─────────────────────────────────────────────────────────
    # 2. XGBOOST
    # ─────────────────────────────────────────────────────────
    print("\n" + "#"*60)
    print("TRAINING XGBOOST...")
    print("#"*60)
    
    xgb_oof = np.zeros(len(train_df))
    xgb_test = np.zeros(len(test_df))
    xgb_fold_aucs = []
    
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
        'enable_categorical': True,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        X_tr = train_df.iloc[train_idx][features]
        y_tr = target[train_idx]
        X_va = train_df.iloc[val_idx][features]
        y_va = target[val_idx]
        X_te = test_df[features].copy()
        
        X_tr, X_va, X_te = add_fold_safe_target_encoding(
            X_tr, y_tr, X_va, X_te,
            te_cols=te_target_cols, smoothing=20.0, random_state=42 + fold
        )
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        dtest = xgb.DMatrix(X_te, enable_categorical=True)
        
        model = xgb.train(
            xgb_params, dtrain, num_boost_round=3000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100, verbose_eval=False
        )
        
        val_pred = model.predict(dval)
        xgb_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        xgb_fold_aucs.append(fold_auc)
        print(f"[XGB]  Fold {fold+1}/{n_splits} | AUC: {fold_auc:.6f} | Iter: {model.best_iteration}")
        xgb_test += model.predict(dtest) / n_splits
    
    xgb_auc = roc_auc_score(target, xgb_oof)
    print(f"\n--> XGBOOST OOF ROC-AUC: {xgb_auc:.6f}")
    
    # ─────────────────────────────────────────────────────────
    # 3. CATBOOST
    # ─────────────────────────────────────────────────────────
    print("\n" + "#"*60)
    print("TRAINING CATBOOST...")
    print("#"*60)
    
    # Prepare string-typed copies for CatBoost
    cat_train_df = train_df.copy()
    cat_test_df = test_df.copy()
    for c in cat_cols:
        cat_train_df[c] = cat_train_df[c].astype(str)
        cat_test_df[c] = cat_test_df[c].astype(str)
    
    cat_oof = np.zeros(len(train_df))
    cat_test = np.zeros(len(test_df))
    cat_fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cat_train_df, target)):
        X_tr = cat_train_df.iloc[train_idx][features]
        y_tr = target[train_idx]
        X_va = cat_train_df.iloc[val_idx][features]
        y_va = target[val_idx]
        X_te = cat_test_df[features].copy()
        
        X_tr, X_va, X_te = add_fold_safe_target_encoding(
            X_tr, y_tr, X_va, X_te,
            te_cols=te_target_cols, smoothing=20.0, random_state=42 + fold
        )
        
        cat_features_in_data = [c for c in cat_cols if c in X_tr.columns]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_features_in_data)
        val_pool = Pool(X_va, y_va, cat_features=cat_features_in_data)
        test_pool = Pool(X_te, cat_features=cat_features_in_data)
        
        model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.06,
            depth=6,
            eval_metric='AUC',
            random_seed=42,
            task_type='CPU',
            thread_count=-1,
            early_stopping_rounds=100,
            verbose=False
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        val_pred = model.predict_proba(val_pool)[:, 1]
        cat_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        cat_fold_aucs.append(fold_auc)
        print(f"[CAT]  Fold {fold+1}/{n_splits} | AUC: {fold_auc:.6f}")
        cat_test += model.predict_proba(test_pool)[:, 1] / n_splits
    
    cat_auc = roc_auc_score(target, cat_oof)
    print(f"\n--> CATBOOST OOF ROC-AUC: {cat_auc:.6f}")
    
    # ─────────────────────────────────────────────────────────
    # 4. SAVE OOF & TEST PREDICTIONS
    # ─────────────────────────────────────────────────────────
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_lgbm.npy', lgb_oof)
    np.save('./preds/test_lgbm.npy', lgb_test)
    np.save('./preds/oof_xgb.npy', xgb_oof)
    np.save('./preds/test_xgb.npy', xgb_test)
    np.save('./preds/oof_catboost.npy', cat_oof)
    np.save('./preds/test_catboost.npy', cat_test)
    print("\nSaved all OOF and test predictions to ./preds/")
    
    # ─────────────────────────────────────────────────────────
    # 5. ENSEMBLE OPTIMIZATION
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("ENSEMBLE OPTIMIZATION")
    print("="*75)
    
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    oof_mat = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    test_mat = np.column_stack([lgb_test, xgb_test, cat_test])
    
    def objective(w):
        w = np.maximum(0, w)
        if w.sum() == 0: return 0
        w = w / w.sum()
        return -roc_auc_score(target, np.dot(oof_mat, w))
    
    res = minimize(objective, np.ones(3)/3, method='SLSQP',
                   bounds=[(0,1)]*3,
                   constraints={'type': 'eq', 'fun': lambda w: 1-sum(w)},
                   options={'maxiter': 500, 'ftol': 1e-7})
    
    best_w = res.x / res.x.sum()
    opt_auc = -res.fun
    
    for name, w in zip(models, best_w):
        print(f"Weight for {name:10s}: {w:.4f}")
    print(f"\nOPTIMAL PROBABILITY BLEND OOF AUC: {opt_auc:.6f}")
    
    # Equal + Optimal rank averaging
    rank_oof_opt = np.zeros_like(target, dtype=float)
    rank_test_opt = np.zeros(len(test_ids), dtype=float)
    rank_oof_eq = np.zeros_like(target, dtype=float)
    rank_test_eq = np.zeros(len(test_ids), dtype=float)
    
    for i, w in enumerate(best_w):
        r_oof = rankdata(oof_mat[:, i]) / len(target)
        r_test = rankdata(test_mat[:, i]) / len(test_ids)
        rank_oof_opt += w * r_oof
        rank_test_opt += w * r_test
        rank_oof_eq += r_oof / 3.0
        rank_test_eq += r_test / 3.0
    
    rank_opt_auc = roc_auc_score(target, rank_oof_opt)
    rank_eq_auc = roc_auc_score(target, rank_oof_eq)
    print(f"OPTIMAL RANK BLEND OOF AUC:  {rank_opt_auc:.6f}")
    print(f"EQUAL RANK BLEND OOF AUC:    {rank_eq_auc:.6f}")
    
    # ─────────────────────────────────────────────────────────
    # 6. GENERATE SUBMISSIONS
    # ─────────────────────────────────────────────────────────
    # Use best rank blend
    best_rank_auc = max(rank_opt_auc, rank_eq_auc)
    best_rank_test = rank_test_opt if rank_opt_auc >= rank_eq_auc else rank_test_eq
    best_rank_label = "optimal" if rank_opt_auc >= rank_eq_auc else "equal"
    
    sub_rank = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': best_rank_test})
    sub_rank.to_csv('submission_rank_ensemble.csv', index=False)
    sub_rank.to_csv('./preds/submission_rank_ensemble.csv', index=False)
    
    final_prob = np.dot(test_mat, best_w)
    sub_prob = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': final_prob})
    sub_prob.to_csv('submission_ensemble.csv', index=False)
    sub_prob.to_csv('./preds/submission_ensemble.csv', index=False)
    
    print(f"\nSaved submission_rank_ensemble.csv ({best_rank_label} weights, OOF: {best_rank_auc:.6f})")
    print(f"Saved submission_ensemble.csv (probability blend, OOF: {opt_auc:.6f})")
    
    # ─────────────────────────────────────────────────────────
    # 7. SUBMIT TO KAGGLE
    # ─────────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("SUBMITTING TO KAGGLE...")
    print("="*75)
    
    sub_file = 'submission_rank_ensemble.csv'
    msg = f'Unified features + fold-safe TE + 3-model {best_rank_label} rank blend (OOF: {best_rank_auc:.6f})'
    
    cmd = f'kaggle competitions submit -c playground-series-s6e9 -f "{sub_file}" -m "{msg}"'
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)
    
    print("\nFetching submission status...")
    status_cmd = 'kaggle competitions submissions -c playground-series-s6e9'
    res_status = subprocess.run(status_cmd, shell=True, capture_output=True, text=True)
    print(res_status.stdout[:2000])
    
    print("="*75)
    print("PIPELINE COMPLETE")
    print("="*75)

if __name__ == '__main__':
    run_full_pipeline()
