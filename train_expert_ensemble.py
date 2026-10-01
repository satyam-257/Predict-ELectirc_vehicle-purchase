import numpy as np
import pandas as pd
import os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from features_v2 import engineer_features_v2

def train_and_ensemble(n_splits=10):
    print("="*75)
    print(f"EXPERT PIPELINE: {n_splits}-FOLD ENSEMBLE (LIGHTGBM + XGBOOST + CATBOOST)")
    print("INSIGHTS APPLIED: FABLE 5.1 RECIPE + GENERATOR DIGIT ARTIFACTS + FREQUENCIES")
    print("="*75)
    
    train_df, test_df, cat_cols = engineer_features_v2()
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    test_ids = test_df['id'].values
    
    print(f"\nTotal Training Samples: {len(train_df):,}")
    print(f"Total Test Samples:     {len(test_df):,}")
    print(f"Total Feature Count:    {len(features)}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # -------------------------------------------------------------
    # 1. LIGHTGBM 10-FOLD
    # -------------------------------------------------------------
    print("\n" + "#"*60)
    print("STARTING LIGHTGBM TRAINING...")
    print("#"*60)
    
    lgb_oof = np.zeros(len(train_df))
    lgb_test = np.zeros(len(test_df))
    lgb_fold_aucs = []
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.04,
        'num_leaves': 127,
        'min_child_samples': 40,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 1.5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        X_tr, y_tr = train_df.iloc[train_idx][features], target[train_idx]
        X_va, y_va = train_df.iloc[val_idx][features], target[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data, categorical_feature=cat_cols)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=200)
        ]
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=3000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        val_pred = model.predict(X_va, num_iteration=model.best_iteration)
        lgb_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        lgb_fold_aucs.append(fold_auc)
        print(f"[LGBM] Fold {fold+1:02d}/{n_splits:02d} | ROC-AUC: {fold_auc:.6f} | Best Iter: {model.best_iteration}")
        
        lgb_test += model.predict(test_df[features], num_iteration=model.best_iteration) / n_splits
        
    lgb_overall_auc = roc_auc_score(target, lgb_oof)
    print(f"\n--> LIGHTGBM OVERALL OOF ROC-AUC: {lgb_overall_auc:.6f}")
    
    # -------------------------------------------------------------
    # 2. XGBOOST 10-FOLD
    # -------------------------------------------------------------
    print("\n" + "#"*60)
    print("STARTING XGBOOST TRAINING...")
    print("#"*60)
    
    xgb_oof = np.zeros(len(train_df))
    xgb_test = np.zeros(len(test_df))
    xgb_fold_aucs = []
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'learning_rate': 0.04,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 30,
        'reg_alpha': 0.05,
        'reg_lambda': 1.5,
        'random_state': 42,
        'enable_categorical': True,
        'n_jobs': -1
    }
    
    dtest = xgb.DMatrix(test_df[features], enable_categorical=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        X_tr, y_tr = train_df.iloc[train_idx][features], target[train_idx]
        X_va, y_va = train_df.iloc[val_idx][features], target[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=3000,
            evals=evals,
            early_stopping_rounds=80,
            verbose_eval=False
        )
        
        val_pred = model.predict(dval)
        xgb_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        xgb_fold_aucs.append(fold_auc)
        print(f"[XGB]  Fold {fold+1:02d}/{n_splits:02d} | ROC-AUC: {fold_auc:.6f} | Best Iter: {model.best_iteration}")
        
        xgb_test += model.predict(dtest) / n_splits
        
    xgb_overall_auc = roc_auc_score(target, xgb_oof)
    print(f"\n--> XGBOOST OVERALL OOF ROC-AUC: {xgb_overall_auc:.6f}")
    
    # -------------------------------------------------------------
    # 3. CATBOOST 10-FOLD
    # -------------------------------------------------------------
    print("\n" + "#"*60)
    print("STARTING CATBOOST TRAINING...")
    print("#"*60)
    
    cat_train_df = train_df.copy()
    cat_test_df = test_df.copy()
    for c in cat_cols:
        cat_train_df[c] = cat_train_df[c].astype(str)
        cat_test_df[c] = cat_test_df[c].astype(str)
        
    cat_oof = np.zeros(len(train_df))
    cat_test = np.zeros(len(test_df))
    cat_fold_aucs = []
    
    cat_test_pool = Pool(cat_test_df[features], cat_features=cat_cols)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(cat_train_df, target)):
        X_tr, y_tr = cat_train_df.iloc[train_idx][features], target[train_idx]
        X_va, y_va = cat_train_df.iloc[val_idx][features], target[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        val_pool = Pool(X_va, y_va, cat_features=cat_cols)
        
        model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.05,
            depth=6,
            eval_metric='AUC',
            random_seed=42,
            task_type='CPU',
            thread_count=-1,
            early_stopping_rounds=80,
            verbose=False
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        val_pred = model.predict_proba(val_pool)[:, 1]
        cat_oof[val_idx] = val_pred
        fold_auc = roc_auc_score(y_va, val_pred)
        cat_fold_aucs.append(fold_auc)
        print(f"[CAT]  Fold {fold+1:02d}/{n_splits:02d} | ROC-AUC: {fold_auc:.6f}")
        
        cat_test += model.predict_proba(cat_test_pool)[:, 1] / n_splits
        
    cat_overall_auc = roc_auc_score(target, cat_oof)
    print(f"\n--> CATBOOST OVERALL OOF ROC-AUC: {cat_overall_auc:.6f}")
    
    # -------------------------------------------------------------
    # 4. ENSEMBLE OPTIMIZATION & SUBMISSION
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print("ENSEMBLE OPTIMIZATION (SLSQP MAXIMIZING ROC-AUC)")
    print("="*75)
    
    models = ['LightGBM', 'XGBoost', 'CatBoost']
    oof_mat = np.column_stack([lgb_oof, xgb_oof, cat_oof])
    test_mat = np.column_stack([lgb_test, xgb_test, cat_test])
    
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_expert_lgbm.npy', lgb_oof)
    np.save('./preds/test_expert_lgbm.npy', lgb_test)
    np.save('./preds/oof_expert_xgb.npy', xgb_oof)
    np.save('./preds/test_expert_xgb.npy', xgb_test)
    np.save('./preds/oof_expert_catboost.npy', cat_oof)
    np.save('./preds/test_expert_catboost.npy', cat_test)
    
    def objective(w):
        w = np.maximum(0, w)
        if w.sum() == 0:
            return 0
        w = w / w.sum()
        blend = np.dot(oof_mat, w)
        return -roc_auc_score(target, blend)
        
    init_w = np.ones(3) / 3.0
    bounds = [(0, 1) for _ in range(3)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x / res.x.sum()
    best_prob_auc = -res.fun
    
    for name, w in zip(models, best_weights):
        print(f"Optimal Weight for {name:10s}: {w:.4f}")
        
    print(f"\nOPTIMAL PROBABILITY BLEND OOF ROC-AUC: {best_prob_auc:.6f}")
    
    # Rank Ensemble
    rank_oof = np.zeros_like(target, dtype=float)
    rank_test = np.zeros(len(test_ids), dtype=float)
    for i, w in enumerate(best_weights):
        rank_oof += w * (rankdata(oof_mat[:, i]) / len(target))
        rank_test += w * (rankdata(test_mat[:, i]) / len(test_ids))
        
    rank_auc = roc_auc_score(target, rank_oof)
    print(f"OPTIMAL RANK BLEND OOF ROC-AUC:        {rank_auc:.6f}")
    
    # Generate Submissions
    final_prob_test = np.dot(test_mat, best_weights)
    sub_blend = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': final_prob_test})
    sub_blend.to_csv('submission_expert_blend.csv', index=False)
    sub_blend.to_csv('submission_ensemble.csv', index=False)
    
    sub_rank = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': rank_test})
    sub_rank.to_csv('submission_expert_rank.csv', index=False)
    sub_rank.to_csv('submission_rank_ensemble.csv', index=False)
    
    print("\nSaved submissions:")
    print("1. submission_expert_blend.csv (also updated submission_ensemble.csv)")
    print("2. submission_expert_rank.csv (also updated submission_rank_ensemble.csv)")
    print("="*75)
    
    return best_prob_auc, rank_auc

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5, help='Number of cross validation folds (default: 5)')
    args = parser.parse_args()
    train_and_ensemble(n_splits=args.folds)
