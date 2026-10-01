import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import engineer_features
from target_encoder import add_fold_safe_target_encoding
import os

def train_xgb():
    print("="*60)
    print("XGBOOST 5-FOLD STRATIFIED TRAINING")
    print("="*60)
    
    train_df, test_df, cat_cols = engineer_features()
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    params = {
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
    
    fold_scores = []
    te_target_cols = ['City_Subsidy_Interaction', 'Sub_Anx_Group', 'City_Car_Interaction']
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        print(f"\n--- Fold {fold + 1} / 5 ---")
        X_train_raw, y_train = train_df.iloc[train_idx][features], target[train_idx]
        X_val_raw, y_val = train_df.iloc[val_idx][features], target[val_idx]
        X_test_raw = test_df[features].copy()
        
        # Apply fold-safe leak-free target encoding
        X_train, X_val, X_test_fold = add_fold_safe_target_encoding(
            X_train_raw, y_train, X_val_raw, X_test_raw,
            te_cols=te_target_cols, smoothing=20.0, random_state=42 + fold
        )
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_test_fold, enable_categorical=True)
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=200
        )
        
        val_pred = model.predict(dval)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.6f} (Best Iteration: {model.best_iteration})")
        
        test_preds += model.predict(dtest) / 5.0
        
    overall_auc = roc_auc_score(target, oof_preds)
    print("\n" + "="*60)
    print(f"XGBOOST OVERALL OOF ROC-AUC: {overall_auc:.6f}")
    print(f"Fold scores: {[round(s, 6) for s in fold_scores]}")
    print("="*60)
    
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_xgb.npy', oof_preds)
    np.save('./preds/test_xgb.npy', test_preds)
    
    sub = pd.DataFrame({
        'id': test_df['id'].values,
        'Will_Buy_EV': test_preds
    })
    sub.to_csv('./preds/submission_xgb.csv', index=False)
    print("Saved ./preds/submission_xgb.csv")
    
    return oof_preds, test_preds, overall_auc

if __name__ == '__main__':
    train_xgb()
