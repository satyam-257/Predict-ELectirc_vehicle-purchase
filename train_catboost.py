import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import engineer_features
from target_encoder import add_fold_safe_target_encoding
import os

def train_catboost():
    print("="*60)
    print("CATBOOST 5-FOLD STRATIFIED TRAINING")
    print("="*60)
    
    train_df, test_df, cat_cols = engineer_features()
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    
    # Cast cat_cols to string for CatBoost
    for c in cat_cols:
        train_df[c] = train_df[c].astype(str)
        test_df[c] = test_df[c].astype(str)
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
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
        
        # CatBoost cat_features must only include columns that exist and are categorical
        cat_features_in_data = [c for c in cat_cols if c in X_train.columns]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features_in_data)
        val_pool = Pool(X_val, y_val, cat_features=cat_features_in_data)
        test_pool = Pool(X_test_fold, cat_features=cat_features_in_data)
        
        model = CatBoostClassifier(
            iterations=2500,
            learning_rate=0.06,
            depth=6,
            eval_metric='AUC',
            random_seed=42,
            task_type='CPU',
            thread_count=-1,
            early_stopping_rounds=100,
            verbose=250
        )
        
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        val_pred = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.6f}")
        
        test_preds += model.predict_proba(test_pool)[:, 1] / 5.0
        
    overall_auc = roc_auc_score(target, oof_preds)
    print("\n" + "="*60)
    print(f"CATBOOST OVERALL OOF ROC-AUC: {overall_auc:.6f}")
    print(f"Fold scores: {[round(s, 6) for s in fold_scores]}")
    print("="*60)
    
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_catboost.npy', oof_preds)
    np.save('./preds/test_catboost.npy', test_preds)
    
    sub = pd.DataFrame({
        'id': test_df['id'].values,
        'Will_Buy_EV': test_preds
    })
    sub.to_csv('./preds/submission_catboost.csv', index=False)
    print("Saved ./preds/submission_catboost.csv")
    
    return oof_preds, test_preds, overall_auc

if __name__ == '__main__':
    train_catboost()
