import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import engineer_features
from target_encoder import add_fold_safe_target_encoding
import os

def train_lgbm():
    print("="*60)
    print("LIGHTGBM 5-FOLD STRATIFIED TRAINING")
    print("="*60)
    
    train_df, test_df, cat_cols = engineer_features()
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    params = {
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
    
    fold_scores = []
    te_target_cols = ['City_Subsidy_Interaction', 'Sub_Anx_Group', 'City_Car_Interaction']
    feature_names = None
    feature_importances = None
    
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
        
        if feature_names is None:
            feature_names = list(X_train.columns)
            feature_importances = pd.DataFrame({'feature': feature_names, 'importance': 0.0})
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=200)
        ]
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=3000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_scores.append(fold_auc)
        print(f"Fold {fold + 1} ROC-AUC: {fold_auc:.6f} (Best Iteration: {model.best_iteration})")
        
        feature_importances['importance'] += model.feature_importance(importance_type='gain') / 5.0
        test_preds += model.predict(X_test_fold, num_iteration=model.best_iteration) / 5.0
        
    overall_auc = roc_auc_score(target, oof_preds)
    print("\n" + "="*60)
    print(f"LIGHTGBM OVERALL OOF ROC-AUC: {overall_auc:.6f}")
    print(f"Fold scores: {[round(s, 6) for s in fold_scores]}")
    print("="*60)
    
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_lgbm.npy', oof_preds)
    np.save('./preds/test_lgbm.npy', test_preds)
    
    # Save individual submission
    sub = pd.DataFrame({
        'id': test_df['id'].values,
        'Will_Buy_EV': test_preds
    })
    sub.to_csv('./preds/submission_lgbm.csv', index=False)
    print("Saved ./preds/submission_lgbm.csv")
    
    # Print top 15 features
    print("\nTop 15 Most Important Features:")
    print(feature_importances.sort_values(by='importance', ascending=False).head(15).to_string(index=False))
    
    return oof_preds, test_preds, overall_auc

if __name__ == '__main__':
    train_lgbm()
