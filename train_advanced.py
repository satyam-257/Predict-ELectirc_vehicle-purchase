import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.optimize import minimize
from scipy.stats import rankdata

def create_advanced_features():
    print("Loading datasets...")
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    
    # Load original dataset if available
    orig_path = './data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv'
    if os.path.exists(orig_path):
        orig = pd.read_csv(orig_path)
        if 'Buyer_ID' in orig.columns:
            orig = orig.drop(columns=['Buyer_ID'])
        orig['id'] = -1
        # Reorder columns
        orig = orig[['id'] + [c for c in train.columns if c not in ['id', 'Will_Buy_EV']] + ['Will_Buy_EV']]
        print(f"Adding {len(orig)} original dataset rows to training data...")
        train = pd.concat([train, orig], axis=0).reset_index(drop=True)
        
    train['target'] = (train['Will_Buy_EV'] == 'Yes').astype(int)
    n_train = len(train)
    
    df = pd.concat([train.drop(columns=['Will_Buy_EV', 'target']), test], axis=0).reset_index(drop=True)
    
    print("Creating advanced features...")
    # Base numeric mappings
    range_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Range_Anxiety_Num'] = df['Range_Anxiety_Level'].map(range_map).fillna(0).astype(int)
    df['Subsidy_Available_Num'] = (df['Subsidy_Available'] == 'Yes').astype(int)
    df['Home_Charging_Num'] = (df['Home_Charging_Possible'] == 'Yes').astype(int)
    
    # Non-linear representations
    df['Env_Concern_Sq'] = df['Environmental_Concern_Level'] ** 2
    df['Env_Concern_Cube'] = df['Environmental_Concern_Level'] ** 3
    df['Log_Income'] = np.log1p(df['Annual_Income_USD'])
    df['Log_Commute'] = np.log1p(df['Daily_Commute_km'])
    df['Sqrt_Income'] = np.sqrt(df['Annual_Income_USD'])
    
    # Charging metrics
    df['Total_Charging'] = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    df['Charging_Diff'] = df['Charging_Stations_Near_Work'] - df['Charging_Stations_Near_Home']
    df['Charging_Ratio'] = (df['Charging_Stations_Near_Home'] + 1) / (df['Charging_Stations_Near_Work'] + 1)
    df['Charging_Per_Km'] = df['Total_Charging'] / (df['Daily_Commute_km'] + 1)
    df['Home_Charging_Per_Km'] = df['Charging_Stations_Near_Home'] / (df['Daily_Commute_km'] + 1)
    df['Work_Charging_Per_Km'] = df['Charging_Stations_Near_Work'] / (df['Daily_Commute_km'] + 1)
    df['Commute_Per_Total_Charging'] = df['Daily_Commute_km'] / (df['Total_Charging'] + 1)
    
    # Income metrics
    df['Income_Per_Car'] = df['Annual_Income_USD'] / (df['Number_of_Cars_Owned'] + 1)
    df['Income_Per_Age'] = df['Annual_Income_USD'] / (df['Age'] + 1)
    df['Income_Per_Commute'] = df['Annual_Income_USD'] / (df['Daily_Commute_km'] + 1)
    df['Cars_Per_Age'] = df['Number_of_Cars_Owned'] / (df['Age'] + 1)
    
    # Interactions
    df['Subsidy_X_EnvConcern'] = df['Subsidy_Available_Num'] * df['Environmental_Concern_Level']
    df['HomeCharge_X_EnvConcern'] = df['Home_Charging_Num'] * df['Environmental_Concern_Level']
    df['Subsidy_X_HomeCharge'] = df['Subsidy_Available_Num'] * df['Home_Charging_Num']
    df['Subsidy_X_HomeCharge_X_Env'] = df['Subsidy_Available_Num'] * df['Home_Charging_Num'] * df['Environmental_Concern_Level']
    df['Income_X_EnvConcern'] = (df['Annual_Income_USD'] / 10000.0) * df['Environmental_Concern_Level']
    df['LogIncome_X_EnvConcern'] = df['Log_Income'] * df['Environmental_Concern_Level']
    df['Commute_X_RangeAnxiety'] = df['Daily_Commute_km'] * (df['Range_Anxiety_Num'] + 1)
    df['Charging_X_RangeAnxiety'] = (df['Total_Charging'] + 1) / (df['Range_Anxiety_Num'] + 1)
    
    df['EV_Readiness_Score'] = (
        df['Environmental_Concern_Level'] * 2.0 +
        df['Subsidy_Available_Num'] * 3.0 +
        df['Home_Charging_Num'] * 1.5 -
        df['Range_Anxiety_Num'] * 2.5
    )
    
    # Categorical Combinations
    df['City_Car'] = df['City_Type'] + '_' + df['Current_Car_Type']
    df['City_Subsidy'] = df['City_Type'] + '_' + df['Subsidy_Available']
    df['City_HomeCharge'] = df['City_Type'] + '_' + df['Home_Charging_Possible']
    df['HomeCharge_RangeAnxiety'] = df['Home_Charging_Possible'] + '_' + df['Range_Anxiety_Level']
    df['Subsidy_RangeAnxiety'] = df['Subsidy_Available'] + '_' + df['Range_Anxiety_Level']
    df['Gender_Car'] = df['Gender'] + '_' + df['Current_Car_Type']
    df['City_RangeAnxiety'] = df['City_Type'] + '_' + df['Range_Anxiety_Level']
    df['Car_RangeAnxiety'] = df['Current_Car_Type'] + '_' + df['Range_Anxiety_Level']
    
    # Group Aggregations
    for grp in [['City_Type', 'Current_Car_Type'], ['City_Type'], ['Current_Car_Type'], ['Range_Anxiety_Level'], ['City_Type', 'Subsidy_Available']]:
        grp_name = '_'.join(grp)
        for num_feat in ['Annual_Income_USD', 'Daily_Commute_km', 'Environmental_Concern_Level', 'Total_Charging', 'Age']:
            agg_stats = df.groupby(grp)[num_feat].agg(['mean', 'std', 'median']).reset_index()
            agg_stats.columns = grp + [f'{grp_name}_{num_feat}_mean', f'{grp_name}_{num_feat}_std', f'{grp_name}_{num_feat}_median']
            df = df.merge(agg_stats, on=grp, how='left')
            df[f'{num_feat}_diff_{grp_name}_mean'] = df[num_feat] - df[f'{grp_name}_{num_feat}_mean']
            df[f'{num_feat}_ratio_{grp_name}_mean'] = df[num_feat] / (df[f'{grp_name}_{num_feat}_mean'] + 1e-6)
            
    # Frequency Encodings
    freq_cols = [
        'Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 
        'Subsidy_Available', 'Range_Anxiety_Level', 'City_Car', 'City_Subsidy', 
        'City_HomeCharge', 'HomeCharge_RangeAnxiety', 'Subsidy_RangeAnxiety', 
        'Gender_Car', 'City_RangeAnxiety', 'Car_RangeAnxiety'
    ]
    for c in freq_cols:
        df[f'{c}_freq'] = df[c].map(df[c].value_counts(normalize=True))
        
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_columns:
        df[c] = df[c].astype('category')
        
    print(f"Total features created: {df.shape[1]}")
    
    train_features = df.iloc[:n_train].copy()
    train_features['target'] = train['target'].values
    test_features = df.iloc[n_train:].copy()
    
    return train_features, test_features, cat_columns

def train_advanced_pipeline():
    train_df, test_df, cat_cols = create_advanced_features()
    
    features = [c for c in train_df.columns if c not in ['id', 'target', 'Will_Buy_EV']]
    target = train_df['target'].values
    test_ids = test_df['id'].values
    
    N_SPLITS = 10
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(train_df))
    test_lgb = np.zeros(len(test_df))
    
    oof_xgb = np.zeros(len(train_df))
    test_xgb = np.zeros(len(test_df))
    
    print("\n" + "="*60)
    print(f"STARTING {N_SPLITS}-FOLD STRATIFIED TRAINING PIPELINE")
    print("="*60)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 40,
        'subsample': 0.85,
        'subsample_freq': 1,
        'colsample_bytree': 0.75,
        'reg_alpha': 0.05,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'learning_rate': 0.03,
        'max_depth': 7,
        'subsample': 0.85,
        'colsample_bytree': 0.75,
        'min_child_weight': 25,
        'reg_alpha': 0.05,
        'reg_lambda': 2.0,
        'random_state': 42,
        'enable_categorical': True,
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        print(f"\n================ FOLD {fold + 1} / {N_SPLITS} ================")
        X_train, y_train = train_df.iloc[train_idx][features], target[train_idx]
        X_val, y_val = train_df.iloc[val_idx][features], target[val_idx]
        
        # 1. LightGBM
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)
        
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=4000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=120, verbose=False)]
        )
        val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        oof_lgb[val_idx] = val_pred_lgb
        test_lgb += lgb_model.predict(test_df[features], num_iteration=lgb_model.best_iteration) / N_SPLITS
        print(f"Fold {fold+1} LGBM AUC: {roc_auc_score(y_val, val_pred_lgb):.6f} (Best Iter: {lgb_model.best_iteration})")
        
        # 2. XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        dtest = xgb.DMatrix(test_df[features], enable_categorical=True)
        
        xgb_model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=4000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=120,
            verbose_eval=False
        )
        val_pred_xgb = xgb_model.predict(dval)
        oof_xgb[val_idx] = val_pred_xgb
        test_xgb += xgb_model.predict(dtest) / N_SPLITS
        print(f"Fold {fold+1} XGB  AUC: {roc_auc_score(y_val, val_pred_xgb):.6f} (Best Iter: {xgb_model.best_iteration})")
        
    print("\n" + "="*60)
    print("TRAINING COMPLETE! COMPUTING FINAL SCORES")
    print("="*60)
    
    auc_lgb = roc_auc_score(target, oof_lgb)
    auc_xgb = roc_auc_score(target, oof_xgb)
    print(f"10-Fold LightGBM OOF ROC-AUC: {auc_lgb:.6f}")
    print(f"10-Fold XGBoost  OOF ROC-AUC: {auc_xgb:.6f}")
    
    # Optimize blend
    def blend_obj(w):
        w1 = w[0]
        w2 = 1.0 - w1
        blend = w1 * oof_lgb + w2 * oof_xgb
        return -roc_auc_score(target, blend)
        
    res = minimize(blend_obj, [0.5], bounds=[(0, 1)], method='L-BFGS-B')
    best_w1 = res.x[0]
    best_w2 = 1.0 - best_w1
    
    best_blend_oof = best_w1 * oof_lgb + best_w2 * oof_xgb
    best_blend_auc = roc_auc_score(target, best_blend_oof)
    print(f"\nBest Blend Weights: LightGBM = {best_w1:.4f}, XGBoost = {best_w2:.4f}")
    print(f"BLENDED 10-FOLD OOF ROC-AUC: {best_blend_auc:.6f}")
    
    # Rank average
    rank_oof = best_w1 * (rankdata(oof_lgb) / len(target)) + best_w2 * (rankdata(oof_xgb) / len(target))
    rank_auc = roc_auc_score(target, rank_oof)
    print(f"RANK-AVERAGED 10-FOLD OOF ROC-AUC: {rank_auc:.6f}")
    
    # Save test predictions
    test_blend = best_w1 * test_lgb + best_w2 * test_xgb
    sub_blend = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_blend})
    sub_blend.to_csv('submission_advanced_10fold_blend.csv', index=False)
    
    test_rank = best_w1 * (rankdata(test_lgb) / len(test_ids)) + best_w2 * (rankdata(test_xgb) / len(test_ids))
    sub_rank = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_rank})
    sub_rank.to_csv('submission_advanced_10fold_rank.csv', index=False)
    
    print("Saved submission_advanced_10fold_blend.csv and submission_advanced_10fold_rank.csv")

if __name__ == '__main__':
    train_advanced_pipeline()
