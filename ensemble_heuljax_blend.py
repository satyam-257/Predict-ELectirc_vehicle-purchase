"""
ensemble_heuljax_blend.py — Blends Heuljax's 173-Feature XGBoost with Ground-Truth Generator Models
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

def rank_norm(arr):
    return rankdata(arr) / len(arr)

def run_blend():
    print("=" * 80)
    print(" S6E9: ENSEMBLE BLENDER — GENERATOR MODELS + HEULJAX XGBOOST")
    print("=" * 80)
    
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    y_true = train['Will_Buy_EV'].eq('Yes').values
    test_ids = test['id'].values
    
    models = {}
    
    # 1. Generator LightGBM
    if os.path.exists('preds/oof_gen_lgbm.npy') and os.path.exists('preds/test_gen_lgbm.npy'):
        models['Gen_LGBM'] = {
            'oof': np.load('preds/oof_gen_lgbm.npy'),
            'test': np.load('preds/test_gen_lgbm.npy')
        }
        
    # 2. Generator XGBoost
    if os.path.exists('preds/oof_gen_xgb.npy') and os.path.exists('preds/test_gen_xgb.npy'):
        models['Gen_XGB'] = {
            'oof': np.load('preds/oof_gen_xgb.npy'),
            'test': np.load('preds/test_gen_xgb.npy')
        }
        
    # 3. Heuljax XGBoost
    if os.path.exists('preds/oof_heuljax_xgb.npy') and os.path.exists('preds/test_heuljax_xgb.npy'):
        models['Heuljax_XGB'] = {
            'oof': np.load('preds/oof_heuljax_xgb.npy'),
            'test': np.load('preds/test_heuljax_xgb.npy')
        }
    elif os.path.exists('oof/HEULJAX_XGB_OOF.parquet') and os.path.exists('test_preds/HEULJAX_XGB_TEST.parquet'):
        oof_df = pd.read_parquet('oof/HEULJAX_XGB_OOF.parquet')
        test_df = pd.read_parquet('test_preds/HEULJAX_XGB_TEST.parquet')
        models['Heuljax_XGB'] = {
            'oof': oof_df['oof_pred'].values,
            'test': test_df['test_pred'].values
        }

    # 4. CatBoost (if available)
    if os.path.exists('preds/oof_catboost.npy') and os.path.exists('preds/test_catboost.npy'):
        oof_cb = np.load('preds/oof_catboost.npy')
        if len(oof_cb) == len(y_true):
            models['CatBoost'] = {
                'oof': oof_cb,
                'test': np.load('preds/test_catboost.npy')
            }

    print(f"\nDiscovered {len(models)} candidate models:")
    for name, m in models.items():
        auc = roc_auc_score(y_true, m['oof'])
        print(f"  • {name:<15}: OOF ROC-AUC = {auc:.6f}")
        
    if len(models) < 2:
        print("\nNeed at least 2 models with completed OOF predictions to blend.")
        return

    # Normalize predictions by rank
    names = list(models.keys())
    k = len(names)
    oof_matrix = np.column_stack([rank_norm(models[n]['oof']) for n in names])
    test_matrix = np.column_stack([rank_norm(models[n]['test']) for n in names])
    
    # 1. Simple Equal Weight Rank Blend
    simple_blend_oof = np.mean(oof_matrix, axis=1)
    simple_auc = roc_auc_score(y_true, simple_blend_oof)
    print(f"\n[Baseline Equal Weight Blend] OOF ROC-AUC: {simple_auc:.6f}")
    
    # 2. Scipy Optimize: Direct ROC-AUC Maximization for Rank Weights
    def loss_func(weights):
        w = np.array(weights)
        w = np.maximum(w, 0)
        s = np.sum(w)
        if s > 0:
            w /= s
        pred = oof_matrix @ w
        # Fast surrogate loss or direct negative AUC
        return -roc_auc_score(y_true, pred)
        
    init_w = np.ones(k) / k
    bounds = [(0.0, 1.0) for _ in range(k)]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
    best_weights = res.x
    best_weights = np.maximum(best_weights, 0)
    best_weights /= np.sum(best_weights)
    
    opt_oof = oof_matrix @ best_weights
    opt_auc = roc_auc_score(y_true, opt_oof)
    
    print("\n" + "=" * 80)
    print(" OPTIMIZED ENSEMBLE WEIGHTS:")
    print("=" * 80)
    for name, w in zip(names, best_weights):
        print(f"  • {name:<15}: weight = {w * 100:6.2f}%")
    print(f"\n---> OPTIMIZED OOF ROC-AUC: {opt_auc:.6f} <---")
    
    # Generate final test predictions
    test_blend = test_matrix @ best_weights
    
    out_file = 'submission_generator_plus_heuljax_ensemble.csv'
    sub_df = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_blend})
    sub_df.to_csv(out_file, index=False)
    print(f"\nSaved ensemble submission to: {out_file}")
    
    # Compare with existing top submission
    if os.path.exists('submission_generator_ensemble_lgb_xgb.csv'):
        prev_sub = pd.read_csv('submission_generator_ensemble_lgb_xgb.csv')
        corr = np.corrcoef(sub_df['Will_Buy_EV'], prev_sub['Will_Buy_EV'])[0, 1]
        print(f"Correlation with previous top submission (LB=0.94384): {corr:.5f}")

if __name__ == '__main__':
    run_blend()
