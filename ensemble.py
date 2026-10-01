import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
import os

def load_data():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    target = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    test_ids = test['id'].values
    return target, test_ids

def blend_and_stack():
    target, test_ids = load_data()
    
    models = []
    oof_list = []
    test_list = []
    
    available_models = [
        ('LightGBM', './preds/oof_lgbm.npy', './preds/test_lgbm.npy'),
        ('XGBoost', './preds/oof_xgb.npy', './preds/test_xgb.npy'),
        ('CatBoost', './preds/oof_catboost.npy', './preds/test_catboost.npy')
    ]
    
    for name, oof_path, test_path in available_models:
        if os.path.exists(oof_path) and os.path.exists(test_path):
            oof = np.load(oof_path)
            test_p = np.load(test_path)
            auc = roc_auc_score(target, oof)
            print(f"Loaded {name:10s} | OOF ROC-AUC: {auc:.6f}")
            models.append(name)
            oof_list.append(oof)
            test_list.append(test_p)
            
    if not models:
        print("No model predictions found in ./preds!")
        return
        
    n_models = len(models)
    oof_mat = np.column_stack(oof_list)
    test_mat = np.column_stack(test_list)
    
    if n_models == 1:
        print("Only 1 model available. Writing submission directly.")
        sub = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_list[0]})
        sub.to_csv('./preds/submission_ensemble.csv', index=False)
        return
        
    print("\n" + "="*60)
    print("FINDING OPTIMAL BLEND WEIGHTS (Maximizing ROC-AUC)")
    print("="*60)
    
    # Negative ROC AUC objective
    def objective(weights):
        weights = np.array(weights)
        weights = np.maximum(0, weights)
        if weights.sum() == 0:
            return 0
        w = weights / weights.sum()
        blend = np.dot(oof_mat, w)
        return -roc_auc_score(target, blend)
        
    init_weights = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    res = minimize(
        objective, 
        init_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    
    best_weights = res.x / res.x.sum()
    best_oof_auc = -res.fun
    
    for name, w in zip(models, best_weights):
        print(f"Weight for {name:10s}: {w:.4f}")
        
    print(f"\nOPTIMAL BLENDED OOF ROC-AUC: {best_oof_auc:.6f}")
    
    # Also evaluate weighted and equal rank averaging
    rank_oof = np.zeros_like(target, dtype=float)
    rank_test = np.zeros(len(test_ids), dtype=float)
    rank_oof_equal = np.zeros_like(target, dtype=float)
    rank_test_equal = np.zeros(len(test_ids), dtype=float)
    
    for i, w in enumerate(best_weights):
        r_oof = rankdata(oof_mat[:, i]) / len(target)
        r_test = rankdata(test_mat[:, i]) / len(test_ids)
        rank_oof += w * r_oof
        rank_test += w * r_test
        rank_oof_equal += (1.0 / n_models) * r_oof
        rank_test_equal += (1.0 / n_models) * r_test
        
    rank_auc = roc_auc_score(target, rank_oof)
    rank_equal_auc = roc_auc_score(target, rank_oof_equal)
    print(f"OPTIMAL RANK-AVERAGED OOF ROC-AUC: {rank_auc:.6f}")
    print(f"EQUAL RANK-AVERAGED OOF ROC-AUC:   {rank_equal_auc:.6f}")
    
    # Generate weighted probability blend
    final_test_pred = np.dot(test_mat, best_weights)
    sub = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': final_test_pred
    })
    sub.to_csv('submission_ensemble.csv', index=False)
    sub.to_csv('./preds/submission_ensemble.csv', index=False)
    print("\nSaved submission_ensemble.csv (probability blend)")
    
    # Generate rank average submission (use the higher of equal vs optimal rank)
    best_rank_test = rank_test if rank_auc >= rank_equal_auc else rank_test_equal
    sub_rank = pd.DataFrame({
        'id': test_ids,
        'Will_Buy_EV': best_rank_test
    })
    sub_rank.to_csv('submission_rank_ensemble.csv', index=False)
    sub_rank.to_csv('./preds/submission_rank_ensemble.csv', index=False)
    print("Saved submission_rank_ensemble.csv (rank blend)")

if __name__ == '__main__':
    blend_and_stack()
