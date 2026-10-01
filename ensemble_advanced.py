import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def load_data_and_models():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    target = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    test_ids = test['id'].values

    candidates = [
        ('LightGBM',      './preds/oof_lgbm.npy',           './preds/test_lgbm.npy'),
        ('XGBoost',       './preds/oof_xgb.npy',            './preds/test_xgb.npy'),
        ('CatBoost',      './preds/oof_catboost.npy',       './preds/test_catboost.npy'),
        ('NeuralNet',     './preds/oof_nn.npy',             './preds/test_nn.npy'),
        ('TabResNet',     './preds/oof_tabular_resnet.npy', './preds/test_tabular_resnet.npy'),
        ('RealMLP',       './preds/oof_realmlp.npy',        './preds/test_realmlp.npy'),
    ]

    models = []
    oof_list = []
    test_list = []

    print("=" * 70)
    print("S6E9 BEYOND BOOSTING: ADVANCED ENSEMBLING & STACKING PIPELINE")
    print("=" * 70)
    print("Loading candidate model predictions...")

    for name, oof_p, test_p in candidates:
        if os.path.exists(oof_p) and os.path.exists(test_p):
            oof = np.load(oof_p)
            t_pred = np.load(test_p)
            auc = roc_auc_score(target, oof)
            print(f"-> [FOUND] {name:12s} | OOF ROC-AUC: {auc:.6f}")
            models.append(name)
            oof_list.append(oof)
            test_list.append(t_pred)
        else:
            print(f"-> [SKIP]  {name:12s} | Files not found: {oof_p}")

    if not models:
        raise FileNotFoundError("No model predictions found in ./preds!")

    oof_mat = np.column_stack(oof_list)
    test_mat = np.column_stack(test_list)

    return target, test_ids, models, oof_mat, test_mat

def caruana_hill_climbing(oof_mat, target, max_steps=100):
    """
    Caruana Forward Hill Climbing with Replacement.
    Regularized discrete model selection to prevent overfitting continuous weights.
    """
    n_models = oof_mat.shape[1]
    # Step 1: Find best single model
    single_aucs = [roc_auc_score(target, oof_mat[:, i]) for i in range(n_models)]
    best_init = np.argmax(single_aucs)
    
    ensemble = [best_init]
    current_pred = oof_mat[:, best_init].copy()
    best_auc = single_aucs[best_init]

    for step in range(2, max_steps + 1):
        best_candidate = None
        candidate_best_auc = -1
        best_cand_pred = None

        for m in range(n_models):
            trial_pred = (current_pred * (step - 1) + oof_mat[:, m]) / step
            trial_auc = roc_auc_score(target, trial_pred)
            if trial_auc > candidate_best_auc:
                candidate_best_auc = trial_auc
                best_candidate = m
                best_cand_pred = trial_pred

        # Only add if it doesn't degrade performance
        if candidate_best_auc >= best_auc:
            best_auc = candidate_best_auc
            best_cand_pred_saved = best_cand_pred
            current_pred = best_cand_pred_saved
            ensemble.append(best_candidate)
        else:
            break

    # Calculate weights from selections
    counts = np.bincount(ensemble, minlength=n_models)
    weights = counts / len(ensemble)
    return weights, best_auc, current_pred

def run_advanced_ensemble():
    target, test_ids, models, oof_mat, test_mat = load_data_and_models()
    n_models = len(models)

    results = {}

    # 1. Simple Mean & Rank Average
    mean_oof = np.mean(oof_mat, axis=1)
    mean_test = np.mean(test_mat, axis=1)
    mean_auc = roc_auc_score(target, mean_oof)
    results['Simple Equal Blend'] = (mean_auc, mean_oof, mean_test)

    rank_oof = np.mean(np.column_stack([rankdata(oof_mat[:, i]) / len(target) for i in range(n_models)]), axis=1)
    rank_test = np.mean(np.column_stack([rankdata(test_mat[:, i]) / len(test_ids) for i in range(n_models)]), axis=1)
    rank_auc = roc_auc_score(target, rank_oof)
    results['Simple Rank Average'] = (rank_auc, rank_oof, rank_test)

    # 2. Continuous SLSQP Optimization (AUC Maximizer)
    def objective(w):
        w = np.maximum(0, w)
        if w.sum() == 0:
            return 0
        w = w / w.sum()
        pred = np.dot(oof_mat, w)
        return -roc_auc_score(target, pred)

    res = minimize(
        objective,
        np.ones(n_models) / n_models,
        method='SLSQP',
        bounds=[(0, 1) for _ in range(n_models)],
        constraints={'type': 'eq', 'fun': lambda w: 1.0 - sum(w)},
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    slsqp_weights = np.maximum(0, res.x)
    slsqp_weights /= slsqp_weights.sum()
    slsqp_oof = np.dot(oof_mat, slsqp_weights)
    slsqp_test = np.dot(test_mat, slsqp_weights)
    slsqp_auc = roc_auc_score(target, slsqp_oof)
    results['SLSQP Weighted Blend'] = (slsqp_auc, slsqp_oof, slsqp_test)

    # 3. Caruana Hill Climbing
    hc_weights, hc_auc, hc_oof = caruana_hill_climbing(oof_mat, target, max_steps=100)
    hc_test = np.dot(test_mat, hc_weights)
    results['Hill Climbing Blend'] = (hc_auc, hc_oof, hc_test)

    # 4. Out-of-Fold Stacking with Logit Transformation
    eps = 1e-6
    logit_oof = logit(np.clip(oof_mat, eps, 1.0 - eps))
    logit_test = logit(np.clip(test_mat, eps, 1.0 - eps))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_oof = np.zeros(len(target))
    stack_test = np.zeros(len(test_ids))

    for tr_idx, va_idx in skf.split(logit_oof, target):
        meta_model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=500)
        meta_model.fit(logit_oof[tr_idx], target[tr_idx])
        stack_oof[va_idx] = meta_model.predict_proba(logit_oof[va_idx])[:, 1]
        stack_test += meta_model.predict_proba(logit_test)[:, 1] / 5.0

    stack_auc = roc_auc_score(target, stack_oof)
    results['Logit Stacking (Meta-Learner)'] = (stack_auc, stack_oof, stack_test)

    # Summary Table
    print("\n" + "=" * 70)
    print("ENSEMBLE STRATEGY COMPARISON")
    print("=" * 70)
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for name, (auc, _, _) in sorted_results:
        print(f"{name:30s} | OOF ROC-AUC: {auc:.6f}")

    print("\n" + "-" * 70)
    print("OPTIMIZED WEIGHT ALLOCATION:")
    print("-" * 70)
    print(f"{'Model':15s} | {'SLSQP Weight':15s} | {'Hill-Climbing Weight':20s}")
    for m, sw, hw in zip(models, slsqp_weights, hc_weights):
        print(f"{m:15s} | {sw:15.4f} | {hw:20.4f}")

    # Best submission
    best_name, (best_auc, best_oof, best_test) = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"BEST STRATEGY: {best_name} (OOF AUC: {best_auc:.6f})")
    print("=" * 70)

    sub_path = './preds/submission_beyond_boosting.csv'
    sub = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': best_test})
    sub.to_csv(sub_path, index=False)
    print(f"Generated submission file: {sub_path}")

    # Also save the blended OOF
    np.save('./preds/oof_beyond_boosting.npy', best_oof)
    print("Saved ./preds/oof_beyond_boosting.npy")

if __name__ == '__main__':
    run_advanced_ensemble()
