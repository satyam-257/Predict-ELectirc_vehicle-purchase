"""
ensemble_v3.py — Final ensemble incorporating all strategy insights from S6E9 discussion

Strategies compared:
  1. Simple equal blend
  2. Rank average
  3. SLSQP continuous weight optimisation
  4. Caruana forward hill-climbing (with replacement)
  5. Bagged hill-climbing (40 bootstraps) — most robust
  6. Logistic Regression stacker (logit-space inputs)
  7. CatBoost meta-learner stacker (if catboost available)

Selection criterion: Nested 5-fold CV on the best strategy (not raw OOF)
All OOF files in ./preds/ are loaded automatically.

Usage:
    python ensemble_v3.py
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_all(preds_dir='./preds'):
    train   = pd.read_csv('./data/train.csv')
    test    = pd.read_csv('./data/test.csv')
    target  = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    test_ids = test['id'].values

    oofs, tests, names = [], [], []
    for f in sorted(os.listdir(preds_dir)):
        if not f.startswith('oof_') or not f.endswith('.npy'):
            continue
        test_f = f.replace('oof_', 'test_')
        oof_path  = f'{preds_dir}/{f}'
        test_path = f'{preds_dir}/{test_f}'
        if not os.path.exists(test_path):
            print(f"  [SKIP] No matching test file for {f}")
            continue
        oof = np.load(oof_path)
        tst = np.load(test_path)
        if len(oof) != len(target):
            print(f"  [SKIP] {f} shape mismatch ({len(oof)} vs {len(target)})")
            continue
        name = f.replace('oof_', '').replace('.npy', '')
        auc  = roc_auc_score(target, oof)
        print(f"  [LOAD] {name:25s}  OOF AUC={auc:.6f}")
        oofs.append(oof)
        tests.append(tst)
        names.append(name)

    oof_mat  = np.column_stack(oofs)
    test_mat = np.column_stack(tests)
    return target, test_ids, names, oof_mat, test_mat


# ─────────────────────────────────────────────────────────────────────────────
# Caruana hill-climbing
# ─────────────────────────────────────────────────────────────────────────────
def caruana_hc(oof_mat, target, max_steps=200):
    n_models   = oof_mat.shape[1]
    solo_aucs  = [roc_auc_score(target, oof_mat[:, i]) for i in range(n_models)]
    best_init  = np.argmax(solo_aucs)
    ensemble   = [best_init]
    cur_pred   = oof_mat[:, best_init].copy()
    best_auc   = solo_aucs[best_init]

    for step in range(2, max_steps + 1):
        bc, ba, bp = None, -1.0, None
        for m in range(n_models):
            trial = (cur_pred * (step - 1) + oof_mat[:, m]) / step
            tauc  = roc_auc_score(target, trial)
            if tauc > ba:
                ba, bc, bp = tauc, m, trial
        if ba >= best_auc:
            best_auc = ba
            cur_pred = bp
            ensemble.append(bc)
        else:
            break

    counts  = np.bincount(ensemble, minlength=n_models)
    weights = counts / len(ensemble)
    return weights, best_auc, cur_pred


# ─────────────────────────────────────────────────────────────────────────────
# Bagged hill-climbing
# ─────────────────────────────────────────────────────────────────────────────
def bagged_hc(oof_mat, target, n_bags=40, seed=42):
    rng = np.random.RandomState(seed)
    n   = len(target)
    all_weights = []
    for _ in range(n_bags):
        idx = rng.choice(n, size=n, replace=True)
        w, _, _ = caruana_hc(oof_mat[idx], target[idx])
        all_weights.append(w)
    mean_w = np.array(all_weights).mean(axis=0)
    mean_w /= mean_w.sum()
    blend  = np.dot(oof_mat, mean_w)
    auc    = roc_auc_score(target, blend)
    return mean_w, auc, blend


# ─────────────────────────────────────────────────────────────────────────────
# SLSQP weight optimisation
# ─────────────────────────────────────────────────────────────────────────────
def slsqp_blend(oof_mat, target):
    n = oof_mat.shape[1]
    def obj(w):
        w = np.maximum(0, w)
        s = w.sum()
        if s == 0:
            return 0.0
        return -roc_auc_score(target, np.dot(oof_mat, w / s))
    res = minimize(obj, np.ones(n) / n, method='SLSQP',
                   bounds=[(0, 1)] * n,
                   constraints={'type': 'eq', 'fun': lambda w: 1.0 - w.sum()},
                   options={'maxiter': 500, 'ftol': 1e-8})
    w = np.maximum(0, res.x)
    w /= w.sum()
    blend = np.dot(oof_mat, w)
    return w, roc_auc_score(target, blend), blend


# ─────────────────────────────────────────────────────────────────────────────
# LR stacker in logit space
# ─────────────────────────────────────────────────────────────────────────────
def lr_stacker(oof_mat, target, n_splits=5):
    eps     = 1e-6
    lo_mat  = logit(np.clip(oof_mat, eps, 1.0 - eps))
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_out = np.zeros(len(target))

    for tr_idx, va_idx in skf.split(lo_mat, target):
        meta = LogisticRegression(C=0.05, penalty='l2', solver='lbfgs', max_iter=1000)
        meta.fit(lo_mat[tr_idx], target[tr_idx])
        oof_out[va_idx] = meta.predict_proba(lo_mat[va_idx])[:, 1]

    # Refit on full data for test predictions
    meta_full = LogisticRegression(C=0.05, penalty='l2', solver='lbfgs', max_iter=1000)
    meta_full.fit(lo_mat, target)
    return meta_full, roc_auc_score(target, oof_out), oof_out


# ─────────────────────────────────────────────────────────────────────────────
# Nested 5-fold CV to verify strategy AUC
# ─────────────────────────────────────────────────────────────────────────────
def nested_cv_score(oof_mat, target, strategy_fn, n_outer=5):
    """Estimate held-out AUC for a given strategy function."""
    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    held_aucs = []
    for tr_idx, va_idx in skf.split(oof_mat, target):
        w, _, _ = strategy_fn(oof_mat[tr_idx], target[tr_idx])
        va_blend = np.dot(oof_mat[va_idx], w)
        held_aucs.append(roc_auc_score(target[va_idx], va_blend))
    return np.mean(held_aucs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_ensemble():
    print("=" * 70)
    print("S6E9 ENSEMBLE v3 — FULL STRATEGY COMPARISON")
    print("=" * 70)

    target, test_ids, names, oof_mat, test_mat = load_all()
    n_models = len(names)

    # Spearman correlation vs best single model (anchor)
    solo_aucs = [roc_auc_score(target, oof_mat[:, i]) for i in range(n_models)]
    anchor    = np.argmax(solo_aucs)
    print(f"\nAnchor model: {names[anchor]}  (AUC={solo_aucs[anchor]:.6f})")
    print("Spearman rho vs anchor:")
    for i, name in enumerate(names):
        r, _ = spearmanr(oof_mat[:, anchor], oof_mat[:, i])
        flag  = "  <- genuine diversity!" if r < 0.985 else ""
        print(f"  {name:25s}  ρ={r:.4f}{flag}")

    results = {}  # name → (oof_auc, nested_auc, oof_blend, test_blend, weights)

    # ── 1. Simple mean ──────────────────────────────────────────────────────
    w_eq   = np.ones(n_models) / n_models
    bl_eq  = np.dot(oof_mat, w_eq)
    tst_eq = np.dot(test_mat, w_eq)
    results['equal_mean'] = (roc_auc_score(target, bl_eq), None, bl_eq, tst_eq, w_eq)

    # ── 2. Rank average ─────────────────────────────────────────────────────
    rank_oof  = np.mean(np.column_stack([rankdata(oof_mat[:, i]) / len(target) for i in range(n_models)]), axis=1)
    rank_test = np.mean(np.column_stack([rankdata(test_mat[:, i]) / len(test_ids) for i in range(n_models)]), axis=1)
    results['rank_avg'] = (roc_auc_score(target, rank_oof), None, rank_oof, rank_test, w_eq)

    # ── 3. SLSQP ────────────────────────────────────────────────────────────
    w_sq, auc_sq, bl_sq = slsqp_blend(oof_mat, target)
    tst_sq = np.dot(test_mat, w_sq)
    results['slsqp'] = (auc_sq, None, bl_sq, tst_sq, w_sq)

    # ── 4. Caruana HC ───────────────────────────────────────────────────────
    w_hc, auc_hc, bl_hc = caruana_hc(oof_mat, target, max_steps=200)
    tst_hc = np.dot(test_mat, w_hc)
    ncv_hc = nested_cv_score(oof_mat, target, caruana_hc, n_outer=5)
    results['hill_climb'] = (auc_hc, ncv_hc, bl_hc, tst_hc, w_hc)

    # ── 5. Bagged HC ────────────────────────────────────────────────────────
    print("\nRunning bagged hill-climb (40 bags)…")
    w_bag, auc_bag, bl_bag = bagged_hc(oof_mat, target, n_bags=40)
    tst_bag = np.dot(test_mat, w_bag)
    results['bagged_hc'] = (auc_bag, None, bl_bag, tst_bag, w_bag)

    # ── 6. LR stacker ───────────────────────────────────────────────────────
    eps = 1e-6
    lo_test  = logit(np.clip(test_mat, eps, 1.0 - eps))
    meta_lr, auc_lr, bl_lr = lr_stacker(oof_mat, target)
    tst_lr = meta_lr.predict_proba(lo_test)[:, 1]
    results['lr_stacker'] = (auc_lr, None, bl_lr, tst_lr, None)

    # ── 7. CatBoost stacker (optional) ──────────────────────────────────────
    try:
        from catboost import CatBoostClassifier, Pool
        print("Running CatBoost meta-learner…")
        skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        bl_cb = np.zeros(len(target))
        eps   = 1e-6
        lo_tr = logit(np.clip(oof_mat,  eps, 1 - eps))
        lo_te = logit(np.clip(test_mat, eps, 1 - eps))
        tst_cb_acc = np.zeros(len(test_ids))

        for tr_idx, va_idx in skf.split(lo_tr, target):
            cb = CatBoostClassifier(
                iterations=300, learning_rate=0.1, depth=4,
                eval_metric='AUC', random_seed=42, verbose=False,
                early_stopping_rounds=30,
            )
            cb.fit(lo_tr[tr_idx], target[tr_idx],
                   eval_set=(lo_tr[va_idx], target[va_idx]))
            bl_cb[va_idx] = cb.predict_proba(lo_tr[va_idx])[:, 1]
            tst_cb_acc   += cb.predict_proba(lo_te)[:, 1] / 5

        auc_cb = roc_auc_score(target, bl_cb)
        results['catboost_stacker'] = (auc_cb, None, bl_cb, tst_cb_acc, None)
    except Exception as e:
        print(f"  CatBoost stacker skipped: {e}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"{'Strategy':22s}  {'OOF AUC':>10s}  {'Nested CV':>10s}  {'vs anchor':>10s}")
    anchor_auc = solo_aucs[anchor]
    sorted_results = sorted(results.items(), key=lambda x: (x[1][1] or x[1][0]), reverse=True)
    for name, (auc, ncv, *_) in sorted_results:
        ncv_str = f"{ncv:.6f}" if ncv is not None else "    —   "
        delta   = auc - anchor_auc
        print(f"  {name:22s}  {auc:.6f}  {ncv_str}  {delta:+.6f}")

    # Weight breakdown for main strategies
    print("\n" + "-" * 70)
    print("WEIGHT BREAKDOWN")
    print("-" * 70)
    for strat_name in ['slsqp', 'hill_climb', 'bagged_hc']:
        if strat_name in results:
            w = results[strat_name][4]
            if w is not None:
                print(f"\n  [{strat_name}]")
                for m, ww in zip(names, w):
                    print(f"    {m:25s}  {ww:.4f}")

    # ── Best submission ──────────────────────────────────────────────────────
    # Prefer nested CV score if available, else OOF AUC
    best_name, (best_oof_auc, best_ncv, best_oof_blend, best_test_blend, best_w) = sorted_results[0]
    print("\n" + "=" * 70)
    print(f"BEST STRATEGY: {best_name}")
    print(f"  OOF AUC    : {best_oof_auc:.6f}")
    if best_ncv:
        print(f"  Nested CV  : {best_ncv:.6f}")
    print("=" * 70)

    os.makedirs('./preds', exist_ok=True)
    sub_best = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': best_test_blend})
    sub_best.to_csv('./preds/submission_v3_best.csv', index=False)
    print(f"Saved: ./preds/submission_v3_best.csv")

    # Also save bagged HC as a safe second choice
    if 'bagged_hc' in results:
        _, _, _, tst_bag, _ = results['bagged_hc']
        sub_bag = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': tst_bag})
        sub_bag.to_csv('./preds/submission_v3_bagged_hc.csv', index=False)
        print(f"Saved: ./preds/submission_v3_bagged_hc.csv")

    # Save best OOF
    np.save('./preds/oof_ensemble_v3.npy', best_oof_blend.astype(np.float32))
    print(f"Saved: ./preds/oof_ensemble_v3.npy")

    # ── Final 2-submission strategy ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDED 2-SUBMISSION STRATEGY (per Georgy Mamarin's advice):")
    print("  Slot 1: Best strategy submission  (highest OOF/nested CV AUC)")
    print("  Slot 2: Bagged HC submission      (most variance-reduced)")
    print("  → One in each slot covers both directions of uncertainty")
    print("=" * 70)


if __name__ == '__main__':
    run_ensemble()
