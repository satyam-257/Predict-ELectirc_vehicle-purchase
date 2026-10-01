"""
ensemble_diagnostics.py — Ensemble health checks from the Kaggle discussion

Implements the three checks from abukayyisahbasyid + tilii7 + amirhosseinkarimiee:

1. Spearman correlation matrix + KS statistics  (are models complementary?)
2. OOF residual analysis  (confidently-wrong rows — shared weakness?)
3. Nested 5-fold CV on hill-climb  (is weight selection overfit?)
4. Bagged hill-climb × 40  (estimation variance check)
5. Oracle AUC self-consistency  (are we at the label ceiling?)

Usage:
    python ensemble_diagnostics.py
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr, ks_2samp
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
# Load all available OOF files
# ─────────────────────────────────────────────────────────────────────────────
def load_oofs(preds_dir='./preds'):
    train = pd.read_csv('./data/train.csv')
    target = (train['Will_Buy_EV'] == 'Yes').astype(int).values

    oofs, names = [], []
    for f in sorted(os.listdir(preds_dir)):
        if f.startswith('oof_') and f.endswith('.npy'):
            arr = np.load(f'{preds_dir}/{f}')
            if len(arr) == len(target):
                oofs.append(arr)
                names.append(f.replace('oof_', '').replace('.npy', ''))

    oof_mat = np.column_stack(oofs)
    return target, names, oof_mat


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spearman correlation + KS stat
# ─────────────────────────────────────────────────────────────────────────────
def check_correlations(names, oof_mat):
    print("\n" + "=" * 70)
    print("CHECK 1: SPEARMAN CORRELATION MATRIX")
    print("=" * 70)
    n = len(names)
    header = f"{'':22s}" + "".join(f"{n[:10]:>12s}" for n in names)
    print(header)
    for i in range(n):
        row = f"{names[i]:22s}"
        for j in range(n):
            r, _ = spearmanr(oof_mat[:, i], oof_mat[:, j])
            flag = " !" if (i != j and r > 0.995) else "  "
            row += f"{r:>10.4f}{flag}"
        print(row)

    print("\n  (!) means rho > 0.995 -> near-redundant models")

    print("\n" + "-" * 70)
    print("KS STATISTICS (high = more complementary, >0.02 is meaningful)")
    print("-" * 70)
    anchor = 0  # compare all vs first model (anchor)
    for j in range(1, n):
        ks, _ = ks_2samp(oof_mat[:, anchor], oof_mat[:, j])
        r, _  = spearmanr(oof_mat[:, anchor], oof_mat[:, j])
        print(f"  {names[anchor]:20s} vs {names[j]:20s}  KS={ks:.4f}  rho={r:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. OOF Residual Analysis  (tilii7's method)
# ─────────────────────────────────────────────────────────────────────────────
def check_residuals(target, names, oof_mat, top_n=20):
    print("\n" + "=" * 70)
    print("CHECK 2: OOF RESIDUAL ANALYSIS (confidently-wrong rows)")
    print("=" * 70)

    for i, name in enumerate(names):
        preds = oof_mat[:, i]
        residuals = preds - target  # positive = predicted 1 when actually 0

        # Most confidently-wrong positives (pred high, true=0)
        wrong_pos_idx = np.argsort(residuals)[::-1][:top_n]
        # Most confidently-wrong negatives (pred low, true=1)
        wrong_neg_idx = np.argsort(residuals)[:top_n]

        auc = roc_auc_score(target, preds)
        mean_wrong_pos = residuals[wrong_pos_idx].mean()
        mean_wrong_neg = residuals[wrong_neg_idx].mean()

        print(f"\n  [{name}]  OOF_AUC={auc:.6f}")
        print(f"    Top-{top_n} confident FALSE POSITIVES: mean(pred-true) = {mean_wrong_pos:.4f}")
        print(f"    Top-{top_n} confident FALSE NEGATIVES: mean(pred-true) = {mean_wrong_neg:.4f}")

    # Find rows where ALL models agree and are wrong
    best_oof = oof_mat.mean(axis=1)
    ensemble_residuals = best_oof - target
    most_wrong = np.argsort(np.abs(ensemble_residuals))[::-1][:10]
    print(f"\n  Top-10 rows where ensemble is most wrong:")
    print(f"  idx={most_wrong.tolist()}")
    print(f"  |residual|={np.abs(ensemble_residuals[most_wrong]).round(3).tolist()}")
    print(f"  true_labels={target[most_wrong].tolist()}")
    print(f"  ensemble_pred={best_oof[most_wrong].round(3).tolist()}")


# ─────────────────────────────────────────────────────────────────────────────
# Caruana Hill Climbing
# ─────────────────────────────────────────────────────────────────────────────
def caruana_hill_climb(oof_mat, target, max_steps=100):
    n_models = oof_mat.shape[1]
    single_aucs = [roc_auc_score(target, oof_mat[:, i]) for i in range(n_models)]
    best_init = np.argmax(single_aucs)
    ensemble = [best_init]
    current_pred = oof_mat[:, best_init].copy()
    best_auc = single_aucs[best_init]

    for step in range(2, max_steps + 1):
        best_candidate, best_cand_auc, best_cand_pred = None, -1, None
        for m in range(n_models):
            trial_pred = (current_pred * (step - 1) + oof_mat[:, m]) / step
            trial_auc  = roc_auc_score(target, trial_pred)
            if trial_auc > best_cand_auc:
                best_cand_auc  = trial_auc
                best_candidate = m
                best_cand_pred = trial_pred

        if best_cand_auc >= best_auc:
            best_auc     = best_cand_auc
            current_pred = best_cand_pred
            ensemble.append(best_candidate)
        else:
            break

    counts  = np.bincount(ensemble, minlength=n_models)
    weights = counts / len(ensemble)
    return weights, best_auc, current_pred


# ─────────────────────────────────────────────────────────────────────────────
# 3. Nested 5-fold CV on hill-climb
# ─────────────────────────────────────────────────────────────────────────────
def check_nested_cv(target, names, oof_mat, n_outer=5):
    print("\n" + "=" * 70)
    print("CHECK 3: NESTED 5-FOLD CV ON HILL-CLIMB (is weight selection overfit?)")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)
    held_out_aucs = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(oof_mat, target)):
        weights, _, _ = caruana_hill_climb(oof_mat[tr_idx], target[tr_idx])
        blend_va = np.dot(oof_mat[va_idx], weights)
        auc_va   = roc_auc_score(target[va_idx], blend_va)
        held_out_aucs.append(auc_va)
        print(f"  Outer fold {fold+1}: held-out AUC = {auc_va:.6f}  weights={weights.round(3).tolist()}")

    # Full-data hill climb for comparison
    weights_full, auc_full, _ = caruana_hill_climb(oof_mat, target)
    nested_mean = np.mean(held_out_aucs)

    print(f"\n  Nested CV mean AUC  : {nested_mean:.6f}")
    print(f"  Full-data HC AUC    : {auc_full:.6f}")
    print(f"  Overfit gap         : {auc_full - nested_mean:+.6f}")
    if abs(auc_full - nested_mean) < 1e-4:
        print("  [OK] Gap < 1e-4 -> weight selection is NOT overfit")
    else:
        print("  [!!] Gap > 1e-4 -> some overfitting to validation folds")

    return nested_mean, auc_full, weights_full


# ─────────────────────────────────────────────────────────────────────────────
# 4. Bagged hill-climb  (40 bootstraps)
# ─────────────────────────────────────────────────────────────────────────────
def check_bagged_hc(target, names, oof_mat, n_bags=40, rng_seed=42):
    print("\n" + "=" * 70)
    print(f"CHECK 4: BAGGED HILL-CLIMB (n_bags={n_bags})")
    print("=" * 70)
    rng = np.random.RandomState(rng_seed)
    n   = len(target)

    bag_weights = []
    for b in range(n_bags):
        idx = rng.choice(n, size=n, replace=True)
        w, _, _ = caruana_hill_climb(oof_mat[idx], target[idx])
        bag_weights.append(w)

    bag_weights = np.array(bag_weights)  # (n_bags, n_models)
    mean_w = bag_weights.mean(axis=0)
    std_w  = bag_weights.std(axis=0)

    # Score with bagged weights on full OOF
    bagged_blend = np.dot(oof_mat, mean_w / mean_w.sum())
    bagged_auc   = roc_auc_score(target, bagged_blend)

    # Full-data HC for comparison
    _, full_hc_auc, full_hc_pred = caruana_hill_climb(oof_mat, target)

    print(f"  Bagged HC AUC   : {bagged_auc:.6f}")
    print(f"  Single HC AUC   : {full_hc_auc:.6f}")
    print(f"  Difference      : {bagged_auc - full_hc_auc:+.6f}")
    print(f"\n  Per-model weight statistics:")
    print(f"  {'Model':20s}  {'Mean_w':>8s}  {'Std_w':>8s}")
    for name, mw, sw in zip(names, mean_w, std_w):
        print(f"  {name:20s}  {mw:8.4f}  {sw:8.4f}")

    if abs(bagged_auc - full_hc_auc) < 1e-4:
        print("\n  [OK] Bagging does not move the AUC -> no estimation variance in weights")
    else:
        print("\n  [!!] Bagging changes AUC -> some instability in weight estimation")

    return bagged_auc, mean_w


# ─────────────────────────────────────────────────────────────────────────────
# 5. Oracle AUC self-consistency  (abukayyisahbasyid's check)
# ─────────────────────────────────────────────────────────────────────────────
def check_oracle_auc(target, names, oof_mat, n_samples=10, rng_seed=42):
    """
    For each model's calibrated probabilities p_i, simulate Y' ~ Bernoulli(p_i).
    The oracle AUC on Y' = best any model could do IF p_i were ground truth.
    
    If real AUC ≈ oracle AUC → no headroom; if real AUC << oracle → headroom exists.
    
    Note (per Georgy Mamarin): this measures dispersion/calibration, not headroom directly.
    A properly calibrated good model and a poorly calibrated weak model can read similarly.
    Interpret alongside other checks.
    """
    print("\n" + "=" * 70)
    print("CHECK 5: ORACLE AUC SELF-CONSISTENCY")
    print("  (oracle = AUC of predicted probs against simulated labels from those probs)")
    print("=" * 70)

    rng = np.random.RandomState(rng_seed)

    for i, name in enumerate(names):
        p = np.clip(oof_mat[:, i], 1e-6, 1 - 1e-6)
        real_auc = roc_auc_score(target, p)

        # Average oracle AUC over n_samples draws
        oracle_aucs = []
        for _ in range(n_samples):
            y_sim = rng.binomial(1, p)
            if y_sim.sum() > 0 and y_sim.sum() < len(y_sim):
                oracle_aucs.append(roc_auc_score(y_sim, p))
        oracle_auc = np.mean(oracle_aucs)

        gap = real_auc - oracle_auc
        print(f"  {name:25s}  real_AUC={real_auc:.6f}  oracle_AUC={oracle_auc:.6f}  gap={gap:+.6f}")

    # Best ensemble OOF
    best_blend = oof_mat.mean(axis=1)
    best_real  = roc_auc_score(target, best_blend)
    p_blend    = np.clip(best_blend, 1e-6, 1 - 1e-6)
    oracle_aucs = []
    for _ in range(n_samples):
        y_sim = rng.binomial(1, p_blend)
        if y_sim.sum() > 0 and y_sim.sum() < len(y_sim):
            oracle_aucs.append(roc_auc_score(y_sim, p_blend))
    blend_oracle = np.mean(oracle_aucs)
    print(f"  {'[ensemble_mean]':25s}  real_AUC={best_real:.6f}  oracle_AUC={blend_oracle:.6f}  gap={best_real-blend_oracle:+.6f}")
    print("  Interpretation: gap ~= 0 -> calibrated; gap >> 0 -> probe is underconfident/overdispersed")
    print("  Per Mamarin: this measures calibration dispersion, NOT remaining headroom directly.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def run_diagnostics():
    target, names, oof_mat = load_oofs()
    n_models = len(names)

    print("\n" + "=" * 70)
    print(f"S6E9 ENSEMBLE DIAGNOSTICS  ({n_models} models)")
    print("=" * 70)

    for i, name in enumerate(names):
        auc = roc_auc_score(target, oof_mat[:, i])
        print(f"  {name:30s}  OOF AUC = {auc:.6f}")

    check_correlations(names, oof_mat)
    check_residuals(target, names, oof_mat, top_n=20)
    nested_mean, full_auc, weights_full = check_nested_cv(target, names, oof_mat)
    bagged_auc, bagged_weights = check_bagged_hc(target, names, oof_mat, n_bags=40)
    check_oracle_auc(target, names, oof_mat, n_samples=20)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Full HC OOF AUC     : {full_auc:.6f}")
    print(f"  Nested CV AUC       : {nested_mean:.6f}")
    print(f"  Bagged HC AUC       : {bagged_auc:.6f}")
    print(f"\n  Full HC weights:")
    for name, w in zip(names, weights_full):
        print(f"    {name:25s}  {w:.4f}")
    print(f"\n  Bagged HC weights:")
    for name, w in zip(names, bagged_weights):
        print(f"    {name:25s}  {w:.4f}")


if __name__ == '__main__':
    run_diagnostics()
