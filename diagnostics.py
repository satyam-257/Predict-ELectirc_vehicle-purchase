import numpy as np
import pandas as pd
import os
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, ks_2samp

def run_diagnostics():
    print("=" * 70)
    print("S6E9 BEYOND BOOSTING: MODEL DIAGNOSTICS & RESIDUAL ANALYSIS")
    print("=" * 70)

    train_path = './data/train.csv'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    train_df = pd.read_csv(train_path)
    target = (train_df['Will_Buy_EV'] == 'Yes').astype(int).values

    model_candidates = [
        ('LightGBM', './preds/oof_lgbm.npy'),
        ('XGBoost', './preds/oof_xgb.npy'),
        ('CatBoost', './preds/oof_catboost.npy'),
        ('NeuralNet', './preds/oof_nn.npy')
    ]

    models = []
    oofs = []

    for name, path in model_candidates:
        if os.path.exists(path):
            arr = np.load(path)
            auc = roc_auc_score(target, arr)
            models.append(name)
            oofs.append(arr)
            print(f"[LOADED] {name:12s} | OOF ROC-AUC: {auc:.6f}")

    if not models:
        print("No OOF files found in ./preds. Run model training first.")
        return

    n_models = len(models)
    oof_matrix = np.column_stack(oofs)

    # 1. Pearson & Spearman Rank Correlations
    print("\n" + "-" * 70)
    print("1. MODEL PREDICTION CORRELATION MATRICES")
    print("-" * 70)
    
    print("\nPearson Correlation:")
    corr_df = pd.DataFrame(np.corrcoef(oof_matrix.T), index=models, columns=models)
    print(corr_df.round(5).to_string())

    print("\nSpearman Rank Correlation:")
    spearman_mat = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            spearman_mat[i, j], _ = spearmanr(oof_matrix[:, i], oof_matrix[:, j])
    spearman_df = pd.DataFrame(spearman_mat, index=models, columns=models)
    print(spearman_df.round(5).to_string())

    # 2. Kolmogorov-Smirnov (KS) Statistics
    print("\n" + "-" * 70)
    print("2. KOLMOGOROV-SMIRNOV (KS) TWO-SAMPLE CDF DIVERGENCE")
    print("(Higher KS statistic indicates more distinct prediction distributions)")
    print("-" * 70)
    ks_mat = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            stat, _ = ks_2samp(oof_matrix[:, i], oof_matrix[:, j])
            ks_mat[i, j] = stat
    ks_df = pd.DataFrame(ks_mat, index=models, columns=models)
    print(ks_df.round(5).to_string())

    # 3. Oracle Self-Consistency Test (Abu Kayyisah Basyid ceiling estimation)
    print("\n" + "-" * 70)
    print("3. ORACLE NOISE CEILING ESTIMATION (Abu Kayyisah Basyid method)")
    print("-" * 70)
    mean_oof = np.mean(oof_matrix, axis=1)
    mean_oof_auc = roc_auc_score(target, mean_oof)
    print(f"Current Ensemble Mean OOF AUC: {mean_oof_auc:.6f}")

    # Clip probabilities to avoid extreme 0 or 1
    p_clipped = np.clip(mean_oof, 1e-6, 1.0 - 1e-6)
    rng = np.random.RandomState(42)
    n_simulations = 30
    sim_aucs = []

    for _ in range(n_simulations):
        # Sample synthetic binary outcomes from fitted probabilities
        y_sim = rng.binomial(1, p_clipped)
        sim_auc = roc_auc_score(y_sim, mean_oof)
        sim_aucs.append(sim_auc)

    mean_oracle_auc = np.mean(sim_aucs)
    std_oracle_auc = np.std(sim_aucs)
    print(f"Estimated Theoretical Oracle Ceiling: {mean_oracle_auc:.6f} +/- {std_oracle_auc:.6f}")
    if abs(mean_oracle_auc - mean_oof_auc) < 0.002:
        print(">> FINDING: Current ensemble is near the theoretical noise limit of the data.")
        print(">> To improve, the model must access non-linear/neural diversity or orthogonal feature representations.")

    # 4. Confident Failure Analysis (Blind Spots)
    print("\n" + "-" * 70)
    print("4. CONFIDENT ERROR & RESIDUAL ANALYSIS (Tree Model Blind Spots)")
    print("-" * 70)
    residuals = np.abs(target - mean_oof)
    worst_1pct_threshold = np.percentile(residuals, 99)
    hard_cases = train_df[residuals >= worst_1pct_threshold].copy()
    hard_cases['target'] = target[residuals >= worst_1pct_threshold]
    hard_cases['mean_pred'] = mean_oof[residuals >= worst_1pct_threshold]
    hard_cases['residual'] = residuals[residuals >= worst_1pct_threshold]

    print(f"Identified {len(hard_cases)} cases with residual >= {worst_1pct_threshold:.4f} (Worst 1%).")
    
    # Analyze by key dimensions
    print("\nBlind spots by Subsidy & Range Anxiety:")
    pivot = hard_cases.groupby(['Subsidy_Available', 'Range_Anxiety_Level']).size().unstack(fill_value=0)
    print(pivot)

    print("\nDistribution of Actual Target among top blind spots:")
    print(hard_cases['target'].value_counts().to_dict())

    print("\n" + "=" * 70)
    print("Diagnostics complete.")
    print("=" * 70)

if __name__ == '__main__':
    run_diagnostics()
