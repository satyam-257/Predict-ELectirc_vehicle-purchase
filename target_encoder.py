import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def add_fold_safe_target_encoding(
    X_train, y_train, X_val, X_test, 
    te_cols=['City_Subsidy_Interaction', 'Sub_Anx_Group', 'City_Car_Interaction'],
    smoothing=20.0, n_inner_splits=5, random_state=42
):
    """
    Computes strictly leak-free target encodings:
    1. Inner StratifiedKFold generates OOF target encodings for X_train.
    2. Full training fold statistics (with Bayesian m-estimate smoothing) map onto X_val and X_test.
    """
    y_series = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train
    global_mean = float(y_series.mean())
    inner_skf = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
    
    X_train_te = X_train.copy()
    X_val_te = X_val.copy()
    X_test_te = X_test.copy()
    
    for col in te_cols:
        if col not in X_train.columns:
            continue
            
        col_te_name = f'{col}_TE'
        X_train_te[col_te_name] = np.nan
        
        # Step 1: Inner CV for X_train OOF (prevents memorization of high cardinality categories)
        for in_tr_idx, in_val_idx in inner_skf.split(X_train, y_series):
            in_tr_y = y_series.iloc[in_tr_idx]
            in_tr_x = X_train.iloc[in_tr_idx][col].astype(str)
            
            stats = in_tr_y.groupby(in_tr_x).agg(['count', 'mean'])
            smoothed = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            
            val_categories = X_train.iloc[in_val_idx][col].astype(str)
            X_train_te.iloc[in_val_idx, X_train_te.columns.get_loc(col_te_name)] = val_categories.map(smoothed).fillna(global_mean).values
            
        # Step 2: Full X_train statistics mapped to X_val and X_test
        full_x = X_train[col].astype(str)
        full_stats = y_series.groupby(full_x).agg(['count', 'mean'])
        full_smoothed = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean) / (full_stats['count'] + smoothing)
        
        X_val_te[col_te_name] = X_val[col].astype(str).map(full_smoothed).fillna(global_mean).values
        X_test_te[col_te_name] = X_test[col].astype(str).map(full_smoothed).fillna(global_mean).values

    return X_train_te, X_val_te, X_test_te
