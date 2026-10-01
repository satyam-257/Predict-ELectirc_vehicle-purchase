"""
features_dl.py — DL-native feature pipeline (genuinely different from features_v2.py)

Key design choices vs features_v2.py / features.py:
  1. Raw continuous numerics (NOT log-transformed) — let the network learn non-linearities
  2. Quantile-binned versions of every numeric (0-9 buckets) — threshold effects
  3. Supervised target-encoding for categoricals (cross-validated, no leakage)
  4. Pairwise interaction features not in GBDT pipeline
  5. NO generator digit artifacts (those are GBDT-specific signal)
  6. Proper z-score normalization output for NN consumption
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer


def engineer_features_dl(train_path='./data/train.csv', test_path='./data/test.csv',
                          n_quantile_bins=10, target_enc_folds=5, random_state=42):
    """
    Returns:
        X_train_num: np.ndarray (n_train, n_num_features)
        X_test_num:  np.ndarray (n_test,  n_num_features)
        X_train_cat: np.ndarray (n_train, n_cat_features)  — integer-encoded
        X_test_cat:  np.ndarray (n_test,  n_cat_features)
        cat_cardinalities: list[int]
        feature_names_num: list[str]
        feature_names_cat: list[str]
        target: np.ndarray
        test_ids: np.ndarray
    """
    print("=" * 65)
    print("DL FEATURE PIPELINE (genuinely different view from GBDTs)")
    print("=" * 65)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    target   = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    test_ids = test['id'].values
    n_train  = len(train)

    # ------------------------------------------------------------------ #
    # 1. Concat for unsupervised transforms (no target leakage here)
    # ------------------------------------------------------------------ #
    df = pd.concat([train.drop(columns=['id', 'Will_Buy_EV']),
                    test.drop(columns=['id'])],
                   axis=0, ignore_index=True)

    # ------------------------------------------------------------------ #
    # 2. Raw ordinal / binary encodings  (different from features_v2)
    # ------------------------------------------------------------------ #
    range_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['range_anx_ord']   = df['Range_Anxiety_Level'].map(range_map).fillna(0).astype(np.float32)
    df['subsidy_bin']     = (df['Subsidy_Available'] == 'Yes').astype(np.float32)
    df['home_charge_bin'] = (df['Home_Charging_Possible'] == 'Yes').astype(np.float32)
    df['gender_bin']      = (df['Gender'] == 'Male').astype(np.float32)

    city_map = {'Urban': 2, 'Suburban': 1, 'Rural': 0}
    df['city_ord'] = df['City_Type'].map(city_map).fillna(0).astype(np.float32)

    car_map = {'Electric': 3, 'Hybrid': 2, 'Gasoline': 1, 'Diesel': 0}
    df['car_ord'] = df['Current_Car_Type'].map(car_map).fillna(1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 3. Raw continuous features (let NN learn the non-linearities)
    # ------------------------------------------------------------------ #
    cont_cols = [
        'Annual_Income_USD', 'Daily_Commute_km', 'Age',
        'Environmental_Concern_Level',
        'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
        'Number_of_Cars_Owned',
    ]
    for c in cont_cols:
        df[c] = df[c].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4. Pairwise / domain features NOT replicated in GBDT pipeline
    # ------------------------------------------------------------------ #
    df['charge_diff']        = df['Charging_Stations_Near_Work'] - df['Charging_Stations_Near_Home']
    df['charge_total']       = df['Charging_Stations_Near_Home'] + df['Charging_Stations_Near_Work']
    df['charge_home_frac']   = (df['Charging_Stations_Near_Home'] + 1) / (df['charge_total'] + 2)
    df['commute_x_anx']      = df['Daily_Commute_km'] * (df['range_anx_ord'] + 1)
    df['income_x_env']       = df['Annual_Income_USD'] / 100_000.0 * df['Environmental_Concern_Level']
    df['income_x_subsidy']   = df['Annual_Income_USD'] / 100_000.0 * df['subsidy_bin']
    df['env_x_subsidy']      = df['Environmental_Concern_Level'] * df['subsidy_bin']
    df['age_x_income']       = df['Age'] * df['Annual_Income_USD'] / 1_000_000.0
    df['cars_x_income']      = df['Number_of_Cars_Owned'] * df['Annual_Income_USD'] / 100_000.0
    df['charge_per_commute'] = df['charge_total'] / (df['Daily_Commute_km'] + 1.0)

    # Fable 5.1 buy-score (raw, single feature — NN can learn from it directly)
    df['buy_score_raw'] = (
        1.2 * (df['Annual_Income_USD'] / 100_000.0) +
        0.6 * df['Environmental_Concern_Level'] +
        2.0 * df['subsidy_bin'] -
        1.0 * (df['range_anx_ord'] == 1).astype(float) -
        3.0 * (df['range_anx_ord'] == 2).astype(float)
    )
    df['buy_score_centered'] = df['buy_score_raw'] - 5.5  # signed distance from threshold

    # ------------------------------------------------------------------ #
    # 5. Quantile bins (0..n_quantile_bins-1) for every continuous col
    #    — helps NN learn threshold effects similar to tree splits
    # ------------------------------------------------------------------ #
    bin_source_cols = cont_cols + [
        'charge_diff', 'charge_total', 'commute_x_anx',
        'income_x_env', 'buy_score_raw',
    ]
    for c in bin_source_cols:
        vals = df[c].values.reshape(-1, 1)
        bins = pd.qcut(df[c], q=n_quantile_bins, labels=False, duplicates='drop')
        df[f'{c}_qbin'] = bins.fillna(0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 6. Categorical columns for embedding layer
    # ------------------------------------------------------------------ #
    raw_cat_cols = [
        'Gender', 'City_Type', 'Current_Car_Type',
        'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level',
    ]

    # Interaction categoricals (string concat — kept as separate tokens)
    df['city_x_car']    = df['City_Type']   + '||' + df['Current_Car_Type']
    df['sub_x_anx']     = df['Subsidy_Available'] + '||' + df['Range_Anxiety_Level']
    df['home_x_anx']    = df['Home_Charging_Possible'] + '||' + df['Range_Anxiety_Level']
    df['city_x_sub']    = df['City_Type']   + '||' + df['Subsidy_Available']
    df['car_x_anx']     = df['Current_Car_Type'] + '||' + df['Range_Anxiety_Level']

    embed_cat_cols = raw_cat_cols + ['city_x_car', 'sub_x_anx', 'home_x_anx', 'city_x_sub', 'car_x_anx']

    cat_encoders = {}
    for c in embed_cat_cols:
        vals = df[c].astype(str)
        uniq = sorted(vals.unique())
        mapping = {v: i + 1 for i, v in enumerate(uniq)}  # 0 = unknown
        cat_encoders[c] = mapping
        df[f'{c}_enc'] = vals.map(mapping).fillna(0).astype(np.int64)

    # ------------------------------------------------------------------ #
    # 7. Supervised target-encoding (cross-validated on train rows only)
    #    Vectorized via pandas groupby — fast on 668k rows
    # ------------------------------------------------------------------ #
    te_cols = ['City_Type', 'Current_Car_Type', 'Range_Anxiety_Level',
               'city_x_car', 'sub_x_anx', 'home_x_anx', 'city_x_sub']

    train_df = df.iloc[:n_train].copy()
    test_df  = df.iloc[n_train:].copy()

    global_mean = float(target.mean())
    te_skf = StratifiedKFold(n_splits=target_enc_folds, shuffle=True, random_state=random_state)

    for c in te_cols:
        train_col = train_df[c].astype(str).values
        test_col  = test_df[c].astype(str).values
        te_oof    = np.full(n_train, global_mean, dtype=np.float32)

        for tr_idx, va_idx in te_skf.split(np.zeros(n_train), target):
            # Vectorized: groupby mean on training slice
            s = pd.Series(target[tr_idx], index=train_col[tr_idx])
            fold_means = s.groupby(level=0).mean()
            # Map to val indices, fill missing with global mean
            te_oof[va_idx] = pd.Series(train_col[va_idx]).map(fold_means).fillna(global_mean).values

        # Test TE: full train groupby mean
        s_full = pd.Series(target, index=train_col)
        full_means = s_full.groupby(level=0).mean()
        te_test = pd.Series(test_col).map(full_means).fillna(global_mean).values.astype(np.float32)

        train_df[f'{c}_te'] = te_oof
        test_df[f'{c}_te']  = te_test

    # ------------------------------------------------------------------ #
    # 8. Collect numeric feature matrix
    # ------------------------------------------------------------------ #
    num_feature_names = (
        cont_cols +
        ['range_anx_ord', 'subsidy_bin', 'home_charge_bin', 'gender_bin', 'city_ord', 'car_ord'] +
        ['charge_diff', 'charge_total', 'charge_home_frac', 'commute_x_anx',
         'income_x_env', 'income_x_subsidy', 'env_x_subsidy',
         'age_x_income', 'cars_x_income', 'charge_per_commute',
         'buy_score_raw', 'buy_score_centered'] +
        [f'{c}_qbin' for c in bin_source_cols] +
        [f'{c}_te' for c in te_cols]
    )
    # deduplicate while preserving order
    seen = set()
    num_feature_names_dedup = []
    for n in num_feature_names:
        if n not in seen:
            seen.add(n)
            num_feature_names_dedup.append(n)
    num_feature_names = num_feature_names_dedup

    cat_feature_names = [f'{c}_enc' for c in embed_cat_cols]
    cat_cardinalities = [len(cat_encoders[c]) + 1 for c in embed_cat_cols]  # +1 for unknown

    X_train_num = train_df[num_feature_names].fillna(0).values.astype(np.float32)
    X_test_num  = test_df[num_feature_names].fillna(0).values.astype(np.float32)
    X_train_cat = train_df[cat_feature_names].values.astype(np.int64)
    X_test_cat  = test_df[cat_feature_names].values.astype(np.int64)

    # ------------------------------------------------------------------ #
    # 9. Z-score normalise numerics (fit on train, apply to test)
    # ------------------------------------------------------------------ #
    qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=random_state)
    X_train_num = qt.fit_transform(X_train_num).astype(np.float32)
    X_test_num  = qt.transform(X_test_num).astype(np.float32)

    print(f"Numeric features : {len(num_feature_names)}")
    print(f"Embedding features: {len(cat_feature_names)}  (cardinalities: {cat_cardinalities})")
    print(f"Train: {X_train_num.shape}   Test: {X_test_num.shape}")

    return (X_train_num, X_test_num,
            X_train_cat, X_test_cat,
            cat_cardinalities,
            num_feature_names, cat_feature_names,
            target, test_ids)


if __name__ == '__main__':
    out = engineer_features_dl()
    print("Done. Shapes:", out[0].shape, out[1].shape)
