import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data/train.csv', usecols=['id', 'Will_Buy_EV'])
test = pd.read_csv('data/test.csv', usecols=['id'])
y_true = train['Will_Buy_EV'].values
n_train = len(train)
n_test = len(test)

print(f"Train size: {n_train}, Test size: {n_test}")

models = {}

# 1. Heuljax XGB
oof_heul = np.load('preds/oof_heuljax_xgb.npy')
test_heul = np.load('preds/test_heuljax_xgb.npy')
models['Heuljax_XGB'] = {'oof': oof_heul, 'test': test_heul}

# 2. BlamerX XGB
oof_blam = np.load('preds/oof_blamerx_xgb.npy')
test_blam = np.load('preds/test_blamerx_xgb.npy')
models['BlamerX_XGB'] = {'oof': oof_blam, 'test': test_blam}

# 3. Zoom Zoom LGBM
oof_zoom = np.load('preds/oof_zoom_zoom_lgb.npy')
test_zoom = np.load('preds/test_zoom_zoom_lgb.npy')
models['Zoom_Zoom_LGBM'] = {'oof': oof_zoom, 'test': test_zoom}

# 4. RealMLP (Neural Net)
df_oof_rmlp = pd.read_csv('kernels_out/realmlp/oof_preds.csv')
df_sub_rmlp = pd.read_csv('kernels_out/realmlp/submission.csv')
models['RealMLP_NN'] = {'oof': df_oof_rmlp['Will_Buy_EV'].values, 'test': df_sub_rmlp['Will_Buy_EV'].values}

# 5. CatBoost Naji
df_oof_cat = pd.read_csv('kernels_out/catboost_naji/oof_CATBOOST.csv')
df_sub_cat = pd.read_csv('kernels_out/catboost_naji/submission_CATBOOST.csv')
models['CatBoost_Naji'] = {'oof': df_oof_cat.iloc[:, 1].values, 'test': df_sub_cat.iloc[:, 1].values}

# 6. XGBoost Naji
df_oof_xgbn = pd.read_csv('kernels_out/xgboost_naji/oof_XGBOOST.csv')
df_sub_xgbn = pd.read_csv('kernels_out/xgboost_naji/submission_XGBOOST.csv')
models['XGBoost_Naji'] = {'oof': df_oof_xgbn.iloc[:, 1].values, 'test': df_sub_xgbn.iloc[:, 1].values}

# 7. Aryan HistGBM
df_oof_aryan = pd.read_csv('kernels_out/aryan_histgbm/oof_proba.csv')
df_test_aryan = pd.read_csv('kernels_out/aryan_histgbm/test_proba.csv')
models['Aryan_HistGBM'] = {'oof': df_oof_aryan.iloc[:, 1].values, 'test': df_test_aryan.iloc[:, 1].values}

# 8. Mizushima LGBM
df_oof_miz = pd.read_csv('kernels_out/mizushima_lgbm/oof_predictions.csv')
df_sub_miz = pd.read_csv('kernels_out/mizushima_lgbm/submission.csv')
models['Mizushima_LGBM'] = {'oof': df_oof_miz.iloc[:, 1].values, 'test': df_sub_miz.iloc[:, 1].values}

print("\n--- Model Verification & Individual AUC ---")
for name, d in models.items():
    oof = d['oof']
    test_p = d['test']
    assert len(oof) == n_train, f"{name} oof len {len(oof)} != {n_train}"
    assert len(test_p) == n_test, f"{name} test len {len(test_p)} != {n_test}"
    assert not np.isnan(oof).any(), f"{name} oof has NaNs"
    assert not np.isnan(test_p).any(), f"{name} test has NaNs"
    auc = roc_auc_score(y_true, oof)
    print(f"{name:16s} | OOF AUC = {auc:.6f} | min={oof.min():.4f}, max={oof.max():.4f} | test min={test_p.min():.4f}, max={test_p.max():.4f}")
