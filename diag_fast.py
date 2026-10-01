import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

train = pd.read_csv('./data/train.csv')
target = (train['Will_Buy_EV'] == 'Yes').astype(int).values

oofs, names = [], []
for f in sorted(os.listdir('./preds')):
    if f.startswith('oof_') and f.endswith('.npy'):
        arr = np.load(f'./preds/{f}')
        if len(arr) == len(target):
            oofs.append(arr)
            names.append(f.replace('oof_','').replace('.npy',''))
oof_mat = np.column_stack(oofs)

def caruana_hc(oof_mat, target, max_steps=200):
    solo = [roc_auc_score(target, oof_mat[:,i]) for i in range(oof_mat.shape[1])]
    best = int(np.argmax(solo))
    ens = [best]; cur = oof_mat[:,best].copy(); bauc = solo[best]
    for step in range(2, max_steps+1):
        bc, ba, bp = None, -1.0, None
        for m in range(oof_mat.shape[1]):
            t = (cur*(step-1)+oof_mat[:,m])/step
            ta = roc_auc_score(target, t)
            if ta > ba:
                ba, bc, bp = ta, m, t
        if ba >= bauc:
            bauc, cur = ba, bp
            ens.append(bc)
        else:
            break
    cnt = np.bincount(ens, minlength=oof_mat.shape[1])
    return cnt/len(ens), bauc, cur

# Bagged HC (20 bags)
print('Running bagged HC (20 bags)...')
rng = np.random.RandomState(42)
N = len(target)
all_w = []
for b in range(20):
    idx = rng.choice(N, size=N, replace=True)
    w, _, _ = caruana_hc(oof_mat[idx], target[idx])
    all_w.append(w)
    print(f'  bag {b+1}/20 done', flush=True)

mw = np.array(all_w).mean(0)
mw /= mw.sum()
bag_blend = np.dot(oof_mat, mw)
bag_auc = roc_auc_score(target, bag_blend)
_, full_auc, _ = caruana_hc(oof_mat, target)

print(f'Bagged HC AUC : {bag_auc:.6f}')
print(f'Single HC AUC : {full_auc:.6f}')
print(f'Difference    : {bag_auc - full_auc:+.6f}')
print('Per-model bagged weights:')
for nm, w in zip(names, mw):
    print(f'  {nm:25s}  {w:.4f}')

# Oracle AUC (10 samples each)
print()
print('Oracle AUC check (10 samples each):')
rng2 = np.random.RandomState(99)
for i, nm in enumerate(names):
    p = np.clip(oof_mat[:,i], 1e-6, 1-1e-6)
    real = roc_auc_score(target, p)
    oracles = []
    for _ in range(10):
        ys = rng2.binomial(1, p)
        if 0 < ys.sum() < len(ys):
            oracles.append(roc_auc_score(ys, p))
    print(f'  {nm:25s}  real={real:.6f}  oracle={np.mean(oracles):.6f}  gap={real-np.mean(oracles):+.6f}')

# ensemble mean oracle
p = np.clip(oof_mat.mean(1), 1e-6, 1-1e-6)
real = roc_auc_score(target, p)
oracles = []
for _ in range(10):
    ys = rng2.binomial(1, p)
    if 0 < ys.sum() < len(ys):
        oracles.append(roc_auc_score(ys, p))
print(f'  {"[ensemble_mean]":25s}  real={real:.6f}  oracle={np.mean(oracles):.6f}  gap={real-np.mean(oracles):+.6f}')
print('Done.')
