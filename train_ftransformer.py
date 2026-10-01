"""
train_ftransformer.py — FT-Transformer for S6E9 EV Adoption

Architecture: Feature Tokenizer + Transformer (Gorishniy et al. 2021)
  - Every feature (numeric + categorical) becomes a learned d-dim token
  - Multi-head self-attention over feature tokens finds interactions
  - Structurally different from gradient boosted trees
  - 10-fold CV, 3 random seeds averaged → robust OOF

Why this helps:
  - GBDTs on features_v2.py are 0.992-0.999 correlated → redundant
  - FT-Transformer on features_dl.py (different view) targets ρ < 0.985
  - 668k rows: DL needs enough data to shine — this dataset qualifies

Usage:
    python train_ftransformer.py
    python train_ftransformer.py --folds 5 --seeds 1  # quick test
"""

import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from features_dl import engineer_features_dl


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# FT-Transformer Components
# ─────────────────────────────────────────────────────────────────────────────
class NumericTokenizer(nn.Module):
    """Projects each numeric feature to a d-dim token via a learned weight."""
    def __init__(self, n_num: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_num, d_token))
        self.bias   = nn.Parameter(torch.empty(n_num, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_num)  → (B, n_num, d_token)
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CatTokenizer(nn.Module):
    """Separate embedding table per categorical feature."""
    def __init__(self, cardinalities: list[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card + 1, d_token, padding_idx=0)
            for card in cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_cat) → (B, n_cat, d_token)
        return torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block (attention + FFN)."""
    def __init__(self, d_token: int, n_heads: int, ffn_factor: float = 4/3, dropout: float = 0.1):
        super().__init__()
        assert d_token % n_heads == 0
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        d_ffn      = max(int(d_token * ffn_factor), d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_tokens, d_token)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer.
    Gorishniy et al. (2021) — https://arxiv.org/abs/2106.11959
    """
    def __init__(self, n_num: int, cat_cardinalities: list[int],
                 d_token: int = 192, n_heads: int = 8, n_layers: int = 3,
                 ffn_factor: float = 4/3, dropout: float = 0.1):
        super().__init__()
        self.num_tok = NumericTokenizer(n_num, d_token)
        self.cat_tok = CatTokenizer(cat_cardinalities, d_token)
        # [CLS] token
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, ffn_factor, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_token)
        self.head     = nn.Linear(d_token, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        # Tokenise
        tok_num = self.num_tok(x_num)            # (B, n_num, d)
        tok_cat = self.cat_tok(x_cat)            # (B, n_cat, d)
        cls     = self.cls_token.expand(x_num.size(0), -1, -1)  # (B, 1, d)

        x = torch.cat([cls, tok_num, tok_cat], dim=1)  # (B, 1+n_num+n_cat, d)
        for blk in self.blocks:
            x = blk(x)

        # Use CLS token representation for classification
        cls_out = self.norm_out(x[:, 0])         # (B, d)
        return self.head(cls_out).squeeze(-1)    # (B,)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_one_fold(model, train_loader, val_num, val_cat, val_y,
                   epochs: int, lr: float, weight_decay: float, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc   = 0.0
    best_preds = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xn, xc)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.inference_mode():
            vn = val_num.to(device)
            vc = val_cat.to(device)
            # batch to avoid OOM on CPU
            preds_list = []
            bs = 16_384
            for i in range(0, vn.size(0), bs):
                logits = model(vn[i:i+bs], vc[i:i+bs])
                preds_list.append(torch.sigmoid(logits).cpu().numpy())
            val_preds = np.concatenate(preds_list)

        val_auc = roc_auc_score(val_y, val_preds)
        if val_auc > best_auc:
            best_auc   = val_auc
            best_preds = val_preds
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:02d}/{epochs:02d}  loss={total_loss/len(train_loader):.4f}  val_auc={val_auc:.6f}  best={best_auc:.6f}")

    model.load_state_dict(best_state)
    return best_auc, best_preds, model


def predict_test(model, X_test_num: np.ndarray, X_test_cat: np.ndarray,
                 device, batch_size: int = 16_384) -> np.ndarray:
    model.eval()
    tn = torch.tensor(X_test_num)
    tc = torch.tensor(X_test_cat)
    preds = []
    with torch.inference_mode():
        for i in range(0, tn.size(0), batch_size):
            logits = model(tn[i:i+batch_size].to(device), tc[i:i+batch_size].to(device))
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run(n_folds: int = 10, seeds: list[int] = None,
        d_token: int = 192, n_heads: int = 8, n_layers: int = 3,
        dropout: float = 0.1, epochs: int = 20,
        batch_size: int = 4096, lr: float = 1e-3, weight_decay: float = 1e-4):

    if seeds is None:
        seeds = [42, 2024, 7]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load features (once, reused across seeds)
    (X_train_num, X_test_num,
     X_train_cat, X_test_cat,
     cat_cardinalities,
     num_feat_names, cat_feat_names,
     target, test_ids) = engineer_features_dl()

    n_num = X_train_num.shape[1]
    n_train = len(target)

    oof_accum  = np.zeros(n_train,       dtype=np.float64)
    test_accum = np.zeros(len(test_ids), dtype=np.float64)
    n_runs     = 0

    for seed in seeds:
        seed_everything(seed)
        print(f"\n{'='*65}")
        print(f"SEED {seed}")
        print(f"{'='*65}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        oof_seed  = np.zeros(n_train,       dtype=np.float64)
        test_seed = np.zeros(len(test_ids), dtype=np.float64)

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_num, target)):
            t0 = time.time()
            print(f"\n  Fold {fold+1}/{n_folds}  (seed={seed})")

            xn_tr = torch.tensor(X_train_num[tr_idx])
            xc_tr = torch.tensor(X_train_cat[tr_idx])
            y_tr  = torch.tensor(target[tr_idx].astype(np.float32))

            xn_va = torch.tensor(X_train_num[va_idx])
            xc_va = torch.tensor(X_train_cat[va_idx])
            y_va  = target[va_idx]

            train_ds = TensorDataset(xn_tr, xc_tr, y_tr)
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, drop_last=False, num_workers=0)

            model = FTTransformer(
                n_num=n_num,
                cat_cardinalities=cat_cardinalities,
                d_token=d_token,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            ).to(device)

            best_auc, best_preds, model = train_one_fold(
                model, train_loader,
                xn_va, xc_va, y_va,
                epochs=epochs, lr=lr, weight_decay=weight_decay, device=device
            )

            oof_seed[va_idx] = best_preds
            test_seed        += predict_test(model, X_test_num, X_test_cat, device) / n_folds

            elapsed = time.time() - t0
            print(f"  ✓ Fold {fold+1} best_auc={best_auc:.6f}  ({elapsed:.0f}s)")

        seed_oof_auc = roc_auc_score(target, oof_seed)
        print(f"\nSeed {seed} OOF AUC: {seed_oof_auc:.6f}")

        oof_accum  += oof_seed
        test_accum += test_seed
        n_runs     += 1

    # Average across seeds
    oof_final  = oof_accum  / n_runs
    test_final = test_accum / n_runs

    final_auc = roc_auc_score(target, oof_final)
    print(f"\n{'='*65}")
    print(f"FINAL OOF AUC ({n_runs} seeds × {n_folds} folds): {final_auc:.6f}")
    print(f"{'='*65}")

    # Correlation check
    from scipy.stats import spearmanr
    import os
    print("\nSpearman ρ vs existing OOFs:")
    preds_dir = './preds'
    for f in sorted(os.listdir(preds_dir)):
        if f.startswith('oof_') and f.endswith('.npy') and 'ftransformer' not in f:
            arr = np.load(f'{preds_dir}/{f}')
            if len(arr) == n_train:
                r, _ = spearmanr(oof_final, arr)
                print(f"  {f:<35}  ρ={r:.4f}")

    # Save
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_ftransformer.npy', oof_final.astype(np.float32))
    np.save('./preds/test_ftransformer.npy', test_final.astype(np.float32))
    print("\nSaved: ./preds/oof_ftransformer.npy")
    print("Saved: ./preds/test_ftransformer.npy")

    return final_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds',       type=int,   default=10)
    parser.add_argument('--seeds',       type=int,   nargs='+', default=[42, 2024, 7])
    parser.add_argument('--d_token',     type=int,   default=192)
    parser.add_argument('--n_heads',     type=int,   default=8)
    parser.add_argument('--n_layers',    type=int,   default=3)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--epochs',      type=int,   default=20)
    parser.add_argument('--batch_size',  type=int,   default=4096)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    args = parser.parse_args()

    run(
        n_folds=args.folds,
        seeds=args.seeds,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
