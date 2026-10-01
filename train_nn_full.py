"""
train_nn_full.py
================
Trains TWO complementary tabular neural architectures for S6E9:
  1. TabularResNet  – residual blocks, entity embeddings, SiLU, LayerNorm
  2. RealMLP        – piecewise-linear numeric embeddings (Gorishniy 2023 style)
                     + gating, wider MLP, cosine warm restarts

Each is trained with 5-Fold StratifiedKFold (seed=42, matching GBDTs).
Outputs:
  preds/oof_resnet.npy   / preds/test_resnet.npy
  preds/oof_realmlp.npy  / preds/test_realmlp.npy
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from features import engineer_features
import os

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

# ──────────────────────────────────────────────
# Architecture 1: TabularResNet
# ──────────────────────────────────────────────
class TabularResNet(nn.Module):
    def __init__(self, in_dim, hidden_dims=(512, 256, 128), dropout=0.15):
        super().__init__()
        dims = [in_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.blocks.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dims[i + 1], dims[i + 1]),
            ))
            # Projection shortcut when dims differ
            if dims[i] != dims[i + 1]:
                self.blocks[-1].add_module('skip_proj', nn.Linear(dims[i], dims[i + 1], bias=False))
        self.res_norms = nn.ModuleList([nn.LayerNorm(d) for d in dims[1:]])
        self.res_acts = nn.ModuleList([nn.SiLU() for _ in dims[1:]])
        self.head = nn.Linear(dims[-1], 1)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            layers = list(block.children())
            skip_proj = None
            if hasattr(layers[-1], 'weight') and 'skip_proj' in [n for n, _ in block.named_children()]:
                skip_proj = layers[-1]
                main_layers = layers[:-1]
            else:
                main_layers = layers

            residual = x
            for layer in main_layers:
                x = layer(x)
            if skip_proj is not None:
                residual = skip_proj(residual)
            x = self.res_acts[i](self.res_norms[i](x + residual))
        return self.head(x).squeeze(-1)

# ──────────────────────────────────────────────
# Architecture 2: RealMLP
# (Piecewise-Linear Embedding for numeric features + GLU gating)
# Inspired by Gorishniy et al. 2023 "On Embeddings for Numerical Features"
# ──────────────────────────────────────────────
class PiecewiseLinearEmbedding(nn.Module):
    """Embeds each numeric feature into `n_bins` piecewise-linear bins."""
    def __init__(self, n_features, n_bins=8, embed_dim=8):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        self.embed_dim = embed_dim
        # Learnable bin boundaries (n_features, n_bins+1) — learnable quantile boundaries
        self.boundaries = nn.Parameter(torch.randn(n_features, n_bins + 1))
        # Embedding weights for each bin
        self.weights = nn.Parameter(torch.randn(n_features, n_bins, embed_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_features, embed_dim))

    def forward(self, x):
        # x: (B, n_features)
        # Compute bin activations (softmax over distance to boundaries)
        boundaries = torch.sort(self.boundaries, dim=-1).values  # keep sorted
        # x_exp: (B, n_features, 1)
        x_exp = x.unsqueeze(-1)
        # dist: (B, n_features, n_bins+1)
        edges = boundaries.unsqueeze(0)
        # Piecewise linear activation: clamp(x - left, 0, right - left) / (right - left + eps)
        left = edges[:, :, :-1]  # (1, n_feat, n_bins)
        right = edges[:, :, 1:]
        width = (right - left).abs() + 1e-5
        act = torch.clamp(x_exp - left, 0.0, width) / width  # (B, n_feat, n_bins)
        # Weighted sum over bins → embed_dim per feature
        # act: (B, n_feat, n_bins), weights: (n_feat, n_bins, embed_dim)
        out = torch.einsum('bfn,fne->bfe', act, self.weights) + self.bias.unsqueeze(0)
        return out.reshape(x.size(0), -1)  # (B, n_feat * embed_dim)


class RealMLP(nn.Module):
    def __init__(self, n_num, n_cat_total_emb, n_bins=8, embed_dim=8,
                 hidden_dims=(512, 512, 256), dropout=0.2):
        super().__init__()
        self.ple = PiecewiseLinearEmbedding(n_num, n_bins=n_bins, embed_dim=embed_dim)
        in_dim = n_num * embed_dim + n_cat_total_emb

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h * 2),  # *2 for GLU gating
                nn.GLU(dim=-1),
                nn.LayerNorm(h),
                nn.Dropout(dropout),
            ]
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(self, x_num, x_cat_emb):
        ple_out = self.ple(x_num)
        x = torch.cat([ple_out, x_cat_emb], dim=-1)
        x = self.mlp(x)
        return self.head(x).squeeze(-1)


# ──────────────────────────────────────────────
# Entity Embedder (shared, for RealMLP)
# ──────────────────────────────────────────────
class EntityEmbedder(nn.Module):
    def __init__(self, cardinalities, embed_dim_mult=1.5):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.total_dim = 0
        for card in cardinalities:
            dim = max(4, min(32, int(round(card ** 0.5 * embed_dim_mult))))
            self.embeddings.append(nn.Embedding(card + 1, dim))
            self.total_dim += dim

    def forward(self, x_cat):
        embs = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])]
        return torch.cat(embs, dim=-1)


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def train_fold(model, train_loader, val_loader, y_val, epochs, lr, warmup_epochs=2):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Warmup + cosine annealing with warm restarts
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs - warmup_epochs), T_mult=1, eta_min=lr * 0.01
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_val_preds = None
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            *inputs, y_b = batch
            if len(inputs) == 1:
                logits = model(inputs[0])
            else:
                logits = model(*inputs)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Schedule
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_cosine.step()

        # Validation
        model.eval()
        preds = []
        with torch.inference_mode():
            for batch in val_loader:
                *inputs, _ = batch
                if len(inputs) == 1:
                    logits = model(inputs[0])
                else:
                    logits = model(*inputs)
                preds.append(torch.sigmoid(logits).cpu().numpy())

        val_preds = np.concatenate(preds)
        auc = roc_auc_score(y_val, val_preds)
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Val AUC: {auc:.6f}")

        if auc > best_auc:
            best_auc = auc
            best_val_preds = val_preds.copy()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_auc, best_val_preds


def get_test_preds(model, test_loader):
    model.eval()
    preds = []
    with torch.inference_mode():
        for batch in test_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                logits = model(batch[0])
            elif isinstance(batch, (list, tuple)):
                logits = model(*batch)
            else:
                logits = model(batch)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


# ──────────────────────────────────────────────
# Prepare data
# ──────────────────────────────────────────────
def prepare_data():
    train_df, test_df, cat_cols = engineer_features()
    target = train_df['target'].values
    ignore = ['id', 'target', 'Will_Buy_EV']
    feature_cols = [c for c in train_df.columns if c not in ignore]

    cat_features = [c for c in feature_cols
                    if c in cat_cols
                    or str(train_df[c].dtype) in ('category', 'object')]
    num_features = [c for c in feature_cols if c not in cat_features]

    print(f"Numeric features: {len(num_features)}  |  Categorical features: {len(cat_features)}")

    # Label-encode categoricals
    cat_cardinalities = []
    for col in cat_features:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        mapping = {v: i + 1 for i, v in enumerate(combined.unique())}
        train_df[col] = train_df[col].astype(str).map(mapping).fillna(0).astype(np.int64)
        test_df[col]  = test_df[col].astype(str).map(mapping).fillna(0).astype(np.int64)
        cat_cardinalities.append(len(mapping))

    X_num_train = train_df[num_features].fillna(0).values.astype(np.float32)
    X_cat_train = train_df[cat_features].values.astype(np.int64)
    X_num_test  = test_df[num_features].fillna(0).values.astype(np.float32)
    X_cat_test  = test_df[cat_features].values.astype(np.int64)

    return (X_num_train, X_cat_train, X_num_test, X_cat_test,
            target, test_df['id'].values, cat_cardinalities,
            len(num_features), len(cat_features))


# ──────────────────────────────────────────────
# Cross-validated training for one architecture
# ──────────────────────────────────────────────
def cross_val_train(
    arch_name,
    model_fn,         # callable(fold_data) -> model
    batch_fn,         # callable(split_data) -> (train_loader, val_loader, test_loader)
    X_num, X_cat, X_num_test, X_cat_test,
    target, test_ids,
    n_splits=5, epochs=20, lr=1e-3, batch_size=4096
):
    print("\n" + "=" * 70)
    print(f"TRAINING: {arch_name}")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(target), dtype=np.float32)
    test_preds = np.zeros(len(test_ids), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num, target)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # Quantile-transform numerics (fold-safe)
        qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal', random_state=42)
        x_num_tr = qt.fit_transform(X_num[tr_idx]).astype(np.float32)
        x_num_va = qt.transform(X_num[va_idx]).astype(np.float32)
        x_num_te = qt.transform(X_num_test).astype(np.float32)

        y_tr = target[tr_idx].astype(np.float32)
        y_va = target[va_idx].astype(np.float32)

        x_cat_tr = X_cat[tr_idx]
        x_cat_va = X_cat[va_idx]

        model, train_loader, val_loader, test_loader = model_fn(
            x_num_tr, x_cat_tr, y_tr,
            x_num_va, x_cat_va, y_va,
            x_num_te, X_cat_test,
            batch_size
        )

        model, best_auc, val_preds = train_fold(model, train_loader, val_loader, y_va, epochs, lr)
        print(f">> Fold {fold + 1} Best AUC: {best_auc:.6f}")

        oof[va_idx] = val_preds
        test_preds += get_test_preds(model, test_loader) / n_splits

    overall_auc = roc_auc_score(target, oof)
    print(f"\n>> {arch_name} Overall OOF AUC: {overall_auc:.6f}")

    os.makedirs('./preds', exist_ok=True)
    tag = arch_name.lower().replace(' ', '_')
    np.save(f'./preds/oof_{tag}.npy', oof)
    np.save(f'./preds/test_{tag}.npy', test_preds)

    sub = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_preds})
    sub.to_csv(f'./preds/submission_{tag}.csv', index=False)
    print(f"Saved ./preds/oof_{tag}.npy, test_{tag}.npy, submission_{tag}.csv")
    return overall_auc


# ──────────────────────────────────────────────
# Model factory functions
# ──────────────────────────────────────────────
def make_resnet_fold(cat_cardinalities, in_dim):
    def _factory(x_num_tr, x_cat_tr, y_tr, x_num_va, x_cat_va, y_va, x_num_te, x_cat_te, bs):
        # Concatenate entity embeddings inline for ResNet
        embedder = EntityEmbedder(cat_cardinalities, embed_dim_mult=1.5)

        class ResNetWithEmb(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedder = embedder
                total_emb = embedder.total_dim
                self.resnet = TabularResNet(in_dim + total_emb, hidden_dims=(512, 256, 128), dropout=0.15)

            def forward(self, x_num, x_cat):
                emb = self.embedder(x_cat)
                x = torch.cat([x_num, emb], dim=-1)
                return self.resnet(x)

        model = ResNetWithEmb()

        def make_loader(xn, xc, y, shuffle, bs):
            ds = TensorDataset(torch.tensor(xn), torch.tensor(xc), torch.tensor(y.astype(np.float32)))
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

        def make_test_loader(xn, xc, bs):
            ds = TensorDataset(torch.tensor(xn), torch.tensor(xc))
            return DataLoader(ds, batch_size=bs * 2, shuffle=False)

        train_loader = make_loader(x_num_tr, x_cat_tr, y_tr, True, bs)
        val_loader   = make_loader(x_num_va, x_cat_va, y_va, False, bs)
        test_loader  = make_test_loader(x_num_te, x_cat_te, bs)

        # Override get_test_preds to handle (xn, xc) tuples
        return model, train_loader, val_loader, test_loader
    return _factory


def make_realmlp_fold(cat_cardinalities, n_num):
    def _factory(x_num_tr, x_cat_tr, y_tr, x_num_va, x_cat_va, y_va, x_num_te, x_cat_te, bs):
        embedder = EntityEmbedder(cat_cardinalities, embed_dim_mult=1.5)

        class RealMLPWithEmb(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedder = embedder
                self.mlp = RealMLP(
                    n_num=n_num,
                    n_cat_total_emb=embedder.total_dim,
                    n_bins=16,
                    embed_dim=8,
                    hidden_dims=(512, 512, 256),
                    dropout=0.2
                )

            def forward(self, x_num, x_cat):
                emb = self.embedder(x_cat)
                return self.mlp(x_num, emb)

        model = RealMLPWithEmb()

        def make_loader(xn, xc, y, shuffle, bs):
            ds = TensorDataset(torch.tensor(xn), torch.tensor(xc), torch.tensor(y.astype(np.float32)))
            return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

        def make_test_loader(xn, xc, bs):
            ds = TensorDataset(torch.tensor(xn), torch.tensor(xc))
            return DataLoader(ds, batch_size=bs * 2, shuffle=False)

        train_loader = make_loader(x_num_tr, x_cat_tr, y_tr, True, bs)
        val_loader   = make_loader(x_num_va, x_cat_va, y_va, False, bs)
        test_loader  = make_test_loader(x_num_te, x_cat_te, bs)
        return model, train_loader, val_loader, test_loader
    return _factory


# ──────────────────────────────────────────────
# Override get_test_preds to handle (xn, xc) batches without y
# ──────────────────────────────────────────────
def get_test_preds_xy(model, test_loader):
    model.eval()
    preds = []
    with torch.inference_mode():
        for xn, xc in test_loader:
            logits = model(xn, xc)
            preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


# ──────────────────────────────────────────────
# Adapted cross-val for (xnum, xcat) models
# ──────────────────────────────────────────────
def cross_val_train_xy(
    arch_name, model_factory,
    X_num, X_cat, X_num_test, X_cat_test,
    target, test_ids,
    n_splits=5, epochs=20, lr=1e-3, batch_size=4096
):
    print("\n" + "=" * 70)
    print(f"TRAINING: {arch_name}")
    print("=" * 70)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(target), dtype=np.float32)
    test_preds = np.zeros(len(test_ids), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_num, target)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        qt = QuantileTransformer(n_quantiles=min(1000, len(tr_idx)), output_distribution='normal', random_state=42)
        x_num_tr = qt.fit_transform(X_num[tr_idx]).astype(np.float32)
        x_num_va = qt.transform(X_num[va_idx]).astype(np.float32)
        x_num_te = qt.transform(X_num_test).astype(np.float32)

        y_tr = target[tr_idx].astype(np.float32)
        y_va = target[va_idx].astype(np.float32)

        x_cat_tr, x_cat_va = X_cat[tr_idx], X_cat[va_idx]

        model, train_loader, val_loader, test_loader = model_factory(
            x_num_tr, x_cat_tr, y_tr,
            x_num_va, x_cat_va, y_va,
            x_num_te, X_cat_test,
            batch_size
        )

        model, best_auc, val_preds = train_fold(
            model, train_loader, val_loader, y_va, epochs, lr
        )
        print(f">> Fold {fold + 1} Best AUC: {best_auc:.6f}")
        oof[va_idx] = val_preds
        test_preds += get_test_preds_xy(model, test_loader) / n_splits

    overall_auc = roc_auc_score(target, oof)
    print(f"\n>> {arch_name} Overall OOF AUC: {overall_auc:.6f}")

    tag = arch_name.lower().replace(' ', '_')
    np.save(f'./preds/oof_{tag}.npy', oof)
    np.save(f'./preds/test_{tag}.npy', test_preds)
    sub = pd.DataFrame({'id': test_ids, 'Will_Buy_EV': test_preds})
    sub.to_csv(f'./preds/submission_{tag}.csv', index=False)
    print(f"Saved ./preds/oof_{tag}.npy, test_{tag}.npy, submission_{tag}.csv")
    return overall_auc


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == '__main__':
    seed_everything(42)

    (X_num, X_cat, X_num_test, X_cat_test,
     target, test_ids, cat_cardinalities,
     n_num, n_cat) = prepare_data()

    results = {}

    # ── 1. TabularResNet (30 epochs) ──────────────────────────────────────
    resnet_factory = make_resnet_fold(cat_cardinalities, n_num)
    auc_resnet = cross_val_train_xy(
        'tabular_resnet', resnet_factory,
        X_num, X_cat, X_num_test, X_cat_test,
        target, test_ids,
        n_splits=5, epochs=30, lr=3e-3, batch_size=4096
    )
    results['TabularResNet (30 epochs)'] = auc_resnet

    # ── 2. RealMLP (30 epochs) ────────────────────────────────────────────
    realmlp_factory = make_realmlp_fold(cat_cardinalities, n_num)
    auc_realmlp = cross_val_train_xy(
        'realmlp', realmlp_factory,
        X_num, X_cat, X_num_test, X_cat_test,
        target, test_ids,
        n_splits=5, epochs=30, lr=1e-3, batch_size=4096
    )
    results['RealMLP (30 epochs)'] = auc_realmlp

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL NEURAL ARCHITECTURE COMPARISON")
    print("=" * 70)
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:35s} | OOF ROC-AUC: {auc:.6f}")

    print("\nNext Step: run `python ensemble_advanced.py` to include these")
    print("predictions in the Hill Climbing + Logit Stacking pipeline.")
