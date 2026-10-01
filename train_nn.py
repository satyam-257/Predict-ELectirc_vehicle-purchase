import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from features import engineer_features
import os

# Set seed for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

class TabularResNet(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embed_dim_mult=1.5, hidden_dims=[256, 128], dropout=0.15):
        super().__init__()
        # Entity Embeddings for categoricals
        self.embeddings = nn.ModuleList()
        total_embed_dim = 0
        for card in cat_cardinalities:
            # Rule of thumb for embedding dimension: min(50, max(4, round(card**0.5 * mult)))
            dim = max(4, min(32, int(round(card ** 0.5 * embed_dim_mult))))
            self.embeddings.append(nn.Embedding(card + 1, dim))
            total_embed_dim += dim
            
        in_features = num_features + total_embed_dim
        
        # Initial input projection
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Residual Block 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        self.block1_norm = nn.LayerNorm(hidden_dims[0])
        self.block1_act = nn.SiLU()
        
        # Transition Block
        self.trans = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Residual Block 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[1])
        )
        self.block2_norm = nn.LayerNorm(hidden_dims[1])
        self.block2_act = nn.SiLU()
        
        # Output Head
        self.head = nn.Linear(hidden_dims[1], 1)
        
    def forward(self, x_num, x_cat):
        # Embed and concatenate categoricals
        if len(self.embeddings) > 0:
            embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + embedded, dim=1)
        else:
            x = x_num
            
        h = self.input_layer(x)
        h = self.block1_act(self.block1_norm(h + self.block1(h)))
        h = self.trans(h)
        h = self.block2_act(self.block2_norm(h + self.block2(h)))
        out = self.head(h)
        return out.squeeze(-1)

def train_neural_net(epochs=10, batch_size=4096, lr=3e-3, n_splits=5):
    print("=" * 70)
    print("S6E9 BEYOND BOOSTING: TABULAR RESNET / DEEP LEARNING PIPELINE")
    print("=" * 70)
    seed_everything(42)
    
    train_df, test_df, cat_cols = engineer_features()
    
    target = train_df['target'].values
    ignore_cols = ['id', 'target', 'Will_Buy_EV']
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]
    
    # Identify numerical vs categorical
    cat_features = [c for c in feature_cols if c in cat_cols or train_df[c].dtype.name == 'category' or train_df[c].dtype == 'object']
    num_features = [c for c in feature_cols if c not in cat_features]
    
    print(f"Features: Total={len(feature_cols)}, Numeric={len(num_features)}, Categorical={len(cat_features)}")
    
    # Label encode categoricals
    cat_cardinalities = []
    for col in cat_features:
        combined = pd.concat([train_df[col], test_df[col]], axis=0).astype(str)
        mapping = {val: idx + 1 for idx, val in enumerate(combined.unique())}
        train_df[col] = train_df[col].astype(str).map(mapping).fillna(0).astype(np.int64)
        test_df[col] = test_df[col].astype(str).map(mapping).fillna(0).astype(np.int64)
        cat_cardinalities.append(len(mapping))
        
    num_train = train_df[num_features].fillna(0).values.astype(np.float32)
    cat_train = train_df[cat_features].values.astype(np.int64)
    num_test = test_df[num_features].fillna(0).values.astype(np.float32)
    cat_test = test_df[cat_features].values.astype(np.int64)
    
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    criterion = nn.BCEWithLogitsLoss()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, target)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        scaler = StandardScaler()
        x_num_tr = scaler.fit_transform(num_train[train_idx])
        x_num_va = scaler.transform(num_train[val_idx])
        x_num_te_fold = scaler.transform(num_test)
        
        y_tr = target[train_idx].astype(np.float32)
        y_va = target[val_idx].astype(np.float32)
        
        train_ds = TensorDataset(
            torch.tensor(x_num_tr), torch.tensor(cat_train[train_idx]), torch.tensor(y_tr)
        )
        val_ds = TensorDataset(
            torch.tensor(x_num_va), torch.tensor(cat_train[val_idx]), torch.tensor(y_va)
        )
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        
        model = TabularResNet(
            num_features=len(num_features),
            cat_cardinalities=cat_cardinalities,
            hidden_dims=[256, 128],
            dropout=0.15
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_auc = 0.0
        best_val_preds = None
        best_model_state = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_num, batch_cat, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_num, batch_cat)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds_list = []
            with torch.inference_mode():
                for batch_num, batch_cat, _ in val_loader:
                    logits = model(batch_num, batch_cat)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_preds_list.append(probs)
                    
            val_preds = np.concatenate(val_preds_list)
            val_auc = roc_auc_score(y_va, val_preds)
            print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Train Loss: {total_loss/len(train_loader):.4f} | Val ROC-AUC: {val_auc:.6f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_preds = val_preds
                best_model_state = model.state_dict()
                
        print(f">> Fold {fold + 1} Best Val ROC-AUC: {best_val_auc:.6f}")
        oof_preds[val_idx] = best_val_preds
        
        # Test predictions with best model
        model.load_state_dict(best_model_state)
        model.eval()
        test_ds = TensorDataset(torch.tensor(x_num_te_fold), torch.tensor(cat_test))
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
        test_preds_list = []
        with torch.inference_mode():
            for batch_num, batch_cat in test_loader:
                logits = model(batch_num, batch_cat)
                probs = torch.sigmoid(logits).cpu().numpy()
                test_preds_list.append(probs)
                
        test_preds += np.concatenate(test_preds_list) / n_splits
        
    overall_auc = roc_auc_score(target, oof_preds)
    print("\n" + "=" * 70)
    print(f"OVERALL NEURAL NETWORK OOF ROC-AUC: {overall_auc:.6f}")
    print("=" * 70)
    
    os.makedirs('./preds', exist_ok=True)
    np.save('./preds/oof_nn.npy', oof_preds)
    np.save('./preds/test_nn.npy', test_preds)
    
    # Save submission
    sub = pd.DataFrame({'id': test_df['id'].values, 'Will_Buy_EV': test_preds})
    sub.to_csv('./preds/submission_nn.csv', index=False)
    print("Saved ./preds/oof_nn.npy, ./preds/test_nn.npy, and ./preds/submission_nn.csv")
    return overall_auc

if __name__ == '__main__':
    train_neural_net(epochs=8, batch_size=4096, lr=3e-3, n_splits=5)
