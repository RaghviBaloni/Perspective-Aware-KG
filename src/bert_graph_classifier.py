import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch_geometric.nn import SAGEConv, LayerNorm, global_mean_pool
from sklearn.metrics import f1_score, classification_report
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

# ---------------- Config ----------------
DATA_DIR = Path("data")
EMB_DIR = DATA_DIR / "embeddings"
PROC_DIR = DATA_DIR / "processed"
TRAINING_DIR = DATA_DIR / "training"
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BERT_MODEL_NAME = "vinai/bertweet-base"
MAX_LEN = 128
NUM_CLASSES = 4
NUM_FOLDS = 10
NUM_EPOCHS = 20
LR = 5e-4  # Higher LR for small dataset
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
SAGE_HIDDEN = 256
SAGE_LAYERS = 2
GNN_DROPOUT = 0.3

label_names = ['reason', 'statement', 'notification', 'none']

# ---------------- Load Data ----------------
print("Loading data...")
mapping_df = pd.read_csv(EMB_DIR / "tweet_id_mapping.csv")
df = pd.read_csv(PROC_DIR / "perspective_enriched.csv")

# Keep only rows whose tweet_id appears in the graph mapping
df = df[df["tweet_id"].astype(str).isin(mapping_df["tweet_id"].astype(str))]
df = df.sort_values("tweet_id")  # ensure same order
mapping_df = mapping_df.sort_values("tweet_id")

id2text = dict(zip(df["tweet_id"].astype(str), df["text"]))
texts = [id2text[str(tid)] for tid in mapping_df["tweet_id"]]
print(f"Total nodes: {len(texts)}")

# Load perspective features
perspective_cols = [
    'information_providing', 'information_seeking', 'epistemic',
    'experiential', 'opinion', 'aot'
]
perspective_features = df[perspective_cols].values
texts = [id2text[str(tid)] for tid in mapping_df["tweet_id"]]

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=False)


# ============================================================
# STAGE 1: Extract BERT Embeddings (with mean pooling)
# ============================================================
def extract_bert_embeddings(texts, device, batch_size=64):
    """Extract BERT embeddings using mean pooling"""
    print("Extracting BERT embeddings...")
    bert = AutoModel.from_pretrained(BERT_MODEL_NAME).to(device)
    bert.eval()
    
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            enc = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            
            # Mean pooling (better than just [CLS])
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            all_embeddings.append(mean_embeddings.cpu())
    
    del bert
    torch.cuda.empty_cache()
    
    return torch.cat(all_embeddings, dim=0)


# ============================================================
# STAGE 2: GNN with Class Balancing
# ============================================================
class BertGNNClassifierBalanced(nn.Module):
    """
    BERT + GNN with:
    1. Perspective features
    2. Residual connections
    3. Class-balanced loss
    """
    def __init__(
        self,
        bert_dim=768,
        persp_dim=6,
        hidden_dim=SAGE_HIDDEN,
        num_layers=SAGE_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=GNN_DROPOUT,
    ):
        super().__init__()
        
        # Input projection (BERT + perspective features)
        self.input_proj = nn.Sequential(
            nn.Linear(bert_dim + persp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GraphSAGE layers with residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(LayerNorm(hidden_dim))
        
        self.dropout = dropout
        
        # Multi-head classifier (helps with imbalanced data)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + bert_dim + persp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x_bert, x_persp, edge_index):
        """
        Args:
            x_bert: [N, 768] frozen BERT embeddings
            x_persp: [N, 6] perspective features
            edge_index: [2, E] graph edges
        """
        # Combine BERT + perspective features
        x_input = torch.cat([x_bert, x_persp], dim=1)
        
        # Initial projection
        x = self.input_proj(x_input)
        x_init = x  # Save for skip connection
        
        # GNN propagation with residuals
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_prev = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_prev
        
        # Multi-scale features (initial + final GNN + original BERT + persp)
        x_multi = torch.cat([x_init, x, x_bert, x_persp], dim=1)
        
        return self.classifier(x_multi)


# ============================================================
# Training with Focal Loss (handles class imbalance)
# ============================================================
class FocalLoss(nn.Module):
    """
    Focal Loss: focuses on hard-to-classify examples
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def compute_class_weights(data, device):
    """Compute inverse frequency class weights"""
    y_train = data.y[data.train_mask].cpu().numpy()
    class_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    
    # Inverse frequency weighting
    total = class_counts.sum()
    class_weights = total / (NUM_CLASSES * class_counts)
    
    # Normalize to sum to NUM_CLASSES
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")
    
    return torch.FloatTensor(class_weights).to(device)


def train_epoch(model, data, bert_emb, persp_feat, criterion, optimizer, device):
    model.train()
    optimizer.zero_grad()
    
    x_bert = bert_emb.to(device)
    x_persp = persp_feat.to(device)
    edge_index = data.edge_index.to(device)
    
    logits = model(x_bert, x_persp, edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(model, data, bert_emb, persp_feat, mask, device):
    model.eval()
    
    x_bert = bert_emb.to(device)
    x_persp = persp_feat.to(device)
    edge_index = data.edge_index.to(device)
    
    logits = model(x_bert, x_persp, edge_index)
    pred = logits.argmax(dim=1)
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()
    
    acc = (y_true == y_pred).mean()
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ============================================================
# Main Experiment
# ============================================================
def run_experiment_fixed():
    print(f"Device: {DEVICE}")
    
    # Extract BERT embeddings once
    bert_embeddings = extract_bert_embeddings(texts, DEVICE)
    print(f"BERT embeddings: {bert_embeddings.shape}")
    
    # Convert perspective features to tensor
    persp_features = torch.FloatTensor(perspective_features)
    print(f"Perspective features: {persp_features.shape}")
    
    all_results = []
    
    for fold in range(NUM_FOLDS):
        print("\n" + "=" * 70)
        print(f"Fold {fold+1}/{NUM_FOLDS}")
        print("=" * 70)
        
        data = torch.load(
            TRAINING_DIR / f"graph_fold_{fold}.pt",
            weights_only=False
        ).to(DEVICE)
        
        # Compute class weights
        class_weights = compute_class_weights(data, DEVICE)
        
        # Initialize model
        model = BertGNNClassifierBalanced().to(DEVICE)
        
        # Focal loss with class weights
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )
        
        best_f1 = 0.0
        patience = 0
        
        for epoch in range(NUM_EPOCHS):
            loss = train_epoch(
                model, data, bert_embeddings, persp_features, 
                criterion, optimizer, DEVICE
            )
            
            train_res = evaluate(
                model, data, bert_embeddings, persp_features, 
                data.train_mask, DEVICE
            )
            
            print(
                f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | "
                f"Loss: {loss:.4f} | "
                f"Acc: {train_res['accuracy']:.4f} | "
                f"F1: {train_res['f1_macro']:.4f}"
            )
            
            scheduler.step()
            
            # Early stopping
            if train_res['f1_macro'] > best_f1:
                best_f1 = train_res['f1_macro']
                patience = 0
            else:
                patience += 1
            
            if patience >= 7:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Test evaluation
        test_res = evaluate(
            model, data, bert_embeddings, persp_features,
            data.test_mask, DEVICE
        )
        
        print("\n" + "-" * 70)
        print("Test Results:")
        print(f"Accuracy:      {test_res['accuracy']:.4f}")
        print(f"F1 (macro):    {test_res['f1_macro']:.4f}")
        print(f"F1 (weighted): {test_res['f1_weighted']:.4f}")
        print("\nPer-class F1:")
        for name, f1 in zip(label_names, test_res["f1_per_class"]):
            print(f"  {name:15s}: {f1:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(
            test_res["y_true"],
            test_res["y_pred"],
            target_names=label_names,
            digits=4,
            zero_division=0
        ))
        
        all_results.append({
            "fold": fold,
            "test_accuracy": test_res["accuracy"],
            "test_f1_macro": test_res["f1_macro"],
            "test_f1_weighted": test_res["f1_weighted"],
            "test_f1_per_class": test_res["f1_per_class"].tolist(),
        })
        
        del model, data
        torch.cuda.empty_cache()
    
    # Final results
    avg_acc = np.mean([r['test_accuracy'] for r in all_results])
    std_acc = np.std([r['test_accuracy'] for r in all_results])
    avg_f1_macro = np.mean([r['test_f1_macro'] for r in all_results])
    std_f1_macro = np.std([r['test_f1_macro'] for r in all_results])
    avg_f1_weighted = np.mean([r['test_f1_weighted'] for r in all_results])
    std_f1_weighted = np.std([r['test_f1_weighted'] for r in all_results])
    avg_f1_per_class = np.mean([r['test_f1_per_class'] for r in all_results], axis=0)
    std_f1_per_class = np.std([r['test_f1_per_class'] for r in all_results], axis=0)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS: BERT + Perspectives + GNN (Balanced)")
    print("=" * 70)
    print(f"Average Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Average F1 (macro):    {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")
    print(f"Average F1 (weighted): {avg_f1_weighted:.4f} ± {std_f1_weighted:.4f}")
    print("\nAverage Per-class F1:")
    for name, f1_avg, f1_std in zip(label_names, avg_f1_per_class, std_f1_per_class):
        print(f"  {name:15s}: {f1_avg:.4f} ± {f1_std:.4f}")
    
    # Save
    results_file = OUTPUT_DIR / f"bert_gnn_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "model": "BERT+Perspectives+GNN_balanced",
            "hyperparameters": {
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": NUM_EPOCHS,
                "focal_gamma": 2.0,
            },
            "summary": {
                "avg_accuracy": float(avg_acc),
                "std_accuracy": float(std_acc),
                "avg_f1_macro": float(avg_f1_macro),
                "std_f1_macro": float(std_f1_macro),
                "avg_f1_weighted": float(avg_f1_weighted),
                "std_f1_weighted": float(std_f1_weighted),
                "avg_f1_per_class": avg_f1_per_class.tolist(),
                "std_f1_per_class": std_f1_per_class.tolist(),
            },
            "fold_results": all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    run_experiment_fixed()
