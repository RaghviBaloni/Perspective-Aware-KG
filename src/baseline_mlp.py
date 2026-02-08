import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ============================================
# CONFIG
# ============================================
EMB_DIR = "/home/baloni/Perspective-Aware-KG/data/baseline_models"
TACO_CSV = "/home/baloni/Perspective-Aware-KG/data/processed/merged_taco_data.csv"
RANDOM_SEED = 123456789
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv(TACO_CSV, dtype={"tweet_id": str})
df = df[df["class"] != "undecided"].copy()
df = df.sort_values("tweet_id").reset_index(drop=True)

label_map = {"reason": 0, "statement": 1, "notification": 2, "none": 3}
y = df["class"].map(label_map).values

# ============================================
# DEFINE MLP
# ============================================
class MLP(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ============================================
# CROSS-VALIDATION SPLITS
# ============================================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), y)):
    print(f"\n========== Fold {fold} ==========")

    # Load embeddings for this fold
    emb = np.load(f"{EMB_DIR}/embeddings_fold_{fold}.npy")
    ids = np.load(f"{EMB_DIR}/tweet_ids_fold_{fold}.npy", allow_pickle=True)

    # Convert to torch
    X = torch.tensor(emb, dtype=torch.float32).to(DEVICE)
    Y = torch.tensor(y, dtype=torch.long).to(DEVICE)

    # Masks
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Model
    model = MLP(in_dim=emb.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Training
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits[train_mask], Y[train_mask])
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    logits = model(X)
    preds = logits.argmax(dim=1).cpu().numpy()
    y_true = Y.cpu().numpy()

    acc = accuracy_score(y_true[test_mask], preds[test_mask])
    f1_macro = f1_score(y_true[test_mask], preds[test_mask], average="macro")
    f1_weighted = f1_score(y_true[test_mask], preds[test_mask], average="weighted")
    f1_per_class = f1_score(y_true[test_mask], preds[test_mask], average=None)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print(f"Weighted F1: {f1_weighted:.4f}")
    print("Per-class F1:", f1_per_class)

    results.append({
        "fold": fold,
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class.tolist()
    })

# ============================================
# AGGREGATE RESULTS
# ============================================
accs = [r["acc"] for r in results]
f1s = [r["f1_macro"] for r in results]
wfs = [r["f1_weighted"] for r in results]
f1_per_class_list = [r["f1_per_class"] for r in results]

# Compute mean and std for per-class F1 scores
f1_per_class_array = np.array(f1_per_class_list)
f1_per_class_mean = np.mean(f1_per_class_array, axis=0)
f1_per_class_std = np.std(f1_per_class_array, axis=0)

# Class names mapping (reverse of label_map)
class_names = {0: "reason", 1: "statement", 2: "notification", 3: "none"}

print("\n================ FINAL RESULTS ================")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Weighted F1: {np.mean(wfs):.4f} ± {np.std(wfs):.4f}")
print("\nPer-class F1 scores:")
for class_id, class_name in class_names.items():
    print(f"  {class_name}: {f1_per_class_mean[class_id]:.4f} ± {f1_per_class_std[class_id]:.4f}")
'''
# ============================================
# AGGREGATE RESULTS
# ============================================
accs = [r["acc"] for r in results]
f1s = [r["f1_macro"] for r in results]
wfs = [r["f1_weighted"] for r in results]

print("\n================ FINAL RESULTS ================")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"Macro F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"Weighted F1: {np.mean(wfs):.4f} ± {np.std(wfs):.4f}")
'''
