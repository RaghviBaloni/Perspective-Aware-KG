import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================
# CONFIG
# ==============================================
MERGED_PATH = "/home/user/Perspective-Aware-KG/data/processed/merged_taco_data.csv"
ANNOTATED_PATH = "/home/user/Perspective-Aware-KG/data/processed/perspective_enriched.csv"

# NOTE: adjust if your embeddings dir is named differently
EMB_DIR = Path("/home/user/Perspective-Aware-KG/data/embedding")

GRAPH_DIR = Path("/home/user/Perspective-Aware-KG/data/graph")
TRAINING_DIR = Path("/home/user/Perspective-Aware-KG/data/training")

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 123456789

print("="*70)
print("INDUCTIVE KNOWLEDGE GRAPH CONSTRUCTION")
print("="*70)

# ==============================================
# LOAD & MERGE DATA
# ==============================================
df_merged = pd.read_csv(MERGED_PATH, dtype={"tweet_id": str, "parent_id": str, "conversation_id": str})
df_annot = pd.read_csv(ANNOTATED_PATH, dtype={"tweet_id": str})

df = pd.merge(df_merged, df_annot, on=["tweet_id", "topic"], suffixes=("_taco", "_persp"))
df = df[df["taco_class"] != "undecided"].copy()
df = df.sort_values("tweet_id").reset_index(drop=True)

print(f"Total tweets after filtering: {len(df)}")

# ==============================================
# PERSPECTIVE FEATURES
# ==============================================
perspective_cols = [
    "information_providing", 
    "information_seeking", 
    "epistemic", 
    "experiential", 
    "opinion", 
    "aot",
]
perspective_feats = df[perspective_cols].values.astype(np.float32)

# ==============================================
# TOPIC ONE-HOT
# ==============================================
topics = df["topic"].astype("category")
topic_ids = topics.cat.codes.values
num_topics = topics.cat.categories.size
topic_onehot = np.eye(num_topics)[topic_ids]

# ==============================================
# DEPTH COMPUTATION (same as your robust version)
# ==============================================
print("\n  Computing hierarchical depth...")

def compute_actual_depth(df):
    depth_cache = {}
    id2parent = df.set_index("tweet_id")["parent_id"].to_dict()
    id2conv = df.set_index("tweet_id")["conversation_id"].to_dict()
    valid_ids = set(df["tweet_id"])

    def get_depth(tweet_id, visited=None):
        if visited is None:
            visited = set()
        if tweet_id in visited:
            return 0
        visited.add(tweet_id)

        if tweet_id in depth_cache:
            return depth_cache[tweet_id]

        conv_id = id2conv.get(tweet_id)

        if tweet_id == conv_id:
            depth = 0
        else:
            parent_id = id2parent.get(tweet_id)
            if parent_id is None or parent_id == conv_id:
                depth = 1
            elif parent_id not in valid_ids:
                depth = 1
            else:
                depth = get_depth(parent_id, visited) + 1

        depth_cache[tweet_id] = depth
        return depth

    return [get_depth(tid) for tid in df["tweet_id"]]

depths = np.array(compute_actual_depth(df), dtype=np.float32)
depth_norm = (depths / depths.max()).reshape(-1, 1)

print(f"    Depth range: 0 to {depths.max()}")
print(f"    Depth distribution: {np.bincount(depths.astype(int))}")

# ==============================================
# CONFIDENCE
# ==============================================
confidence = df["confidence"].values.astype(np.float32).reshape(-1, 1)

# ==============================================
# LABELS
# ==============================================
label_map = {"reason": 0, "statement": 1, "notification": 2, "none": 3}
y = torch.tensor(df["taco_class"].map(label_map).values, dtype=torch.long)

# ==============================================
# STRATIFIED SPLITS (MATCH TEXT CV)
# ==============================================
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)
splits = list(skf.split(np.arange(len(df)), y.numpy()))

# ==============================================
# KNN EDGES
# ==============================================
def build_knn_edges(embeddings, k=3, threshold=0.75):
    sim = cosine_similarity(embeddings)
    src, dst = [], []
    for i in range(len(embeddings)):
        top_k = np.argsort(sim[i])[-k-1:-1]
        for j in top_k:
            if sim[i, j] > threshold:
                src.extend([i, j])
                dst.extend([j, i])
    return torch.tensor([src, dst], dtype=torch.long)

# ==============================================
# MAIN LOOP: BUILD INDUCTIVE GRAPHS PER FOLD
# ==============================================
for fold in range(10):
    print(f"\n=== Building inductive graphs for Fold {fold} ===")

    # --------------------------
    # Load fold-specific embeddings
    # --------------------------
    emb_path = EMB_DIR / f"embeddings_fold_{fold}.npy"
    id_path = EMB_DIR / f"tweet_ids_fold_{fold}.npy"

    embs = np.load(emb_path)
    ids = np.load(id_path, allow_pickle=True)
    ids = [str(t).strip() for t in ids]

    emb_dict = {tid: emb for tid, emb in zip(ids, embs)}
    aligned_embs = np.stack([emb_dict[str(t)] for t in df["tweet_id"]], axis=0)

    # --------------------------
    # Build full node features (for inference graph)
    # --------------------------
    X_all = np.concatenate([
        aligned_embs,
        perspective_feats,
        topic_onehot,
        depth_norm,
        confidence
    ], axis=1).astype(np.float32)

    x_all = torch.from_numpy(X_all)

    # --------------------------
    # Build full edges (reply + conv + KNN) on ALL nodes
    # --------------------------
    reply_src, reply_dst = [], []
    for _, row in df.iterrows():
        c = row["tweet_id"]
        p = row["parent_id"]
        if p in df["tweet_id"].values and c != p:
            ci = df.index[df["tweet_id"] == c][0]
            pi = df.index[df["tweet_id"] == p][0]
            reply_src.extend([ci, pi])
            reply_dst.extend([pi, ci])
    reply_edge_index = torch.tensor([reply_src, reply_dst], dtype=torch.long)

    conv_src, conv_dst = [], []
    for conv_id, group in df.groupby("conversation_id"):
        root = group.index[group["tweet_id"] == conv_id][0]
        for idx in group.index:
            if idx != root:
                conv_src.extend([root, idx])
                conv_dst.extend([idx, root])
    conv_edge_index = torch.tensor([conv_src, conv_dst], dtype=torch.long)

    knn_edge_index_all = build_knn_edges(aligned_embs, k=3, threshold=0.75)

    edge_index_all = torch.cat([reply_edge_index, conv_edge_index, knn_edge_index_all], dim=1)

    # --------------------------
    # Train / test indices for this fold
    # --------------------------
    train_idx, test_idx = splits[fold]
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)

    # ==========================================
    # 1) TRAIN GRAPH (INDUCTIVE): ONLY TRAIN NODES
    # ==========================================
    # Map old indices -> new [0..num_train-1]
    old_to_new = {old: new for new, old in enumerate(train_idx)}

    # Filter edges where BOTH endpoints are in train_idx
    mask_train_edges = []
    src_all = edge_index_all[0].numpy()
    dst_all = edge_index_all[1].numpy()
    for s, d in zip(src_all, dst_all):
        if s in old_to_new and d in old_to_new:
            mask_train_edges.append(True)
        else:
            mask_train_edges.append(False)
    mask_train_edges = np.array(mask_train_edges)

    train_src_old = src_all[mask_train_edges]
    train_dst_old = dst_all[mask_train_edges]

    train_src_new = [old_to_new[s] for s in train_src_old]
    train_dst_new = [old_to_new[d] for d in train_dst_old]

    edge_index_train = torch.tensor([train_src_new, train_dst_new], dtype=torch.long)

    # Node features and labels for train graph
    x_train = torch.from_numpy(X_all[train_idx])
    y_train = y[train_idx]

    train_mask_train_graph = torch.ones(len(train_idx), dtype=torch.bool)

    data_train = Data(
        x=x_train,
        edge_index=edge_index_train,
        y=y_train,
        train_mask=train_mask_train_graph
    )

    torch.save(data_train, TRAINING_DIR / f"train_graph_fold_{fold}.pt")
    print(f"Saved train graph: train_graph_fold_{fold}.pt")

    # ==========================================
    # 2) INFERENCE GRAPH: TRAIN + TEST NODES
    # ==========================================
    # Here we keep ALL nodes and ALL edges (as in your original KG),
    # but this graph is ONLY used after training, for evaluation.
    train_mask_full = torch.zeros(len(df), dtype=torch.bool)
    test_mask_full = torch.zeros(len(df), dtype=torch.bool)
    train_mask_full[train_idx] = True
    test_mask_full[test_idx] = True

    data_full = Data(
        x=x_all,
        edge_index=edge_index_all,
        y=y,
        train_mask=train_mask_full,
        test_mask=test_mask_full
    )

    torch.save(data_full, TRAINING_DIR / f"infer_graph_fold_{fold}.pt")
    print(f"Saved inference graph: infer_graph_fold_{fold}.pt")

# ==============================================
# METADATA (OPTIONAL)
# ==============================================
meta = {
    "num_nodes": len(df),
    "num_topics": int(num_topics),
    "perspective_cols": perspective_cols,
    "label_map": label_map,
}
with open(GRAPH_DIR / "metadata_inductive.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nAll inductive folds complete!")
