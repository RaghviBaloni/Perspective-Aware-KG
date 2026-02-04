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
MERGED_PATH = "/home/baloni/Perspective-Aware-KG/data/processed/merged_taco_data.csv"
ANNOTATED_PATH = "/home/baloni/Perspective-Aware-KG/data/processed/perspective_enriched.csv"
EMB_DIR = Path("/home/baloni/Perspective-Aware-KG/data/embedding")
OUTPUT_DIR = Path("/home/baloni/Perspective-Aware-KG/data/graph")
TRAINING_DIR = Path("/home/baloni/Perspective-Aware-KG/data/training")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 123456789

# ==============================================
# LOAD & MERGE DATA
# ==============================================
print("="*70)
print("FOLD-AWARE KNOWLEDGE GRAPH CONSTRUCTION")
print("="*70)

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
# DEPTH COMPUTATION (robust + cycle-safe)
# ==============================================
print("\n  Computing hierarchical depth...")

def compute_actual_depth(df):
    """
    Compute true depth via recursive traversal of parent chain.
    Handles:
      - cycles
      - missing parents
      - filtered-out parents
      - long reply chains
    """
    depth_cache = {}
    id2parent = df.set_index("tweet_id")["parent_id"].to_dict()
    id2conv = df.set_index("tweet_id")["conversation_id"].to_dict()
    valid_ids = set(df["tweet_id"])

    def get_depth(tweet_id, visited=None):
        if visited is None:
            visited = set()

        # Prevent infinite loops
        if tweet_id in visited:
            return 0
        visited.add(tweet_id)

        # Cached?
        if tweet_id in depth_cache:
            return depth_cache[tweet_id]

        conv_id = id2conv.get(tweet_id)

        # Root tweet
        if tweet_id == conv_id:
            depth = 0

        else:
            parent_id = id2parent.get(tweet_id)

            # Missing parent or direct reply to root
            if parent_id is None or parent_id == conv_id:
                depth = 1

            # Parent exists but was filtered out
            elif parent_id not in valid_ids:
                depth = 1

            # Recursive case
            else:
                depth = get_depth(parent_id, visited) + 1

        depth_cache[tweet_id] = depth
        return depth

    return [get_depth(tid) for tid in df["tweet_id"]]

depths = np.array(compute_actual_depth(df), dtype=np.float32)
depth_norm = (depths / depths.max()).reshape(-1, 1)

print(f"    Depth range: 0 to {depths.max()}")
print(f"    Depth distribution: {np.bincount(depths.astype(int))}")

'''
# ==============================================
# DEPTH COMPUTATION
# ==============================================
def compute_depth(df):
    depth_cache = {}
    id2parent = df.set_index("tweet_id")["parent_id"].to_dict()
    id2conv = df.set_index("tweet_id")["conversation_id"].to_dict()

    def get_depth(tid):
        if tid in depth_cache:
            return depth_cache[tid]
        conv = id2conv[tid]
        if tid == conv:
            depth_cache[tid] = 0
            return 0
        parent = id2parent.get(tid)
        if parent is None or parent == conv:
            depth_cache[tid] = 1
            return 1
        d = get_depth(parent) + 1
        depth_cache[tid] = d
        return d

    return np.array([get_depth(t) for t in df["tweet_id"]], dtype=np.float32)

depths = compute_depth(df)
depth_norm = (depths / depths.max()).reshape(-1, 1)
'''

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
skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)
splits = list(skf.split(np.arange(len(df)), y.numpy()))

# ==============================================
# BUILD KNN EDGES
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
# MAIN LOOP: BUILD KG PER FOLD
# ==============================================
for fold in range(10):
    print(f"\n=== Building KG for Fold {fold} ===")

    # Load fold-specific embeddings
    emb_path = EMB_DIR / f"embeddings_fold_{fold}.npy"
    id_path = EMB_DIR / f"tweet_ids_fold_{fold}.npy"

    embs = np.load(emb_path)
    ids = np.load(id_path, allow_pickle=True)
    ids = [str(t).strip() for t in ids]

    # Align embeddings with df order
    emb_dict = {tid: emb for tid, emb in zip(ids, embs)}
    aligned_embs = np.stack([emb_dict[str(t)] for t in df["tweet_id"]], axis=0)

    # Build node features
    X = np.concatenate([
        aligned_embs,
        perspective_feats,
        topic_onehot,
        depth_norm,
        confidence
    ], axis=1).astype(np.float32)

    x = torch.from_numpy(X)

    # Build edges
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

    # Conversation edges
    conv_src, conv_dst = [], []
    for conv_id, group in df.groupby("conversation_id"):
        root = group.index[group["tweet_id"] == conv_id][0]
        for idx in group.index:
            if idx != root:
                conv_src.extend([root, idx])
                conv_dst.extend([idx, root])
    conv_edge_index = torch.tensor([conv_src, conv_dst], dtype=torch.long)

    # KNN edges
    knn_edge_index = build_knn_edges(aligned_embs, k=3, threshold=0.75)

    # Combine edges
    edge_index = torch.cat([reply_edge_index, conv_edge_index, knn_edge_index], dim=1)

    # Masks
    train_idx, test_idx = splits[fold]
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Save graph
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        test_mask=test_mask,
    )

    torch.save(data, TRAINING_DIR / f"embed_graph_fold_{fold}.pt")
    print(f"Saved: embed_graph_fold_{fold}.pt")

print("\nAll folds complete!")
