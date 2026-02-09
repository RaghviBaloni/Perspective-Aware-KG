
import torch
import json

# Load one fold
data = torch.load("/data/training/embed_graph_fold_1.pt", weights_only=False)
metadata = json.load(open("/data/graph/metadata_inductive.json"))

print("Graph loaded successfully!")
print(f"  Nodes: {data.x.shape[0]}")
print(f"  Features: {data.x.shape[1]}")
print(f"  Edges: {data.edge_index.shape[1]}")
print(f"  Labels: {data.y.shape[0]}")
print(f"\nMasks:")
print(f"  Train: {data.train_mask.sum().item()}")
print(f"  Test: {data.test_mask.sum().item()}")
print(f"\nLabel distribution (train):")
train_labels = data.y[data.train_mask].numpy()
import numpy as np
print(np.bincount(train_labels))

# Check for issues
assert data.x.shape[0] == data.y.shape[0], "Feature/label mismatch"
assert data.train_mask.sum() + data.test_mask.sum() == data.x.shape[0], "Mask coverage issue"
assert data.edge_index.max() < data.x.shape[0], "Edge index out of bounds"

print("\n All checks passed!")

#Edge Connectivity

import networkx as nx

# Convert to NetworkX for analysis
G = nx.Graph()
G.add_nodes_from(range(data.x.shape[0]))
edge_list = data.edge_index.t().tolist()
G.add_edges_from(edge_list)

print(f"Graph is connected: {nx.is_connected(G)}")
print(f"Number of connected components: {nx.number_connected_components(G)}")
print(f"Average clustering coefficient: {nx.average_clustering(G):.4f}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

#Feature Quality

# Check for NaN/Inf
print(f"NaN in features: {torch.isnan(data.x).any()}")
print(f"Inf in features: {torch.isinf(data.x).any()}")

# Check feature ranges
print(f"\nFeature statistics:")
print(f"  Min: {data.x.min().item():.4f}")
print(f"  Max: {data.x.max().item():.4f}")
print(f"  Mean: {data.x.mean().item():.4f}")
print(f"  Std: {data.x.std().item():.4f}")

# Check perspectives (should be binary)
persp_features = data.x[:, 768:774]  # Assuming perspectives start at 768
print(f"\nPerspective feature unique values: {torch.unique(persp_features)}")

