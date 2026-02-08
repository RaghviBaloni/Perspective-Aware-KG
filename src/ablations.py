import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import f1_score
import numpy as np
from pathlib import Path
import json

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# Feature selection
# ---------------------------
def select_features(x, config, num_topics):
    idx = 0
    parts = []

    # Embeddings
    if config["emb"]:
        parts.append(x[:, idx:idx+768])
    idx += 768

    # Perspectives
    if config["pers"]:
        parts.append(x[:, idx:idx+6])
    idx += 6

    # Topic
    if config["topic"]:
        parts.append(x[:, idx:idx+num_topics])
    idx += num_topics

    # Depth
    if config["depth"]:
        parts.append(x[:, idx:idx+1])
    idx += 1

    # Confidence
    if config["conf"]:
        parts.append(x[:, idx:idx+1])
    idx += 1

    return torch.cat(parts, dim=1)


# ---------------------------
# Simple MLP
# ---------------------------
class MLP(torch.nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=4):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------
# Run ablation
# ---------------------------
def run_ablation(config, fold, num_topics, data):
    x = select_features(data.x, config, num_topics)
    in_dim = x.shape[1]

    model = MLP(in_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(x[data.train_mask])
        loss = F.cross_entropy(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    # Test
    model.eval()
    out = model(x)
    pred = out.argmax(dim=1)

    y_true = data.y[data.test_mask].cpu().numpy()
    y_pred = pred[data.test_mask].cpu().numpy()

    return {
        "acc": (y_pred == y_true).mean(),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted")
    }
# Add this to the end of the file, replacing the incomplete if __name__ block:

if __name__ == "__main__":
    DATA_DIR = Path("/home/baloni/Perspective-Aware-KG/data/baseline_models")
    RESULTS_DIR = Path("/home/baloni/Perspective-Aware-KG/results")
    RESULTS_DIR.mkdir(exist_ok=True)
    
    configs = { 
        "emb_only": {"emb": True, "pers": False, "topic": False, "depth": False, "conf": False}, 
        "emb_pers": {"emb": True, "pers": True, "topic": False, "depth": False, "conf": False}, 
        "embs_topic": {"emb": True, "pers": False, "topic": True, "depth": False, "conf": False}, 
        "embs_depth": {"emb": True, "pers": False, "topic": False, "depth": True, "conf": False}, 
        "embs_conf": {"emb": True, "pers": False, "topic": False, "depth": False, "conf": True},
        "embs_topic": {"emb": True, "pers": False, "topic": True, "depth": False, "conf": False}, 
        "all": {"emb": True, "pers": True, "topic": True, "depth": True, "conf": True}
    }
    
    # Determine num_topics by loading first fold
    data_sample = torch.load(DATA_DIR / "embed_graph_fold_0.pt", weights_only=False)
    num_topics = data_sample.x.shape[1] - 768 - 6 - 1 - 1  # total - emb - pers - depth - conf
    
    # Store results
    ablation_results = {}
    
    # Run ablations for each configuration
    for config_name, config in configs.items():
        print(f"\n{'='*50}")
        print(f"Running ablation: {config_name}")
        print(f"{'='*50}")
        
        fold_results = []
        
        # Run across all 10 folds
        for fold in range(10):
            print(f"\nFold {fold}...")
            
            # Load data for this fold
            data = torch.load(DATA_DIR / f"embed_graph_fold_{fold}.pt", weights_only=False).to(DEVICE)
            
            # Run ablation
            metrics = run_ablation(config, fold, num_topics, data)
            fold_results.append(metrics)
            
            print(f"  Acc: {metrics['acc']:.4f}")
            print(f"  Macro F1: {metrics['macro_f1']:.4f}")
            print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        
        # Aggregate results for this configuration
        accs = [r["acc"] for r in fold_results]
        macro_f1s = [r["macro_f1"] for r in fold_results]
        weighted_f1s = [r["weighted_f1"] for r in fold_results]
        
        ablation_results[config_name] = {
            "acc_mean": np.mean(accs),
            "acc_std": np.std(accs),
            "macro_f1_mean": np.mean(macro_f1s),
            "macro_f1_std": np.std(macro_f1s),
            "weighted_f1_mean": np.mean(weighted_f1s),
            "weighted_f1_std": np.std(weighted_f1s),
            "fold_results": fold_results
        }
        
        print(f"\n{config_name} - Summary:")
        print(f"  Accuracy: {ablation_results[config_name]['acc_mean']:.4f} ± {ablation_results[config_name]['acc_std']:.4f}")
        print(f"  Macro F1: {ablation_results[config_name]['macro_f1_mean']:.4f} ± {ablation_results[config_name]['macro_f1_std']:.4f}")
        print(f"  Weighted F1: {ablation_results[config_name]['weighted_f1_mean']:.4f} ± {ablation_results[config_name]['weighted_f1_std']:.4f}")
    
    # Save results to JSON
    results_file = RESULTS_DIR / "ablation_results_1.json"
    # Convert numpy values to Python floats for JSON serialization
    results_serializable = {}
    for config_name, metrics in ablation_results.items():
        results_serializable[config_name] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in metrics.items()
            if k != "fold_results"
        }
    
    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results saved to {results_file}")
    print(f"{'='*50}")