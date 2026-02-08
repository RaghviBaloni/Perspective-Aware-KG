import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, LayerNorm
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import classification_report, f1_score
from pathlib import Path
import json
from datetime import datetime

#-----Config-----
GRAPH_DIR = Path("/home/baloni/Perspective-Aware-KG/data/graph")
TRAINING_DIR = Path("/home/baloni/Perspective-Aware-KG/data/training")
OUTPUT_DIR = Path("/home/baloni/Perspective-Aware-KG/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_DIM = 512
NUM_LAYERS = 3
DROPOUT = 0.5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
NUM_EPOCHS = 300
PATIENCE = 30  # early stopping (currently not used if commented)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#-----Models-----
class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_channels, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout
    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x

class GNNRefiner(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, model_type='GCN', dropout=0.5):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout

        if model_type == 'GCN':
            Conv = GCNConv
        elif model_type == 'GAT':
            Conv = lambda in_c, out_c: GATConv(in_c, out_c, heads=4, concat=False, dropout=dropout)
        elif model_type == 'SAGE':
            Conv = SAGEConv
        else:
            raise ValueError(f"Unknown GNN type: {model_type}")

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(Conv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        if num_layers > 1:
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

class HybridMLP_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_mlp, hidden_gnn, out_channels,
                 gnn_layers=2, gnn_type='GCN', dropout=0.5):
        super().__init__()
        self.mlp = MLP(in_channels, hidden_mlp, out_channels, dropout=dropout)
        self.gnn = GNNRefiner(out_channels, hidden_gnn, out_channels,
                              num_layers=gnn_layers, model_type=gnn_type, dropout=dropout)

    def forward(self, x, edge_index):
        mlp_logits = self.mlp(x)
        gnn_logits = self.gnn(mlp_logits, edge_index)
        return mlp_logits + gnn_logits

class ArgumentGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads=4, concat=True, dropout=DROPOUT))
        self.norms.append(LayerNorm(hidden_channels * 4))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=True, dropout=DROPOUT))
            self.norms.append(LayerNorm(hidden_channels * 4))

        self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4, concat=False, dropout=DROPOUT))
        self.norms.append(LayerNorm(hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = DROPOUT

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

class ArgumentGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = DROPOUT

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

class ArgumentSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(LayerNorm(hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.norms.append(LayerNorm(hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = DROPOUT

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms[:-1]):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.norms[-1](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

#-----Training / Eval-----
def train_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_on_mask(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred[mask].cpu().numpy()

    correct = (pred[mask] == data.y[mask]).sum()
    acc = (correct / mask.sum()).item()

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_per_class = f1_score(y_true, y_pred, average=None)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'y_true': y_true,
        'y_pred': y_pred,
    }

#-----Main Loop-----
def run_experiment(model_name='MLP'):
    print('=' * 70)
    print(f"INDUCTIVE GNN Training: {model_name}")
    print('=' * 70)

    label_names = ['reason', 'statement', 'notification', 'none']
    all_results = []

    for fold in range(10):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/10")
        print(f"{'='*70}")

        # 1) Load TRAIN graph (only train nodes)
        data_train = torch.load(TRAINING_DIR / f"train_graph_fold_{fold}.pt", weights_only=False).to(DEVICE)

        # 2) Build model
        in_channels = data_train.x.shape[1]

        if model_name == 'GAT':
            model = ArgumentGAT(
                in_channels=in_channels,
                hidden_channels=HIDDEN_DIM,
                out_channels=4,
                num_layers=NUM_LAYERS
            ).to(DEVICE)
        elif model_name == 'GCN':
            model = ArgumentGCN(
                in_channels=in_channels,
                hidden_channels=HIDDEN_DIM,
                out_channels=4,
                num_layers=NUM_LAYERS
            ).to(DEVICE)
        elif model_name == 'SAGE':
            model = ArgumentSAGE(
                in_channels=in_channels,
                hidden_channels=HIDDEN_DIM,
                out_channels=4,
                num_layers=NUM_LAYERS
            ).to(DEVICE)
        elif model_name == "MLP":
            model = MLP(
                in_channels=in_channels,
                hidden_channels=256,
                out_channels=4
            ).to(DEVICE)
        elif model_name == "HYBRID":
            model = HybridMLP_GNN(
                in_channels=in_channels,
                hidden_mlp=256,
                hidden_gnn=128,
                out_channels=4,
                gnn_layers=2,
                gnn_type='SAGE',
                dropout=DROPOUT
            ).to(DEVICE)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )

        train_mask = data_train.train_mask

        # 3) Train on TRAIN graph only (inductive)
        for epoch in range(NUM_EPOCHS):
            loss = train_epoch(model, data_train, optimizer, train_mask)
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f}")

        print(f"Final Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss:.4f}")

        # 4) Evaluate on INFERENCE graph (train + test nodes)
        data_infer = torch.load(TRAINING_DIR / f"infer_graph_fold_{fold}.pt", weights_only=False).to(DEVICE)

        test_results = evaluate_on_mask(model, data_infer, data_infer.test_mask)

        print(f"\n{'='*70}")
        print(f"Fold {fold} Results (INDUCTIVE EVAL):")
        print(f"{'='*70}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_results['f1_macro']:.4f}")
        print(f"Test F1 (weighted): {test_results['f1_weighted']:.4f}")

        print(f"\nPer-class F1:")
        for name, f1 in zip(label_names, test_results['f1_per_class']):
            print(f"  {name:15s}: {f1:.4f}")

        print(f"\nClassification Report:")
        print(classification_report(
            test_results['y_true'],
            test_results['y_pred'],
            target_names=label_names,
            digits=4
        ))

        all_results.append({
            'fold': fold,
            'test_accuracy': test_results['accuracy'],
            'test_f1_macro': test_results['f1_macro'],
            'test_f1_weighted': test_results['f1_weighted'],
            'test_f1_per_class': test_results['f1_per_class'].tolist(),
            'y_true': test_results['y_true'].tolist(),
            'y_pred': test_results['y_pred'].tolist(),
        })

    avg_acc = np.mean([r['test_accuracy'] for r in all_results])
    std_acc = np.std([r['test_accuracy'] for r in all_results])

    avg_f1_macro = np.mean([r['test_f1_macro'] for r in all_results])
    std_f1_macro = np.std([r['test_f1_macro'] for r in all_results])

    avg_f1_weighted = np.mean([r['test_f1_weighted'] for r in all_results])
    std_f1_weighted = np.std([r['test_f1_weighted'] for r in all_results])

    avg_f1_per_class = np.mean([r['test_f1_per_class'] for r in all_results], axis=0)

    print(f"\n{'='*70}")
    print(f"FINAL INDUCTIVE RESULTS: {model_name}")
    print(f"{'='*70}")
    print(f"Average Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    print(f"Average F1 (macro):    {avg_f1_macro:.4f} ± {std_f1_macro:.4f}")
    print(f"Average F1 (weighted): {avg_f1_weighted:.4f} ± {std_f1_weighted:.4f}")

    print(f"\nAverage Per-class F1:")
    for name, f1 in zip(label_names, avg_f1_per_class):
        print(f"  {name:15s}: {f1:.4f}")

    results_file = OUTPUT_DIR / f"{model_name}_inductive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'model': model_name,
            'mode': 'inductive',
            'hyperparameters': {
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'num_epochs': NUM_EPOCHS,
                'patience': PATIENCE,
            },
            'summary': {
                'avg_accuracy': avg_acc,
                'std_accuracy': std_acc,
                'avg_f1_macro': avg_f1_macro,
                'std_f1_macro': std_f1_macro,
                'avg_f1_weighted': avg_f1_weighted,
                'std_f1_weighted': std_f1_weighted,
                'avg_f1_per_class': avg_f1_per_class.tolist(),
            },
            'fold_results': all_results,
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return all_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train INDUCTIVE GNN on TACO dataset')
    parser.add_argument(
        '--model',
        type=str,
        default='MLP',
        choices=['GAT', 'GCN', 'SAGE', 'MLP', 'HYBRID'],
        help='GNN architecture to use'
    )

    args = parser.parse_args()

    print(f"\nDevice: {DEVICE}")
    print(f"Model: {args.model}")
    print(f"Hidden dim: {HIDDEN_DIM}")
    print(f"Num layers: {NUM_LAYERS}")
    print(f"Dropout: {DROPOUT}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}\n")

    run_experiment(model_name=args.model)
