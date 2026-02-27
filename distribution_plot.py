import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model_architecture import DrugResponsePredictor
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import Subset

# Load config
config = yaml.safe_load(open("config/config.yaml"))
device = torch.device("cpu")

# Load data
train_loader, val_loader = get_dataloaders(config)

small_dataset = Subset(val_loader.dataset, range(2000))
val_loader = GeoDataLoader(small_dataset, batch_size=32, shuffle=False)

# Load model
model = DrugResponsePredictor(
    gene_dim=config["gene_dim"],
    embedding_dim=config["embedding_dim"],
    num_heads=config["num_heads"],
    dropout=config["dropout"]
)

model.load_state_dict(torch.load("results/models/best_model.pt", map_location=device))
model.to(device)
model.eval()

true_vals = []
pred_vals = []

with torch.no_grad():
    for drug_graph, gene_expr, labels in val_loader:
        drug_graph = drug_graph.to(device)
        gene_expr = gene_expr.to(device)
        labels = labels.to(device)
        preds = model(drug_graph, gene_expr)
        true_vals.extend(labels.cpu().numpy())
        pred_vals.extend(preds.cpu().numpy())

true_vals = np.array(true_vals).flatten()
pred_vals = np.array(pred_vals).flatten()

plt.figure(figsize=(10,7))
plt.hist(true_vals, bins=60, alpha=0.6, label="True IC50")
plt.hist(pred_vals, bins=60, alpha=0.6, label="Predicted IC50")

plt.xlabel("IC50 Value", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.title("True vs Predicted IC50 Distribution", fontsize=18, fontweight="bold")
plt.legend()

plt.tight_layout()
plt.savefig("true_vs_pred_distribution.png", dpi=400)
plt.show()

print("Distribution plot saved.")