import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from dataset import get_dataloaders
from model_architecture import DrugResponsePredictor
import yaml
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import Subset

# -------------------------
# Load Configuration
# -------------------------
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

device = torch.device("cpu")

# -------------------------
# Load Data
# -------------------------
train_loader, val_loader = get_dataloaders(config)

# Take small subset for fast plotting
small_indices = list(range(2000))
small_dataset = Subset(val_loader.dataset, small_indices)

val_loader = GeoDataLoader(
    small_dataset,
    batch_size=32,
    shuffle=False
)

# -------------------------
# Load Model
# -------------------------
model = DrugResponsePredictor(
    gene_dim=config["gene_dim"],
    embedding_dim=config["embedding_dim"],
    num_heads=config["num_heads"],
    dropout=config["dropout"]
)

model.load_state_dict(torch.load("results/models/best_model.pt", map_location=device))
model.to(device)
model.eval()

# -------------------------
# Generate Predictions
# -------------------------
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

# -------------------------
# Compute R²
# -------------------------
r2 = r2_score(true_vals, pred_vals)
print(f"R² Score: {r2:.4f}")

# -------------------------
# Scatter Plot
# -------------------------
min_val = min(true_vals.min(), pred_vals.min())
max_val = max(true_vals.max(), pred_vals.max())

plt.figure(figsize=(8,8))
plt.scatter(true_vals, pred_vals, alpha=0.15, s=12)

plt.plot([min_val, max_val],
         [min_val, max_val],
         linewidth=2)

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)

plt.xlabel("True IC50", fontsize=16)
plt.ylabel("Predicted IC50", fontsize=16)
plt.title(f"Predicted vs True IC50 (R² = {r2:.3f})",
          fontsize=18,
          fontweight="bold")

plt.tight_layout()
plt.savefig("pred_vs_true_fixed.png", dpi=400)
plt.show()

# -------------------------
# Residual Distribution
# -------------------------
residuals = pred_vals - true_vals

plt.figure(figsize=(8,6))
plt.hist(residuals, bins=50)

plt.xlabel("Prediction Error (Residual)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.title("Residual Error Distribution",
          fontsize=16,
          fontweight="bold")

plt.tight_layout()
plt.savefig("residual_distribution.png", dpi=400)
plt.show()