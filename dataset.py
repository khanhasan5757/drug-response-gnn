"""
GDSC Drug Response Dataset Loader
=================================

Loads real GDSC dataset with SMILES and IC50 values.

Each batch returns:
    - drug_graph  : PyG Batch object
    - gene_expr   : Tensor [batch_size, gene_dim]
    - ic50        : Tensor [batch_size, 1]

Compatible with:
    torch_geometric.loader.DataLoader
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from torch_geometric.loader import DataLoader

from smiles_to_graph import SMILEStoGraphConverter


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

class GDSCDrugResponseDataset(Dataset):
    """
    Dataset returning:
        (graph, gene_expression, ic50)
    """

    def __init__(self, csv_path: str, gene_dim: int = 12072):

        self.df = pd.read_csv(csv_path)

        self.gene_dim = gene_dim
        self.converter = SMILEStoGraphConverter()

        print(f"Loaded dataset with {len(self.df):,} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        smiles = row["smiles"]
        ic50 = float(row["ic50"])

        # Convert SMILES → molecular graph
        graph = self.converter.smiles_to_graph(smiles)

        if graph is None:
            raise ValueError(f"Invalid SMILES at index {idx}")

        # Placeholder gene expression
        # (Later replace with CCLE expression vector)
        gene_expression = torch.randn(self.gene_dim)

        label = torch.tensor([ic50], dtype=torch.float)

        return graph, gene_expression, label


# ------------------------------------------------------------
# Dataloaders
# ------------------------------------------------------------

def get_dataloaders(config):

    dataset = GDSCDrugResponseDataset(
        csv_path="data/processed/final_gdsc_dataset.csv",
        gene_dim=config.get("gene_dim", 12072)
    )

    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=0.2,
        random_state=42
    )

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False
    )

    return train_loader, val_loader
