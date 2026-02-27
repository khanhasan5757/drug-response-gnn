"""
Core Model Architecture: Drug-Gene Interaction Prediction
Integrates Graph Neural Networks (for drugs) with dense networks (for gene expression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ======================================================
# DRUG ENCODER
# ======================================================

class DrugEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 32,      # ✅ FIXED
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GATConv(input_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, concat=True)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim,
                        heads=num_heads, dropout=dropout, concat=True)
            )

        # Output layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, output_dim,
                    heads=1, concat=False, dropout=dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim * num_heads)
             for _ in range(num_layers - 1)]
        )

    def forward(self, x, edge_index, batch):

        for i, layer in enumerate(self.gat_layers[:-1]):
            x = layer(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.gat_layers[-1](x, edge_index)

        return global_mean_pool(x, batch)


# ======================================================
# CELL LINE ENCODER
# ======================================================

class CellLineEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int = 12072,
        hidden_dims=(1024, 512, 256),
        output_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()

        layers = []
        prev = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ======================================================
# ATTENTION FUSION
# ======================================================

class MultiHeadAttentionFusion(nn.Module):

    def __init__(self, embedding_dim=128, num_heads=4, dropout=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.fc = nn.Linear(embedding_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug, cell):

        B = drug.size(0)

        def reshape(x):
            return x.view(B, self.num_heads, self.head_dim).transpose(0, 1)

        Q1 = reshape(self.query(drug))
        K1 = reshape(self.key(cell))
        V1 = reshape(self.value(cell))

        attn1 = torch.softmax(Q1 @ K1.transpose(-2, -1) / self.head_dim**0.5, dim=-1)
        ctx1 = (attn1 @ V1).transpose(0, 1).reshape(B, -1)

        Q2 = reshape(self.query(cell))
        K2 = reshape(self.key(drug))
        V2 = reshape(self.value(drug))

        attn2 = torch.softmax(Q2 @ K2.transpose(-2, -1) / self.head_dim**0.5, dim=-1)
        ctx2 = (attn2 @ V2).transpose(0, 1).reshape(B, -1)

        return self.fc(torch.cat([ctx1, ctx2], dim=1))


# ======================================================
# FULL MODEL
# ======================================================

class DrugResponsePredictor(nn.Module):

    def __init__(
        self,
        gene_dim=12072,
        embedding_dim=128,
        num_heads=4,
        dropout=0.2
    ):
        super().__init__()

        self.drug_encoder = DrugEncoder(
            input_dim=32,      # ✅ FIXED
            output_dim=embedding_dim,
            dropout=dropout
        )

        self.cell_encoder = CellLineEncoder(
            input_dim=gene_dim,
            output_dim=embedding_dim,
            dropout=dropout
        )

        self.fusion = MultiHeadAttentionFusion(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 1)
        )

    def forward(self, drug_graph, gene_expression):

        drug_embed = self.drug_encoder(
            drug_graph.x,
            drug_graph.edge_index,
            drug_graph.batch
        )

        cell_embed = self.cell_encoder(gene_expression)

        fused = self.fusion(drug_embed, cell_embed)

        return self.predictor(fused)
