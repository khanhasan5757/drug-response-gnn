"""
Training Script – CPU SAFE MODE (1 Epoch)
=========================================
Stable CPU test version
Author: Hasan Khan
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
import time

from dataset import get_dataloaders
from model_architecture import DrugResponsePredictor


# --------------------------------------------------
# Trainer
# --------------------------------------------------

class DrugResponseTrainer:

    def __init__(self, model, config, device):

        self.device = torch.device(device)
        self.model = model.to(self.device)

        self.config = config
        self.results_dir = config.get("results_dir", "results")
        self.model_dir = os.path.join(self.results_dir, "models")

        os.makedirs(self.model_dir, exist_ok=True)

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"])
        )

        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config["lr_decay_steps"],
            gamma=config["lr_decay_gamma"]
        )

        print("✅ Trainer initialized")
        print("🖥 Device:", self.device)

    # --------------------------------------------------

    def train_epoch(self, loader):

        self.model.train()
        total_loss = 0

        start_time = time.time()

        for drug_graph, gene_expr, labels in tqdm(loader):

            drug_graph = drug_graph.to(self.device)
            gene_expr = gene_expr.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            preds = self.model(drug_graph, gene_expr)
            loss = self.criterion(preds, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        epoch_time = (time.time() - start_time) / 60
        print(f"⏱ Epoch time: {epoch_time:.2f} minutes")

        return total_loss / len(loader)

    # --------------------------------------------------

    def validate(self, loader):

        self.model.eval()
        total_loss = 0

        preds_all = []
        labels_all = []

        with torch.no_grad():
            for drug_graph, gene_expr, labels in tqdm(loader):

                drug_graph = drug_graph.to(self.device)
                gene_expr = gene_expr.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(drug_graph, gene_expr)

                loss = self.criterion(preds, labels)
                total_loss += loss.item()

                preds_all.append(preds.cpu())
                labels_all.append(labels.cpu())

        preds_all = torch.cat(preds_all).numpy()
        labels_all = torch.cat(labels_all).numpy()

        rmse = np.sqrt(mean_squared_error(labels_all, preds_all))

        return {
            "val_loss": total_loss / len(loader),
            "rmse": rmse
        }

    # --------------------------------------------------

    def train(self, train_loader, val_loader):

        num_epochs = 1  # 🔥 FORCE 1 EPOCH ONLY

        for epoch in range(num_epochs):

            print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            self.scheduler.step()

            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss:   {val_metrics['val_loss']:.4f}")
            print(f"RMSE:       {val_metrics['rmse']:.4f}")

        print("\n🎉 CPU 1-Epoch Test Complete")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Torch version:", torch.__version__)
    print("🚀 Forcing device: CPU")

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    device = "cpu"  # 🔥 HARD FORCED CPU

    train_loader, val_loader = get_dataloaders(config)

    model = DrugResponsePredictor(
        gene_dim=config["gene_dim"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"]
    )

    trainer = DrugResponseTrainer(
        model=model,
        config=config,
        device=device
    )

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()