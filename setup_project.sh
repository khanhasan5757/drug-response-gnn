# Quick Reference: Implementation Snippets & Copy-Paste Code

## 1️⃣ COMPLETE SETUP SCRIPT

Save as `setup_project.sh`:

```bash
#!/bin/bash

# Create project structure
mkdir -p drug-gene-gnn/{data/{raw,processed},src/{data_processing,models,training,explainability,utils},notebooks,results/{models,logs,predictions,figures},config,tests}

cd drug-gene-gnn

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install rdkit scikit-learn pandas numpy scipy matplotlib seaborn plotly tensorboard jupyter pytest pyyaml

# Create main config file
cat > config/config.yaml << 'EOF'
# Model Configuration
gene_dim: 12072
embedding_dim: 128
num_heads: 4
dropout: 0.2

# Training Configuration
learning_rate: 0.001
weight_decay: 1e-5
batch_size: 32
num_epochs: 200
early_stopping_patience: 20

# Learning Rate Schedule
lr_decay_steps: 50
lr_decay_gamma: 0.5

# Directories
results_dir: results
data_dir: data/processed
log_interval: 10
EOF

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch==2.1.0
torch-geometric==2.4.0
rdkit==2023.09.1
scikit-learn==1.2.2
pandas==2.0.0
numpy==1.24.0
scipy==1.10.1
matplotlib==3.7.0
seaborn==0.12.2
plotly==5.14.0
tensorboard==2.13.0
pyyaml==6.0
jupyter==1.0.0
pytest==7.3.1
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Data
data/raw/
data/processed/
*.pkl
*.pickle

# Results
results/
*.pt
*.pth

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

echo "✅ Project structure created!"
echo "Next: Download GDSC & CCLE data into data/raw/"
```

Run with: `bash setup_project.sh`

---

## 2️⃣ DATA DOWNLOAD & VERIFICATION

Save as `data/download_data.py`:

```python
import pandas as pd
import os
from pathlib import Path

def download_gdsc():
    """
    Download GDSC data (manual download required)
    """
    print("""
    ========== GDSC Download Instructions ==========
    1. Go to: https://www.cancerrxgene.org/
    2. Download:
       - IC50/AUC values (e.g., GDSC2_public_raw_data.xlsx)
       - Cell line metadata
       - Drug metadata with SMILES
    3. Place files in: data/raw/gdsc/
    ============================================
    """)

def download_ccle():
    """
    Download CCLE data (manual download required)
    """
    print("""
    ========== CCLE Download Instructions ==========
    1. Go to: https://sites.broadinstitute.org/ccle
    2. Download:
       - CCLE RNA-seq (RSEM)
       - Cell line metadata
    3. Place files in: data/raw/ccle/
    ============================================
    """)

def verify_data():
    """Verify downloaded data integrity"""
    print("\n[INFO] Verifying downloaded data...\n")
    
    gdsc_dir = Path("data/raw/gdsc")
    ccle_dir = Path("data/raw/ccle")
    
    # Check GDSC
    if gdsc_dir.exists():
        files = list(gdsc_dir.glob("*"))
        print(f"✓ GDSC folder found with {len(files)} files")
        for f in files:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    else:
        print("✗ GDSC folder not found")
    
    # Check CCLE
    if ccle_dir.exists():
        files = list(ccle_dir.glob("*"))
        print(f"✓ CCLE folder found with {len(files)} files")
        for f in files:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
    else:
        print("✗ CCLE folder not found")

if __name__ == "__main__":
    download_gdsc()
    download_ccle()
    verify_data()
```

---

## 3️⃣ MINIMAL TRAINING EXAMPLE

Save as `train_minimal.py`:

```python
import torch
from torch_geometric.data import DataLoader, Batch
from src.models.predictor import DrugResponsePredictor
from src.training.trainer import DrugResponseTrainer
import yaml

# Load config
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

# Initialize model
model = DrugResponsePredictor(
    gene_dim=config['gene_dim'],
    embedding_dim=config['embedding_dim'],
    num_heads=config['num_heads'],
    dropout=config['dropout']
)

# Initialize trainer
trainer = DrugResponseTrainer(model, 'config/config.yaml', device='cuda')

# Dummy data for testing (replace with actual DataLoader)
# In real scenario: train_loader = DataLoader(train_dataset, batch_size=32)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train (with real data)
# trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'])
# trainer.save_checkpoint({}, is_best=True)

print("✓ Model initialized successfully!")
```

---

## 4️⃣ PREDICTION WITH TRAINED MODEL

Save as `predict.py`:

```python
import torch
import pandas as pd
from src.models.predictor import DrugResponsePredictor
from src.data_processing.smiles_to_graph import SMILEStoGraphConverter
from torch_geometric.data import Batch

def predict_drug_response(model_path, drug_smiles, gene_expr_vector):
    """
    Predict IC50 for drug + cell line combination
    
    Args:
        model_path (str): Path to trained model
        drug_smiles (str): SMILES string of drug
        gene_expr_vector (torch.Tensor): Gene expression (12072,)
    
    Returns:
        float: Predicted IC50 value
    """
    # Load model
    model = DrugResponsePredictor()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Convert SMILES to graph
    converter = SMILEStoGraphConverter()
    drug_graph = converter.smiles_to_graph(drug_smiles)
    
    # Ensure correct device
    device = next(model.parameters()).device
    drug_graph = drug_graph.to(device)
    gene_expr = gene_expr_vector.to(device)
    
    # Predict
    with torch.no_grad():
        # Add batch dimension
        drug_batch = Batch.from_data_list([drug_graph])
        gene_batch = gene_expr.unsqueeze(0) if gene_expr.dim() == 1 else gene_expr
        
        ic50_pred = model(drug_batch, gene_batch)
    
    return ic50_pred.item()

# Example usage
if __name__ == "__main__":
    # Drug: Imatinib (Gleevec)
    imatinib_smiles = "CC(=O)Nc1ccc(cc1)NC(=O)c2ccc(cc2)NC(=O)C"
    
    # Dummy cell line expression (use real data)
    gene_expr = torch.randn(12072)
    
    # Predict
    ic50 = predict_drug_response(
        'results/models/best_model.pt',
        imatinib_smiles,
        gene_expr
    )
    
    print(f"Predicted IC50: {ic50:.4f} µM")
```

---

## 5️⃣ BATCH INFERENCE ON MULTIPLE DRUGS

```python
import pandas as pd
import torch
from src.models.predictor import DrugResponsePredictor
from src.data_processing.smiles_to_graph import SMILEStoGraphConverter
from torch_geometric.data import Batch
from tqdm import tqdm

def batch_predict(model_path, drugs_df, cell_line_expr):
    """
    Predict IC50 for multiple drugs × one cell line
    
    Args:
        model_path (str): Path to trained model
        drugs_df (pd.DataFrame): Columns: ['drug_name', 'smiles']
        cell_line_expr (torch.Tensor): Gene expression [12072]
    
    Returns:
        pd.DataFrame: Predictions
    """
    model = DrugResponsePredictor()
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()
    
    device = next(model.parameters()).device
    converter = SMILEStoGraphConverter()
    
    results = []
    
    for idx, row in tqdm(drugs_df.iterrows(), total=len(drugs_df)):
        drug_name = row['drug_name']
        smiles = row['smiles']
        
        # Convert to graph
        drug_graph = converter.smiles_to_graph(smiles)
        if drug_graph is None:
            results.append({
                'drug': drug_name,
                'ic50_pred': None,
                'error': 'Invalid SMILES'
            })
            continue
        
        # Predict
        with torch.no_grad():
            drug_batch = Batch.from_data_list([drug_graph.to(device)])
            gene_batch = cell_line_expr.unsqueeze(0).to(device)
            ic50 = model(drug_batch, gene_batch).item()
        
        results.append({
            'drug': drug_name,
            'ic50_pred': ic50,
            'error': None
        })
    
    return pd.DataFrame(results)

# Example
drugs = pd.DataFrame({
    'drug_name': ['Imatinib', 'Erlotinib', 'Sorafenib'],
    'smiles': [
        'CC(=O)Nc1ccc(cc1)NC(=O)c2ccc(cc2)NC(=O)C',
        'COc1cc2nccnc2cc1OCCCN1CCOCC1',
        'NC(=O)c1cc(Oc2ccc(NC(=O)c3ccc(Cl)c(Cl)c3)cc2)ccn1'
    ]
})

results = batch_predict('results/models/best_model.pt', drugs, torch.randn(12072))
print(results)
```

---

## 6️⃣ VISUALIZATION HELPERS

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_predictions_vs_actual(y_true, y_pred, title="Drug Response Prediction"):
    """Scatter plot of predictions vs actual"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual IC50')
    axes[0].set_ylabel('Predicted IC50')
    axes[0].set_title(title)
    
    # Residuals
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel('Residual')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Error Distribution')
    
    plt.tight_layout()
    return fig

def plot_training_curves(train_losses, val_losses, metrics_history):
    """Plot training and validation loss"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics
    metrics = metrics_history
    if metrics:
        epochs = range(len(metrics))
        r2_scores = [m.get('r2_score', 0) for m in metrics]
        axes[1].plot(epochs, r2_scores, 'g-', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('R² Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    return fig
```

---

## 7️⃣ TESTING TEMPLATE

Save as `tests/test_basic.py`:

```python
import pytest
import torch
from src.models.predictor import DrugResponsePredictor
from src.data_processing.smiles_to_graph import SMILEStoGraphConverter

class TestSMILESConversion:
    def test_valid_smiles(self):
        converter = SMILEStoGraphConverter()
        graph = converter.smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O")
        assert graph is not None
        assert graph.x.shape[0] > 0  # Has atoms
        assert graph.edge_index.shape[1] > 0  # Has bonds
    
    def test_invalid_smiles(self):
        converter = SMILEStoGraphConverter()
        graph = converter.smiles_to_graph("INVALID_SMILES")
        assert graph is None

class TestModel:
    def test_model_initialization(self):
        model = DrugResponsePredictor(gene_dim=12072)
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0
    
    def test_forward_pass(self):
        from torch_geometric.data import Batch, Data
        
        model = DrugResponsePredictor()
        
        # Create dummy drug graph
        drug_graph = Data(
            x=torch.randn(10, 44),  # 10 atoms
            edge_index=torch.randint(0, 10, (2, 15)),  # 15 bonds
        )
        drug_batch = Batch.from_data_list([drug_graph])
        
        # Gene expression
        gene_expr = torch.randn(1, 12072)
        
        # Forward pass
        output = model(drug_batch, gene_expr)
        assert output.shape == (1, 1)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run tests: `python -m pytest tests/ -v`

---

## 8️⃣ GITHUB ACTIONS CI/CD

Save as `.github/workflows/tests.yml`:

```yaml
name: Run Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, "3.10"]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install torch-geometric
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
    
    - name: Check code style
      run: |
        pip install flake8 black
        black --check src/
        flake8 src/ --max-line-length=100
```

---

## 9️⃣ QUICK TROUBLESHOOTING

### Problem: "ModuleNotFoundError: No module named 'torch_geometric'"

```bash
pip install torch-geometric
# If still fails, try:
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv
```

### Problem: GPU Out of Memory

```python
# Option 1: Reduce batch size
batch_size = 16  # Instead of 32

# Option 2: Gradient accumulation
for i, batch in enumerate(train_loader):
    # ... forward & backward ...
    if (i + 1) % 2 == 0:  # Accumulate 2 batches
        optimizer.step()
        optimizer.zero_grad()

# Option 3: Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(...)
scaler.scale(loss).backward()
```

### Problem: "RuntimeError: CUDA out of memory"

```python
# Clear cache
torch.cuda.empty_cache()

# Move non-essential data to CPU
model.to('cpu')
```

---

## 🔟 DEPLOYMENT CHECKLIST

Before submitting your project:

```bash
# 1. Clean code
black src/
flake8 src/

# 2. Run tests
pytest tests/ -v

# 3. Generate results
python train.py --config config/config.yaml

# 4. Check notebooks
jupyter nbconvert --to html notebooks/*.ipynb

# 5. Create tarball
tar -czf drug-gene-gnn.tar.gz --exclude=data --exclude=.git .

# 6. Final verification
git status  # Should be clean
ls -lh results/models/best_model.pt  # Should exist
```

---

**You've got all the code you need! Now just execute the checklist. Good luck! 🚀**
