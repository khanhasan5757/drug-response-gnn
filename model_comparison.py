import matplotlib.pyplot as plt

models = ['Linear Regression', 'Random Forest', 'MLP Baseline', 'GNN + Attention (Ours)']
rmse = [3.8, 3.4, 3.1, 2.66]  # reference comparison values

plt.figure(figsize=(10,7))
bars = plt.bar(models, rmse)

plt.ylabel('Validation RMSE', fontsize=16)
plt.title('Model Performance Comparison on GDSC', fontsize=18, fontweight='bold')

plt.xticks(rotation=20, fontsize=12)
plt.yticks(fontsize=14)

# Highlight your model
bars[-1].set_linewidth(3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=400)
plt.show()

print("Model comparison plot saved.")