import json
import matplotlib.pyplot as plt

h = json.load(open('results/logs/training_log.json'))

epochs = [x['epoch'] for x in h]
rmse = [x['rmse'] for x in h]

best = min(rmse)
best_epoch = epochs[rmse.index(best)]

plt.figure(figsize=(10,7))
plt.plot(epochs, rmse, linewidth=3)
plt.axhline(best, linestyle='--')
plt.scatter(best_epoch, best, s=120)

plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Validation RMSE', fontsize=16)
plt.title('Validation RMSE with Best Epoch Highlight', fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig('rmse_highlight.png', dpi=400)
plt.show()

print("RMSE highlight plot saved.")