import matplotlib.pyplot as plt

labels = ['Drugs', 'Cell Lines', 'Drug–Cell Pairs']
values = [280, 1000, 253819]   # adjust if needed

plt.figure(figsize=(10,7))
plt.bar(labels, values)

plt.ylabel('Count', fontsize=16)
plt.title('Dataset Scale Overview (GDSC)', fontsize=18, fontweight='bold')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.savefig('dataset_scale.png', dpi=400)
plt.show()

print("Dataset scale plot saved.")