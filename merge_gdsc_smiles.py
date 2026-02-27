import pandas as pd

# Load IC50 data
ic50 = pd.read_csv("data/processed/GDSC1_cleaned_v2.csv")

# Load compound annotations
compounds = pd.read_csv("data/raw/compound_annotations.csv")

# Keep only required columns
compounds = compounds[["DRUG_ID", "CANONICAL_SMILES"]]

# Rename for merge
compounds = compounds.rename(columns={
    "DRUG_ID": "drug_id",
    "CANONICAL_SMILES": "smiles"
})

# Merge
merged = ic50.merge(compounds, on="drug_id", how="inner")

print("Merged dataset size:", merged.shape)
print(merged.head())

# Save
merged.to_csv("data/processed/gdsc_with_smiles.csv", index=False)

print("\n✅ Saved: data/processed/gdsc_with_smiles.csv")
