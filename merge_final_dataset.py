"""
FINAL GDSC DATASET BUILDER
--------------------------
Uses community drug annotation with canonical SMILES.

Output:
data/processed/final_gdsc_dataset.csv
"""

import pandas as pd


# --------------------------------------------------
# Normalize drug names
# --------------------------------------------------

def normalize(x):
    x = str(x).lower()
    x = x.replace("-", " ")
    x = x.replace("_", " ")
    x = x.replace(" hydrochloride", "")
    x = x.replace(" hydrobromide", "")
    x = x.replace(" mesylate", "")
    x = x.replace(" phosphate", "")
    x = x.replace(" sulfate", "")
    x = x.replace(" sodium", "")
    x = x.replace(" potassium", "")
    x = x.replace(",", "")
    x = x.strip()
    return x


# --------------------------------------------------
# Load files
# --------------------------------------------------

print("📥 Loading datasets...")

ic50 = pd.read_csv("data/processed/GDSC1_cleaned_v2.csv")
drug_anno = pd.read_csv("data/raw/GDSC_DrugAnnotation.csv")

print("IC50:", ic50.shape)
print("Drug annotations:", drug_anno.shape)


# --------------------------------------------------
# Prepare drug annotation table
# --------------------------------------------------

drug_anno = drug_anno.rename(columns={
    "Unnamed: 0": "drug_name",
    "CanonicalSMILESrdkit": "smiles"
})

drug_anno["drug_name_clean"] = drug_anno["drug_name"].apply(normalize)

drug_anno = drug_anno[["drug_name_clean", "smiles"]]


# --------------------------------------------------
# Prepare IC50 table
# --------------------------------------------------

ic50["drug_name_clean"] = ic50["drug_name"].apply(normalize)


# --------------------------------------------------
# Merge
# --------------------------------------------------

df = ic50.merge(
    drug_anno,
    on="drug_name_clean",
    how="inner"
)

print("After merge:", df.shape)


# --------------------------------------------------
# Final dataset
# --------------------------------------------------

final_df = df[[
    "drug_name",
    "smiles",
    "cosmic_id",
    "cell_line_name",
    "ln_ic50"
]]

final_df = final_df.rename(columns={
    "cosmic_id": "cell_line",
    "ln_ic50": "ic50"
})


# --------------------------------------------------
# Save
# --------------------------------------------------

final_df.to_csv(
    "data/processed/final_gdsc_dataset.csv",
    index=False
)

print("\n✅ FINAL DATASET CREATED")
print("Rows:", final_df.shape[0])
print("Columns:", final_df.shape[1])
print(final_df.head())
