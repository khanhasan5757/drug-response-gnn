import pandas as pd
from pubchempy import get_compounds

# Load drug names
drugs = pd.read_csv("data/raw/screened_compounds_rel_8.5.csv")

drug_names = drugs["DRUG_NAME"].dropna().unique()

records = []

print("Fetching SMILES from PubChem...")

for name in drug_names:
    try:
        compounds = get_compounds(name, "name")
        if len(compounds) > 0:
            smiles = compounds[0].canonical_smiles
            records.append({
                "drug_name": name,
                "smiles": smiles
            })
            print(f"✔ {name}")
        else:
            print(f"✖ {name}")
    except Exception as e:
        print(f"Error for {name}: {e}")

df = pd.DataFrame(records)
df.to_csv("data/processed/drug_smiles_pubchem.csv", index=False)

print("\nSaved: data/processed/drug_smiles_pubchem.csv")
