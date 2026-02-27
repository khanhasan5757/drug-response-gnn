"""
SMILES to Molecular Graph Conversion Module
Converts drug SMILES strings to PyTorch Geometric Graph objects using RDKit
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from typing import List, Optional


class SMILEStoGraphConverter:
    """
    Converts SMILES strings into PyTorch Geometric graph objects.

    Atom features: 32D
    Bond features: 6D
    """

    ALLOWED_BONDS = [
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC
    ]

    HYBRIDIZATIONS = [
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
        Chem.HybridizationType.SP3D,
        Chem.HybridizationType.SP3D2
    ]

    def __init__(self):
        self.atom_feature_dim = 32
        self.bond_feature_dim = 6

    # --------------------------------------------------

    def _get_atom_features(self, atom: Chem.Atom):

        atom_map = {
            1: 0,   # H
            6: 1,   # C
            7: 2,   # N
            8: 3,   # O
            9: 4,   # F
            16: 5,  # S
            17: 6,  # Cl
            35: 7,  # Br
            53: 8,  # I
            15: 9   # P
        }

        atomic = [0] * 10
        if atom.GetAtomicNum() in atom_map:
            atomic[atom_map[atom.GetAtomicNum()]] = 1

        degree = [0] * 6
        degree[min(atom.GetDegree(), 5)] = 1

        hydrogens = [0] * 6
        hydrogens[min(atom.GetTotalNumHs(), 5)] = 1

        charge = [0, 0, 0]
        if atom.GetFormalCharge() == -1:
            charge[0] = 1
        elif atom.GetFormalCharge() == 0:
            charge[1] = 1
        elif atom.GetFormalCharge() == 1:
            charge[2] = 1

        aromatic = [int(atom.GetIsAromatic())]

        hybrid = [0] * 5
        if atom.GetHybridization() in self.HYBRIDIZATIONS:
            hybrid[self.HYBRIDIZATIONS.index(atom.GetHybridization())] = 1

        ring = [int(atom.IsInRing())]

        features = (
            atomic +
            degree +
            hydrogens +
            charge +
            aromatic +
            hybrid +
            ring
        )

        return np.array(features, dtype=np.float32)

    # --------------------------------------------------

    def _get_bond_features(self, bond: Chem.Bond):

        bond_type = [0] * 4
        if bond.GetBondType() in self.ALLOWED_BONDS:
            bond_type[self.ALLOWED_BONDS.index(bond.GetBondType())] = 1

        aromatic = int(bond.GetIsAromatic())
        conjugated = int(bond.GetIsConjugated())

        return np.array(
            bond_type + [aromatic, conjugated],
            dtype=np.float32
        )

    # --------------------------------------------------

    def smiles_to_graph(self, smiles: str, label: Optional[float] = None):

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None

        mol = Chem.AddHs(mol)

        atom_features = [
            self._get_atom_features(atom)
            for atom in mol.GetAtoms()
        ]

        x = torch.tensor(atom_features, dtype=torch.float)

        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            feat = self._get_bond_features(bond)

            edge_index.append([i, j])
            edge_index.append([j, i])

            edge_attr.append(feat)
            edge_attr.append(feat)

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 6), dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

        data.smiles = smiles

        if label is not None:
            data.y = torch.tensor([label], dtype=torch.float)

        return data

    # --------------------------------------------------

    def smiles_list_to_graphs(
        self,
        smiles_list: List[str],
        labels: Optional[List[float]] = None
    ):

        graphs = []

        for i, s in enumerate(smiles_list):
            lbl = labels[i] if labels is not None else None
            g = self.smiles_to_graph(s, lbl)
            if g is not None:
                graphs.append(g)

        print(
            f"Successfully converted {len(graphs)}/{len(smiles_list)} SMILES strings"
        )
        return graphs


# --------------------------------------------------
# TEST MODULE
# --------------------------------------------------

if __name__ == "__main__":

    converter = SMILEStoGraphConverter()

    smiles_examples = [
        "CC(=O)Oc1ccccc1C(=O)O",   # Aspirin
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "C1=CC=C(C=C1)C2=CC=NN2"   # Example
    ]

    graphs = converter.smiles_list_to_graphs(smiles_examples)

    for i, g in enumerate(graphs):
        print(f"\nGraph {i}")
        print("Atoms:", g.x.shape[0])
        print("Atom feature dim:", g.x.shape[1])
        print("Bond feature dim:", g.edge_attr.shape[1])
