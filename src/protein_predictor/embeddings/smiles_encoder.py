"""
embeddings/smiles_encoder.py
----------------------------
Map chromophore codes to canonical SMILES and encode with ChemBERTa
(max-pool over token representations → 768-d vector).

Usage
-----
    from protein_predictor.embeddings import ChemBERTaEncoder, SMILES_DICT

    enc = ChemBERTaEncoder()
    df["smiles"] = df["Chromophore/ligand"].str.strip().str.upper().map(SMILES_DICT)
    df["smiles_vectors"] = df["smiles"].apply(enc.encode)
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# ── Canonical SMILES look-up table ───────────────────────────────────────────
SMILES_DICT: dict[str, str] = {
    "NRQ": r"CSCCC(=N)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "CRQ": r"NC(=O)CCC(=N)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "NRP": r"CC(C)CC(=N)C1=NC(=C/c2ccc(O)cc2)/C(=O)N1CC(O)=O",
    "CH6": r"CSCC[C@H](N)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "CRO": r"[C@@H](O)[C@H](N)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "5SQ": r"N[C@@H](Cc1c[nH]cn1)C2=NC(=C\c3ccc(O)cc3)/C(=O)N2CC(O)=O",
    "4M9": r"NC(=O)CCC(=N)C1=NC(=C\c2c[nH]c3ccccc23)/C(=O)N1CC(O)=O",
    "CR2": r"NCC1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "OFM": r"C[C@H]1O[C@@](O)(N=C1C2=N\C(=C/c3ccc(O)cc3)C(=O)N2CC(O)=O)[C@@H](N)Cc4ccccc4",
    "CR8": r"N[C@@H](Cc1[nH]cnc1)c2nc(C=C3C=CC(=O)C=C3)c([O-])n2CC(O)=O",
    "CFY": r"N[C@@H](Cc1ccccc1)[C@@]2(O)SCC(=N2)C3=NC(=C\c4ccc(O)cc4)/C(=O)N3CC(O)=O",
    "OIM": r"CC[C@H](C)[C@H](N)[C@@]1(O)O[C@H](C)C(=N1)C2=N\C(=C/c3ccc(O)cc3)C(=O)N2CC(O)=O",
    "CH7": r"OC(=O)CN1C(=O)C(=C/c2ccc(O)cc2)/N=C1C3=NCCCC3",
    "GYS": r"N[C@@H](CO)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "WCR": r"C[C@@]1(O)NC(=C\c2ccc(O)cc2)/C(=O)N1CC(O)=O",
    "DYG": r"N[C@@H](CC(O)=O)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "FAD": r"Cc1cc2N=C3C(=O)NC(=O)N=C3N(C[C@H](O)[C@H](O)[C@H](O)CO[P@](O)(=O)O[P@@](O)(=O)OC[C@H]4O[C@H]([C@H](O)[C@@H]4O)n5cnc6c(N)ncnc56)c2cc1C",
    "PIA": r"C[C@H](N)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "BLR": r"CC1=C(C=C)C(/NC1=O)=C/c2[nH]c(Cc3[nH]c(\C=C4/NC(=O)C(=C4C)C=C)c(C)c3CCC(O)=O)c(CCC(O)=O)c2C",
    "CRF": r"C[C@@H](O)[C@H](N)C1=N\C(=C/c2c[nH]c3ccccc23)C(=O)N1CC(O)=O",
    "NYG": r"N[C@@H](CC(N)=O)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "FMN": r"Cc1cc2N=C3C(=O)NC(=O)N=C3N(C[C@H](O)[C@H](O)[C@H](O)CO[P](O)(O)=O)c2cc1C",
    "B2H": r"C[C@@H](O)[C@H](N)c1nc(Cc2c[nH]c3ccccc23)c(O)n1CC(O)=O",
    "SWG": r"N[C@@H](CO)C1=N\C(=C/c2c[nH]c3ccccc23)C(=O)N1CC(O)=O",
    "CSH": r"N[C@@H](CO)[C@H]1N[C@@H](Cc2c[nH]cn2)C(=O)N1CC(O)=O",
    "BJF": r"CC(C)C[C@H](N)c1nc(CC(C)C)c(O)n1CC(O)=O",
    "GYC": r"N[C@@H](CS)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
    "CCY": r"N[C@@H](CS)[C@H]1N[C@@H](Cc2ccc(O)cc2)C(=O)N1CC(O)=O",
    "CR7": r"NCCCC[C@H](N)C1=N\C(=C/c2ccc(O)cc2)C(=O)N1CC(O)=O",
}


class ChemBERTaEncoder:
    """
    Encode a SMILES string → 768-d numpy vector using ChemBERTa.

    Max-pool is taken over all token representations (same as the notebook).
    Returns None for non-string inputs (missing SMILES).
    """

    MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModel.from_pretrained(self.MODEL_NAME)
        self.model.eval()

    @torch.no_grad()
    def encode(self, smiles: str) -> np.ndarray | None:
        if not isinstance(smiles, str):
            return None
        inputs  = self.tokenizer(smiles, return_tensors="pt",
                                 truncation=True, padding=True)
        outputs = self.model(**inputs)
        pooled  = outputs.last_hidden_state.max(dim=1).values.squeeze()
        return pooled.numpy().astype(np.float32)
