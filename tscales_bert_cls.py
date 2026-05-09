# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — T-scales + BERT/CLS  (drop-in replacement for ESM embeddings)
#
# Pipeline:
#   Protein sequence
#       → T-scales matrix  (seq_len × 5)
#       → Linear projection  (seq_len × d_model)   [acts as token embedding]
#       → BERT encoder  (Transformer)
#       → CLS token  (d_model,)                    ← replaces ESM vector
#       → concat with ChemBERTa SMILES + experimental features
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import torch
from torch import nn


# ── 1. T-scales matrix (same as before) ──────────────────────────────────────

tsc_df = pd.read_csv("Datasets/T_scales_table.csv")
tsc_df.columns = tsc_df.columns.str.strip()
tsc_df["symbol"] = tsc_df["symbol"].str.strip()

T_SCALE = {}
for _, row in tsc_df.iterrows():
    aa = row["symbol"]
    T_SCALE[aa] = [float(row[f"T_{i}"]) for i in range(1, 6)]

def sequence_to_tscales(seq):
    """Returns (seq_len, 5) matrix — one row per residue."""
    vecs = [T_SCALE.get(aa, [0.0, 0.0, 0.0, 0.0, 0.0]) for aa in seq]
    return np.array(vecs, dtype=np.float32)          # (L, 5)


# ── 2. BERT encoder that accepts T-scales matrices directly ──────────────────

class TScalesBERTEncoder(nn.Module):
    """
    Lightweight BERT-style encoder whose input is a T-scales matrix.

    Instead of a token-embedding lookup table, it uses a linear layer to
    project the 5 T-scale values per residue into d_model dimensions.
    A learnable [CLS] token is prepended; its final hidden state is returned
    as the sequence-level vector.

    Args
    ----
    d_model   : transformer hidden size  (default 256)
    nhead     : attention heads          (default 8)
    num_layers: transformer layers       (default 4)
    max_len   : max sequence length      (default 1024)
    dropout   : dropout rate            (default 0.1)
    """

    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 max_len: int = 1024,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model

        # Project 5 T-scale values → d_model  (replaces token embedding table)
        self.input_proj = nn.Linear(5, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional encodings  (CLS + max_len residues)
        self.pos_embedding = nn.Embedding(max_len + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,          # (batch, seq, d_model)
            norm_first=True,           # Pre-LN — more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (batch, seq_len, 5)   — padded T-scales matrix
        mask : (batch, seq_len)      — True where padding, False elsewhere
                                       (same convention as nn.Transformer)

        Returns
        -------
        cls_out : (batch, d_model)   — CLS token representation
        """
        B, L, _ = x.shape

        # Project residues into embedding space
        x = self.input_proj(x)                           # (B, L, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                 # (B, L+1, d_model)

        # Positional encodings
        positions = torch.arange(L + 1, device=x.device).unsqueeze(0)  # (1, L+1)
        x = x + self.pos_embedding(positions)

        # Extend key_padding_mask to account for prepended CLS (never masked)
        if mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
            mask = torch.cat([cls_mask, mask], dim=1)    # (B, L+1)

        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)

        return x[:, 0, :]                                # CLS token: (B, d_model)


# ── 3. Collate function — batch sequences of variable length ─────────────────

def collate_tscales(sequences: list[str],
                    max_len: int = 1024,
                    device: str = "cpu"):
    """
    Converts a list of protein sequences into a padded tensor + mask.

    Returns
    -------
    x    : (B, max_len, 5)
    mask : (B, max_len)   — True at padding positions
    """
    matrices = [sequence_to_tscales(seq[:max_len]) for seq in sequences]
    lengths  = [m.shape[0] for m in matrices]
    pad_len  = max(lengths)

    x    = np.zeros((len(sequences), pad_len, 5), dtype=np.float32)
    mask = np.ones( (len(sequences), pad_len),    dtype=bool)

    for i, (mat, ln) in enumerate(zip(matrices, lengths)):
        x[i, :ln, :] = mat
        mask[i, :ln]  = False                            # real tokens → not masked

    return (torch.tensor(x,    device=device),
            torch.tensor(mask, device=device))


# ── 4. Encode the full DataFrame (produces a vector per sequence) ─────────────

def encode_tscales_cls(sequences: pd.Series,
                       encoder: TScalesBERTEncoder,
                       batch_size: int = 16,
                       device: str = "cpu") -> np.ndarray:
    """
    Returns np.ndarray of shape (N, d_model) — one CLS vector per sequence.
    """
    encoder.eval()
    encoder.to(device)
    all_embs = []
    seqs = sequences.tolist()

    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i : i + batch_size]
            x, mask = collate_tscales(batch, device=device)
            cls_out = encoder(x, mask)                   # (B, d_model)
            all_embs.append(cls_out.cpu().numpy())

    return np.vstack(all_embs)                           # (N, d_model)


# ── 5. Drop-in replacement for the ESM concat cell ───────────────────────────
#
# Replace your existing concat() function with this one:
#
#   encoder = TScalesBERTEncoder(d_model=256)
#   # (train encoder jointly with your downstream model,
#   #  OR pre-train it, OR use it frozen with random weights as a baseline)
#
#   tscales_cls = encode_tscales_cls(df["Protein sequence"], encoder)
#   df["tscales_cls"] = list(tscales_cls)
#
#   def concat_model2(row):
#       return np.concatenate([
#           row["tscales_cls"],          # 256-d  (replaces 1280-d ESM)
#           row["smiles_vectors"],        # 768-d  (same ChemBERTa)
#           np.array([
#               row["Stokes shift"],
#               row["EC value"],
#               row["QY value"],
#               row["kDa"]
#           ])
#       ])
#
#   df["inputs_model2"] = df.apply(concat_model2, axis=1)


# ── 6. Quick sanity check ─────────────────────────────────────────────────────

if __name__ == "__main__":
    test_seqs = pd.Series([
        "MKTAYIAKQRQISFVKSHFSRQ",
        "ACDEFGHIKLMNPQRSTVWY",
    ])

    encoder = TScalesBERTEncoder(d_model=256, nhead=8, num_layers=4)
    out = encode_tscales_cls(test_seqs, encoder)

    print(f"Input sequences : {len(test_seqs)}")
    print(f"CLS output shape: {out.shape}")   # (2, 256)
    print("Sanity check passed ✓")
