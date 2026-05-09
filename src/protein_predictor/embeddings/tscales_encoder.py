"""
embeddings/tscales_encoder.py
------------------------------
Thin wrapper around the user-supplied TScalesBERTEncoder from
`tscales_bert_cls.py` (must be importable from your Python path).

Usage
-----
    from protein_predictor.embeddings import TScalesEncoder

    enc = TScalesEncoder()
    df["tscales_cls"] = enc.encode_series(df["Protein sequence"])
"""

import numpy as np

# tscales_bert_cls.py must be on sys.path (place it next to main.py or install it)
from tscales_bert_cls import TScalesBERTEncoder, encode_tscales_cls


class TScalesEncoder:
    """
    Wraps TScalesBERTEncoder so it matches the same interface as ESMEncoder.

    Parameters
    ----------
    d_model   : int   — transformer model dimension (default 256)
    nhead     : int   — number of attention heads (default 8)
    num_layers: int   — number of transformer layers (default 4)
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        self.encoder = TScalesBERTEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_layers
        )

    def encode_series(self, sequences) -> list[np.ndarray]:
        """
        Encode a pandas Series (or list) of sequences.
        Returns a list of 1-D float32 arrays of shape (d_model,).
        """
        embeddings = encode_tscales_cls(sequences, self.encoder)
        return [e.astype(np.float32) for e in embeddings]
