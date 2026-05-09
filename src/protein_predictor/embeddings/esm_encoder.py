"""
embeddings/esm_encoder.py
-------------------------
Generate mean-pooled ESM-2 (esm2_t33_650M_UR50D) embeddings
for raw protein sequences.

The `esm` library is imported lazily inside __init__ so that the rest of
the package can be imported without ESM being installed (useful if you only
want to run the T-scales model).

Usage
-----
    from protein_predictor.embeddings import ESMEncoder

    enc = ESMEncoder()
    df["esm"] = df["Protein sequence"].apply(enc.encode)
"""

import numpy as np
import torch


class ESMEncoder:
    """
    Loads ESM-2 once and encodes sequences on demand.

    Parameters
    ----------
    model_name : str  — ESM pretrained model name (default: esm2_t33_650M_UR50D)
    repr_layer : int  — transformer layer to extract (33 = final for esm2_t33)
    """

    def __init__(
        self,
        model_name: str = "esm2_t33_650M_UR50D",
        repr_layer: int = 33,
    ):
        import esm as _esm

        loader = getattr(_esm.pretrained, model_name)
        self.model, self.alphabet = loader()

        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer = repr_layer

    @torch.no_grad()
    def encode(self, sequence: str) -> np.ndarray:
        """
        Encode one amino-acid sequence -> float32 array of shape (1280,).
        BOS and EOS tokens are stripped before mean-pooling over residues.
        """
        data = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)
        results   = self.model(tokens, repr_layers=[self.repr_layer])
        reps      = results["representations"][self.repr_layer]
        embedding = reps[0, 1:-1].mean(0)   # strip BOS/EOS, mean-pool
        return embedding.numpy().astype(np.float32)