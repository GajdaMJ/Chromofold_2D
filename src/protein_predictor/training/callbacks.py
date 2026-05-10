"""
training/callbacks.py
---------------------
MetricsCallback collects per-epoch losses into plain Python lists
so they can be plotted after training without WandB or TensorBoard.
"""

import pytorch_lightning as pl


class MetricsCallback(pl.Callback):
    """Accumulate train / val metrics over epochs."""

    def __init__(self):
        self.train_loss = []
        self.val_loss   = []
        self.val_mse_wl = []   # emission wavelength
        self.val_mse_br = []   # brightness
        self.val_rmse   = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        if "train_loss" in m:
            self.train_loss.append(m["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        mapping = {
            "val_loss":   "val_loss",
            "val_mse_wl": "val_mse_wl",
            "val_mse_br": "val_mse_br",
            "val_rmse":   "val_rmse",
        }
        for attr, key in mapping.items():
            if key in m:
                getattr(self, attr).append(m[key].item())