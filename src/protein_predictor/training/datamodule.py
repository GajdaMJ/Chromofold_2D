"""
training/datamodule.py
----------------------
LightningDataModule that wraps train / val / test FPDatasets.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from protein_predictor.config import BATCH_SIZE


class FPDataModule(pl.LightningDataModule):
    """
    Parameters
    ----------
    train_ds / val_ds / test_ds : FPDataset
    batch_size : int  (overrides config default if supplied)
    """

    def __init__(self, train_ds, val_ds, test_ds, batch_size: int = BATCH_SIZE):
        super().__init__()
        self.train_ds   = train_ds
        self.val_ds     = val_ds
        self.test_ds    = test_ds
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=0)
