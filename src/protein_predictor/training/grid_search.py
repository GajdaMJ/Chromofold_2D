"""
training/grid_search.py
-----------------------
Exhaustive search over (n_layers × n_neurons) from config.ARCH_GRID.

Produces a results dict keyed by (n_layers, n_neurons) with entries:
    {"val_rmse": float, "history": MetricsCallback}

This dict is consumed by utils/figures.py to draw heatmaps and bar charts.
"""

import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from protein_predictor.config import (
    ARCH_GRID, BATCH_SIZE, MAX_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, DROPOUT, PATIENCE,
)
from protein_predictor.models           import FluorescenceNet
from protein_predictor.training.callbacks  import MetricsCallback
from protein_predictor.training.datamodule import FPDataModule


def _silence_loggers():
    os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run_grid_search(
    train_ds,
    val_ds,
    test_ds,
    input_dim: int,
    label:     str = "model",
) -> dict:
    """
    Train one FluorescenceNet per (n_layers, n_neurons) in ARCH_GRID.

    Parameters
    ----------
    train_ds / val_ds / test_ds : FPDataset
    input_dim : int   — must match the dataset's feature dimensionality
    label     : str   — printed prefix in progress lines ("ESM" / "T-scales")

    Returns
    -------
    dict  :  (n_layers, n_neurons)  →  {"val_rmse": float, "history": MetricsCallback}
    """
    _silence_loggers()
    results = {}

    for n_layers, n_neurons in ARCH_GRID:
        key = (n_layers, n_neurons)
        tag = f"{n_layers}L × {n_neurons:3d}N"

        cb = MetricsCallback()
        dm = FPDataModule(train_ds, val_ds, test_ds, batch_size=BATCH_SIZE)

        model = FluorescenceNet(
            input_dim=input_dim,
            hidden_sz=n_neurons,
            n_layers=n_layers,
            dropout=DROPOUT,
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[
                cb,
                EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"),
            ],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=dm)

        val_rmse = trainer.callback_metrics.get("val_rmse", float("nan"))
        if hasattr(val_rmse, "item"):
            val_rmse = val_rmse.item()

        results[key] = {"val_rmse": val_rmse, "history": cb}
        print(f"  [{label:8s}]  {tag}  →  val RMSE = {val_rmse:.4f}")

    return results
