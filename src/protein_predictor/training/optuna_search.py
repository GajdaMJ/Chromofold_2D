"""
training/optuna_search.py
-------------------------
Optuna-based hyperparameter search over:
    hidden_sz  (64 … 512, step 64)
    n_layers   (1 / 2 / 3)
    lr         (log-uniform, 1e-4 … 1e-2)
    batch_size (32 / 64 / 128)

Returns the best_params dict which can be passed directly to
FluorescenceNet / FPDataModule in main.py.
"""

import logging
import os

import optuna
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from protein_predictor.config import (
    OPTUNA_TRIALS, OPTUNA_EPOCHS, DROPOUT, WEIGHT_DECAY,
)
from protein_predictor.models           import FluorescenceNet
from protein_predictor.training.datamodule import FPDataModule


def _silence_loggers():
    os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def run_optuna(
    train_ds,
    val_ds,
    test_ds,
    input_dim: int,
    n_trials:  int = OPTUNA_TRIALS,
) -> dict:
    """
    Run an Optuna study and return the best hyperparameter dict.

    Keys in the returned dict
    -------------------------
        hidden_sz, n_layers, lr, batch_size
    """
    _silence_loggers()

    def objective(trial):
        hidden_sz  = trial.suggest_int("hidden_sz",  64, 512, step=64)
        n_layers   = trial.suggest_int("n_layers",   1, 3)
        lr         = trial.suggest_float("lr",       1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        dm = FPDataModule(train_ds, val_ds, test_ds, batch_size=batch_size)
        model = FluorescenceNet(
            input_dim=input_dim,
            hidden_sz=hidden_sz,
            n_layers=n_layers,
            dropout=DROPOUT,
            lr=lr,
            weight_decay=WEIGHT_DECAY,
        )
        trainer = pl.Trainer(
            max_epochs=OPTUNA_EPOCHS,
            callbacks=[EarlyStopping(monitor="val_loss", patience=8, mode="min")],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=dm)
        return trainer.callback_metrics["val_loss"].item()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print(f"  Best params : {study.best_params}")
    print(f"  Best val loss : {study.best_value:.4f}")
    return study.best_params
