"""
training/optuna_search.py
-------------------------
Optuna fine-tuning AFTER architecture search.

Architectural hyperparameters:
    hidden_sz
    n_layers

are FIXED from grid_search.py.

Optuna ONLY tunes:
    lr
    batch_size
"""

import logging
import os

import optuna
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping

from protein_predictor.config import (
    OPTUNA_TRIALS,
    OPTUNA_EPOCHS,
    DROPOUT,
    WEIGHT_DECAY,
)

from protein_predictor.models import FluorescenceNet

from protein_predictor.training.datamodule import FPDataModule


# ─────────────────────────────────────────────────────────────────────────────
# SILENCE LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def _silence_loggers():

    os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"

    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    logging.getLogger(
        "lightning.pytorch"
    ).setLevel(logging.ERROR)

    logging.getLogger(
        "pytorch_lightning"
    ).setLevel(logging.ERROR)


# ─────────────────────────────────────────────────────────────────────────────
# OPTUNA
# ─────────────────────────────────────────────────────────────────────────────
def run_optuna(
    train_ds,
    val_ds,
    test_ds,
    input_dim: int,
    n_layers: int,
    hidden_sz: int,
    n_trials: int = OPTUNA_TRIALS,
) -> dict:
    """
    Fine-tune ONLY optimizer/training hyperparameters.

    Architecture is fixed from grid search.

    Returns
    -------
    dict:
        {
            "hidden_sz",
            "n_layers",
            "lr",
            "batch_size",
        }
    """

    _silence_loggers()

    # ────────────────────────────────────────────────────────────────────────
    # OBJECTIVE
    # ────────────────────────────────────────────────────────────────────────
    def objective(trial):

        # ONLY optimize these
        lr = trial.suggest_float(
            "lr",
            1e-4,
            1e-2,
            log=True,
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            [32, 64, 128],
        )

        dm = FPDataModule(
            train_ds,
            val_ds,
            test_ds,
            batch_size=batch_size,
        )

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
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=8,
                    mode="min",
                )
            ],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

        trainer.fit(
            model,
            datamodule=dm,
        )

        return trainer.callback_metrics[
            "val_loss"
        ].item()

    # ────────────────────────────────────────────────────────────────────────
    # STUDY
    # ────────────────────────────────────────────────────────────────────────
    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=False,
    )

    # ────────────────────────────────────────────────────────────────────────
    # MERGE FIXED + OPTIMIZED PARAMS
    # ────────────────────────────────────────────────────────────────────────
    best_params = study.best_params

    best_params["hidden_sz"] = hidden_sz
    best_params["n_layers"] = n_layers

    print(f"  Fixed architecture : {n_layers}L × {hidden_sz}N")
    print(f"  Best Optuna params : {best_params}")
    print(f"  Best val loss      : {study.best_value:.4f}")

    return best_params