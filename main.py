"""
main.py
=======
Entry point for the protein_predictor package.

Run from the project root after installing the package:
    pip install -e .
    python main.py

Workflow
--------
1.  Load CSV and compute embeddings (ESM-2, T-scales, ChemBERTa).
    ‼ This step is slow (~minutes per model). Embeddings are cached to disk
      so subsequent runs skip it automatically.

2.  Build normalised train / val / test datasets.

3.  Architecture grid search:  LAYER_OPTIONS × NEURON_OPTIONS architectures
    are each trained with early stopping, and val RMSE is recorded.

4.  Generate grid-search figures:
        figures/rmse_grid.png   — heatmap
        figures/rmse_bars.png   — bar chart

5.  (Optional) Optuna search for fine-grained hyperparameter optimisation.

6.  Train the final best models on train+val combined.

7.  Generate training figures:
        figures/loss_curves.png
        figures/per_target_curves.png

8.  Evaluate on the test set and generate:
        figures/test_scatter.png

All figures are saved to  protein_predictor/figures/
All config knobs are in   protein_predictor/src/protein_predictor/config.py
"""

# ── Fix import path BEFORE any package imports ───────────────────────────────
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
# tscales_bert_cls.py lives next to main.py — make sure it is also findable
sys.path.insert(0, str(Path(__file__).resolve().parent))
# ─────────────────────────────────────────────────────────────────────────────

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# ── Package imports ───────────────────────────────────────────────────────────
from protein_predictor.config import (
    CSV_PATH, DATA_DIR, FIGURES_DIR, CHECKPOINTS,
    MAX_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DROPOUT, PATIENCE,
    BATCH_SIZE, OPTUNA_TRIALS,
)
from protein_predictor.embeddings import ESMEncoder, TScalesEncoder, ChemBERTaEncoder, SMILES_DICT
from protein_predictor.data       import prepare_datasets
from protein_predictor.models     import FluorescenceNet
from protein_predictor.training   import (
    MetricsCallback, FPDataModule,
    run_grid_search, run_optuna,
)
from protein_predictor.utils import (
    plot_rmse_grid, plot_rmse_bars,
    plot_loss_curves, plot_per_target_curves,
    plot_test_scatter,
)

# ── Silence Lightning noise ───────────────────────────────────────────────────
os.environ["PL_DISABLE_PROGRESS_BAR"] = "1"
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

# Make sure output dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  —  Load CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_csv() -> pd.DataFrame:
    print(f"\n[1/8] Loading dataset from  {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"      {len(df)} rows, columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  —  Compute / load cached embeddings
# ─────────────────────────────────────────────────────────────────────────────

_CACHE = DATA_DIR / "embeddings_cache.pkl"

def compute_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three new columns to df:
        "esm"           — np.ndarray (1280,)  from ESM-2
        "tscales_cls"   — np.ndarray (256,)   from TScalesBERT
        "smiles_vectors"— np.ndarray (768,)   from ChemBERTa
    Results are cached in data/embeddings_cache.pkl to avoid recomputation.
    """
    if _CACHE.exists():
        print(f"\n[2/8] Loading cached embeddings from {_CACHE}")
        cached = pd.read_pickle(_CACHE)
        for col in ["esm", "tscales_cls", "smiles_vectors"]:
            df[col] = cached[col].values
        return df

    print("\n[2/8] Computing embeddings (this takes a while the first time)…")

    # ESM-2
    print("      → ESM-2 …")
    esm_enc  = ESMEncoder()
    df["esm"] = df["Protein sequence"].apply(esm_enc.encode)

    # T-scales
    print("      → T-scales BERT …")
    tsc_enc = TScalesEncoder()
    df["tscales_cls"] = tsc_enc.encode_series(df["Protein sequence"])

    # ChemBERTa (SMILES)
    print("      → ChemBERTa …")
    df["smiles"] = df["Chromophore/ligand"].str.strip().str.upper().map(SMILES_DICT)
    chem_enc     = ChemBERTaEncoder()
    df["smiles_vectors"] = df["smiles"].apply(chem_enc.encode)

    # Cache to disk
    df[["esm", "tscales_cls", "smiles_vectors"]].to_pickle(_CACHE)
    print(f"      Cached → {_CACHE}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  —  Prepare datasets
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(df: pd.DataFrame) -> dict:
    print("\n[3/8] Building train / val / test datasets …")
    splits = prepare_datasets(df)
    for model_key, s in splits.items():
        print(f"      {model_key:8s}  input_dim={s['input_dim']}  "
              f"train={len(s['train'])}  val={len(s['val'])}  test={len(s['test'])}")
    return splits


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4  —  Architecture grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_architecture_search(splits: dict) -> tuple[dict, dict]:
    print("\n[4/8] Architecture grid search …")

    print("  — Model 1 (ESM) —")
    results_esm = run_grid_search(
        train_ds  = splits["esm"]["train"],
        val_ds    = splits["esm"]["val"],
        test_ds   = splits["esm"]["test"],
        input_dim = splits["esm"]["input_dim"],
        label     = "ESM",
    )

    print("  — Model 2 (T-scales) —")
    results_tsc = run_grid_search(
        train_ds  = splits["tscales"]["train"],
        val_ds    = splits["tscales"]["val"],
        test_ds   = splits["tscales"]["test"],
        input_dim = splits["tscales"]["input_dim"],
        label     = "T-scales",
    )

    # Print bests
    best_esm = min(results_esm, key=lambda k: results_esm[k]["val_rmse"])
    best_tsc = min(results_tsc, key=lambda k: results_tsc[k]["val_rmse"])
    print(f"\n  Best ESM arch     : {best_esm[0]}L × {best_esm[1]}N  "
          f"(val RMSE={results_esm[best_esm]['val_rmse']:.4f})")
    print(f"  Best T-scales arch: {best_tsc[0]}L × {best_tsc[1]}N  "
          f"(val RMSE={results_tsc[best_tsc]['val_rmse']:.4f})")

    return results_esm, results_tsc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5  —  Grid-search figures
# ─────────────────────────────────────────────────────────────────────────────

def make_grid_figures(results_esm: dict, results_tsc: dict):
    print("\n[5/8] Saving grid-search figures …")
    plot_rmse_grid(results_esm, results_tsc, save=True)
    plot_rmse_bars(results_esm, results_tsc, save=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6  —  (Optional) Optuna fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def optuna_fine_tune(
    splits: dict,
    results_esm: dict,
    results_tsc: dict,
    run: bool = True,
) -> tuple[dict, dict]:
    """
    Uses best architecture from grid search and only optimizes:
        lr, batch_size
    """

    if not run:
        print("\n[6/8] Skipping Optuna (run=False)")
        return None, None

    print(f"\n[6/8] Optuna search ({OPTUNA_TRIALS} trials each) …")

    # ── Best architectures from grid search ────────────────────────────────
    best_esm_arch = min(results_esm, key=lambda k: results_esm[k]["val_rmse"])
    best_tsc_arch = min(results_tsc, key=lambda k: results_tsc[k]["val_rmse"])

    esm_layers, esm_hidden = best_esm_arch
    tsc_layers, tsc_hidden = best_tsc_arch

    print(
        f"  Best ESM architecture      : "
        f"{esm_layers}L × {esm_hidden}N"
    )

    print(
        f"  Best T-scales architecture : "
        f"{tsc_layers}L × {tsc_hidden}N"
    )

    print("  — Model 1 (ESM) —")

    best_esm = run_optuna(
        train_ds=splits["esm"]["train"],
        val_ds=splits["esm"]["val"],
        test_ds=splits["esm"]["test"],
        input_dim=splits["esm"]["input_dim"],
        hidden_sz=esm_hidden,
        n_layers=esm_layers,
        n_trials=OPTUNA_TRIALS,
    )

    print("  — Model 2 (T-scales) —")

    best_tsc = run_optuna(
        train_ds=splits["tscales"]["train"],
        val_ds=splits["tscales"]["val"],
        test_ds=splits["tscales"]["test"],
        input_dim=splits["tscales"]["input_dim"],
        hidden_sz=tsc_hidden,
        n_layers=tsc_layers,
        n_trials=OPTUNA_TRIALS,
    )

    return best_esm, best_tsc
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7  —  Train final models
# ─────────────────────────────────────────────────────────────────────────────

def train_final_models(
    splits:     dict,
    best_esm:   dict | None,
    best_tsc:   dict | None,
    results_esm: dict,
    results_tsc: dict,
) -> tuple:
    """
    Train final models using Optuna params (if available) or best grid-search arch.
    Returns (model_esm, model_tsc, cb_esm, cb_tsc).
    """
    print("\n[7/8] Training final models …")

    def _resolve_params(best_optuna, grid_results):
        """Pick Optuna params if available, otherwise fall back to best grid arch."""
        if best_optuna is not None:
            return (
                best_optuna["hidden_sz"],
                best_optuna["n_layers"],
                best_optuna["lr"],
                best_optuna["batch_size"],
            )
        best_key = min(grid_results, key=lambda k: grid_results[k]["val_rmse"])
        return best_key[1], best_key[0], LEARNING_RATE, BATCH_SIZE

    def _train_one(split_key, best_optuna, grid_results, label):
        hidden_sz, n_layers, lr, batch_size = _resolve_params(best_optuna, grid_results)
        print(f"  [{label}] hidden={hidden_sz}, layers={n_layers}, "
              f"lr={lr:.5f}, batch={batch_size}")

        s  = splits[split_key]
        cb = MetricsCallback()
        dm = FPDataModule(s["train"], s["val"], s["test"], batch_size=batch_size)

        model = FluorescenceNet(
            input_dim=s["input_dim"],
            hidden_sz=hidden_sz,
            n_layers=n_layers,
            dropout=DROPOUT,
            lr=lr,
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
            enable_progress_bar=True,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=dm)
        return model, cb, trainer, dm

    model_esm, cb_esm, trainer_esm, dm_esm = _train_one(
        "esm", best_esm, results_esm, "ESM"
    )
    model_tsc, cb_tsc, trainer_tsc, dm_tsc = _train_one(
        "tscales", best_tsc, results_tsc, "T-scales"
    )

    return model_esm, model_tsc, cb_esm, cb_tsc, trainer_esm, trainer_tsc, dm_esm, dm_tsc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8  —  Evaluate + final figures
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_and_plot(
    splits,
    model_esm, model_tsc,
    cb_esm,    cb_tsc,
    trainer_esm, trainer_tsc,
    dm_esm,    dm_tsc,
):
    print("\n[8/8] Evaluating on test set and saving final figures …")

    # Loss curves
    plot_loss_curves(cb_esm, cb_tsc, save=True)
    plot_per_target_curves(cb_esm, cb_tsc, save=True)

    # Collect test predictions in original units
    def _predict(model, trainer, dm, split_key):
        trainer.test(model, datamodule=dm, verbose=False)
        # Collect predictions manually
        model.eval()
        all_preds = []
        with torch.no_grad():
            for x, _ in dm.test_dataloader():
                all_preds.append(model(x).numpy())
        preds_normalised = np.vstack(all_preds)
        scaler = splits[split_key]["y_scaler"]
        return scaler.inverse_transform(preds_normalised)

    y_pred_esm = _predict(model_esm, trainer_esm, dm_esm, "esm")
    y_pred_tsc = _predict(model_tsc, trainer_tsc, dm_tsc, "tscales")
    y_true     = splits["esm"]["y_raw_test"]   # same split for both

    plot_test_scatter(y_true, y_pred_esm, y_pred_tsc, save=True)

    # Print test RMSE per target
    print("\n  FINAL TEST RMSE (original units)")
    print("  " + "─" * 48)
    for label, preds in [("ESM", y_pred_esm), ("T-scales", y_pred_tsc)]:
        rmse_wl = float(np.sqrt(((preds[:, 0] - y_true[:, 0]) ** 2).mean()))
        rmse_br = float(np.sqrt(((preds[:, 1] - y_true[:, 1]) ** 2).mean()))
        print(f"  {label:10s}  Emission wavelength : {rmse_wl:.2f} nm")
        print(f"  {''  :10s}  Brightness          : {rmse_br:.4f}")

    print(f"\n  All figures saved to  {FIGURES_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Load ───────────────────────────────────────────────────────────────
    df = load_csv()

    # ── 2. Embed ──────────────────────────────────────────────────────────────
    df = compute_embeddings(df)

    # ── 3. Split ──────────────────────────────────────────────────────────────
    splits = build_datasets(df)

    # ── 4. Grid search ────────────────────────────────────────────────────────
    results_esm, results_tsc = run_architecture_search(splits)

    # ── 5. Grid figures ───────────────────────────────────────────────────────
    make_grid_figures(results_esm, results_tsc)

   # ── 6. Optuna (only lr + batch_size are optimized) ──────────────────────────
    best_esm, best_tsc = optuna_fine_tune(
    splits,
    results_esm,
    results_tsc,
    run=True,
)

    # ── 7. Final training ─────────────────────────────────────────────────────
    model_esm, model_tsc, cb_esm, cb_tsc, \
    trainer_esm, trainer_tsc, dm_esm, dm_tsc = train_final_models(
        splits, best_esm, best_tsc, results_esm, results_tsc
    )

    # ── 8. Evaluate + figures ─────────────────────────────────────────────────
    evaluate_and_plot(
        splits,
        model_esm, model_tsc,
        cb_esm,    cb_tsc,
        trainer_esm, trainer_tsc,
        dm_esm,    dm_tsc,
    )