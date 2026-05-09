"""
config.py
---------
Central configuration for the protein_predictor package.
Edit values here instead of hunting through individual modules.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# Root of the installed package (protein_predictor/src/protein_predictor/)
_PKG_ROOT = Path(__file__).resolve().parent

# Project root (protein_predictor/) — one level above src/
PROJECT_ROOT = _PKG_ROOT.parent.parent

DATA_DIR     = PROJECT_ROOT / "data"          # put your CSV here
FIGURES_DIR  = PROJECT_ROOT / "figures"       # all plots saved here
CHECKPOINTS  = PROJECT_ROOT / "checkpoints"   # Lightning checkpoints

CSV_PATH     = DATA_DIR / "Fluorescent-Protein-Database.csv"

# ── Target columns in the CSV ─────────────────────────────────────────────────
TARGET_COLS = ["Emission wavelength", "Brightness"]

# ── Embedding dimensions ──────────────────────────────────────────────────────
ESM_DIM       = 1280   # esm2_t33_650M_UR50D  layer-33 mean-pool
TSCALES_DIM   = 256    # TScalesBERTEncoder   d_model
CHEMBERTA_DIM = 768    # ChemBERTa            max-pool over tokens
EXTRA_FEATS   = 2      # Stokes shift + kDa

INPUT_DIM_ESM     = ESM_DIM     + CHEMBERTA_DIM + EXTRA_FEATS   # 2050
INPUT_DIM_TSCALES = TSCALES_DIM + CHEMBERTA_DIM + EXTRA_FEATS   # 1026

# ── Architecture grid ─────────────────────────────────────────────────────────
NEURON_OPTIONS = [16, 32, 64, 128, 256]   # neurons per hidden layer
LAYER_OPTIONS  = [1, 2, 3]                # number of hidden layers

# All (n_layers, n_neurons) pairs iterated during grid search
ARCH_GRID = [
    (n_layers, n_neurons)
    for n_layers in LAYER_OPTIONS
    for n_neurons in NEURON_OPTIONS
]

# ── Data split ────────────────────────────────────────────────────────────────
TEST_FRAC   = 0.20    # held-out test fraction
VAL_FRAC    = 0.15    # fraction of *remaining* data used for validation
RANDOM_SEED = 42

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64
MAX_EPOCHS    = 150
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3    # dropout probability between hidden layers
PATIENCE      = 15     # early-stopping patience in epochs

# ── Optuna ────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 50
OPTUNA_EPOCHS = 30     # shorter budget inside each Optuna trial

# ── Figures ───────────────────────────────────────────────────────────────────
FIG_DPI = 150
