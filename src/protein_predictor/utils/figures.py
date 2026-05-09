"""
utils/figures.py
----------------
All figure-generation functions for the protein_predictor project.

Figures
-------
1. plot_rmse_grid        — val RMSE heatmap (layers × neurons) for both models
2. plot_rmse_bars        — bar chart: RMSE per architecture coloured by depth
3. plot_loss_curves      — train / val loss curves (both models side-by-side)
4. plot_per_target_curves— separate MSE curves per output target per model
5. plot_test_scatter     — predicted vs actual on the test set (original units)

All functions accept save=True (default) to write PNGs into config.FIGURES_DIR,
and return the matplotlib Figure object for interactive use.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from protein_predictor.config import (
    FIGURES_DIR, FIG_DPI, NEURON_OPTIONS, LAYER_OPTIONS,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
_DEPTH_COLORS = ["#a8d8ea", "#4C9BE8", "#1a5276"]   # 1 / 2 / 3 hidden layers
_ESM_COLOR    = "#2ecc71"
_TSC_COLOR    = "#e74c3c"


# ─────────────────────────────────────────────────────────────────────────────
# 1. RMSE Heatmap  (layers × neurons)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_grid(
    results_esm:     dict,
    results_tscales: dict,
    save:            bool = True,
) -> plt.Figure:
    """
    2-panel heatmap where rows = n_layers and columns = n_neurons.
    The best cell in each panel is highlighted with a gold border.

    Parameters
    ----------
    results_esm / results_tscales
        dict keyed by (n_layers, n_neurons) → {"val_rmse": float, ...}
    """
    n_rows = len(LAYER_OPTIONS)
    n_cols = len(NEURON_OPTIONS)

    def _to_matrix(results: dict) -> np.ndarray:
        M = np.full((n_rows, n_cols), np.nan)
        for (nl, nn), v in results.items():
            r, c = LAYER_OPTIONS.index(nl), NEURON_OPTIONS.index(nn)
            M[r, c] = v["val_rmse"]
        return M

    M1 = _to_matrix(results_esm)
    M2 = _to_matrix(results_tscales)
    vmin = np.nanmin([M1, M2])
    vmax = np.nanmax([M1, M2])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Validation RMSE — Architecture Grid Search",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, M, title in zip(
        axes,
        [M1, M2],
        ["Model 1 — ESM", "Model 2 — T-scales + BERT"],
    ):
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn_r",
                       vmin=vmin, vmax=vmax, interpolation="nearest")

        # Annotate each cell with its RMSE value
        for r in range(n_rows):
            for c in range(n_cols):
                val = M[r, c]
                if not np.isnan(val):
                    brightness = (val - vmin) / (vmax - vmin + 1e-9)
                    text_color = "white" if brightness > 0.5 else "black"
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=text_color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([str(n) for n in NEURON_OPTIONS])
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([f"{l} layer{'s' if l > 1 else ''}" for l in LAYER_OPTIONS])
        ax.set_xlabel("Neurons per layer", fontsize=11)
        ax.set_title(title, fontsize=12, pad=8)

        # Gold border on the best cell
        br, bc = np.unravel_index(np.nanargmin(M), M.shape)
        ax.add_patch(plt.Rectangle(
            (bc - 0.48, br - 0.48), 0.96, 0.96,
            fill=False, edgecolor="gold", linewidth=3, zorder=5,
        ))

    plt.colorbar(im, ax=axes, label="Val RMSE (normalised)", shrink=0.8, pad=0.02)
    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "rmse_grid.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. RMSE Bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_rmse_bars(
    results_esm:     dict,
    results_tscales: dict,
    save:            bool = True,
) -> plt.Figure:
    """
    Side-by-side bar charts with bars coloured by depth (1 / 2 / 3 layers)
    and a gold outline on the best architecture for each model.
    """
    sorted_keys = sorted(results_esm.keys())
    labels      = [f"{nl}L×{nn}N" for nl, nn in sorted_keys]
    rmse1       = [results_esm[k]["val_rmse"]     for k in sorted_keys]
    rmse2       = [results_tscales[k]["val_rmse"] for k in sorted_keys]
    bar_colors  = [_DEPTH_COLORS[nl - 1] for nl, _ in sorted_keys]
    x           = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.suptitle("Validation RMSE vs Network Architecture  (lower is better)",
                 fontsize=14, fontweight="bold")

    for ax, rmse, title in zip(
        axes,
        [rmse1, rmse2],
        ["Model 1 — ESM", "Model 2 — T-scales + BERT"],
    ):
        bars = ax.bar(x, rmse, color=bar_colors, edgecolor="white", linewidth=0.8)

        best = int(np.argmin(rmse))
        bars[best].set_edgecolor("gold")
        bars[best].set_linewidth(3)

        for bar, val in zip(bars, rmse):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + max(rmse) * 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=50,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
        ax.set_xlabel("Architecture  (L = layers, N = neurons/layer)", fontsize=11)
        ax.set_ylabel("Val RMSE (normalised)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)

        ax.legend(handles=[
            Patch(facecolor=_DEPTH_COLORS[0], label="1 hidden layer"),
            Patch(facecolor=_DEPTH_COLORS[1], label="2 hidden layers"),
            Patch(facecolor=_DEPTH_COLORS[2], label="3 hidden layers"),
            Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Best"),
        ], fontsize=9, loc="upper right")

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "rmse_bars.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Loss curves (train vs val, both models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curves(
    cb_esm:     "MetricsCallback",
    cb_tscales: "MetricsCallback",
    save:       bool = True,
) -> plt.Figure:
    """
    Side-by-side log-scale loss curves for the final (best-arch) models.

    Parameters
    ----------
    cb_esm / cb_tscales : MetricsCallback  (collected during final training)
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Training & Validation Loss Curves", fontsize=14, fontweight="bold")

    _data = [
        (axes[0], cb_esm,     "Model 1 — ESM",          _ESM_COLOR, "#27ae60"),
        (axes[1], cb_tscales, "Model 2 — T-scales + BERT", _TSC_COLOR, "#c0392b"),
    ]

    for ax, cb, title, c_tr, c_va in _data:
        if cb.train_loss:
            ax.plot(range(1, len(cb.train_loss) + 1), cb.train_loss,
                    color=c_tr, lw=2, label="Train loss")
        if cb.val_loss:
            ax.plot(range(1, len(cb.val_loss) + 1), cb.val_loss,
                    color=c_va, lw=2, ls="--", label="Val loss")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Combined MSE loss (log scale)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "loss_curves.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. Per-target MSE curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_target_curves(
    cb_esm:     "MetricsCallback",
    cb_tscales: "MetricsCallback",
    save:       bool = True,
) -> plt.Figure:
    """
    2×2 grid: rows = targets (emission WL, brightness), cols = models.
    Shows how each output's validation MSE evolves over training.
    Useful for diagnosing which target is responsible for high total MSE.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey="row")
    fig.suptitle("Per-Target Validation MSE Over Training",
                 fontsize=14, fontweight="bold")

    target_names = ["Emission wavelength", "Brightness"]
    model_names  = ["Model 1 — ESM", "Model 2 — T-scales + BERT"]
    cb_list      = [cb_esm, cb_tscales]
    color_pairs  = [(_ESM_COLOR, "#1abc9c"), (_TSC_COLOR, "#c0392b")]

    for col, (cb, mname, cpair) in enumerate(zip(cb_list, model_names, color_pairs)):
        for row, (series, tname) in enumerate(
            zip([cb.val_mse_wl, cb.val_mse_br], target_names)
        ):
            ax = axes[row][col]
            if series:
                ax.plot(range(1, len(series) + 1), series,
                        color=cpair[row], lw=2)
            ax.set_yscale("log")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Val MSE (log scale)", fontsize=10)
            ax.set_title(f"{mname}\n{tname}", fontsize=10)
            ax.grid(alpha=0.3)

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "per_target_curves.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 5. Predicted vs actual scatter (test set, original units)
# ─────────────────────────────────────────────────────────────────────────────

def plot_test_scatter(
    y_true:     np.ndarray,
    y_pred_esm: np.ndarray,
    y_pred_tsc: np.ndarray,
    save:       bool = True,
) -> plt.Figure:
    """
    2 rows × 2 cols scatter plots.
      rows → targets (emission wavelength nm, brightness)
      cols → models  (ESM, T-scales)

    Parameters
    ----------
    y_true / y_pred_esm / y_pred_tsc : np.ndarray  shape (n_test, 2)
        Values must be in ORIGINAL units (inverse-transform before calling).
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Predicted vs Actual — Test Set (original units)",
                 fontsize=14, fontweight="bold")

    tgt_names  = ["Emission wavelength (nm)", "Brightness"]
    units      = ["nm", ""]
    model_data = [(y_pred_esm, "Model 1 — ESM",          _ESM_COLOR),
                  (y_pred_tsc, "Model 2 — T-scales + BERT", _TSC_COLOR)]

    for col, (preds, mname, color) in enumerate(model_data):
        for row, (tname, unit) in enumerate(zip(tgt_names, units)):
            ax    = axes[row][col]
            truth = y_true[:, row]
            pred  = preds[:, row]
            rmse  = float(np.sqrt(((pred - truth) ** 2).mean()))

            mn, mx = truth.min(), truth.max()
            ax.scatter(truth, pred, alpha=0.7, s=45,
                       color=color, edgecolors="white", linewidths=0.5)
            ax.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Ideal (y = x)")

            u = f" {unit}" if unit else ""
            ax.set_xlabel(f"Actual {tname}", fontsize=10)
            ax.set_ylabel(f"Predicted {tname}", fontsize=10)
            ax.set_title(f"{mname}\nRMSE = {rmse:.2f}{u}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)

    plt.tight_layout()

    if save:
        path = FIGURES_DIR / "test_scatter.png"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved → {path}")
    return fig
