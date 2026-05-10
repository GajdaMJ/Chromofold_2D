"""
models/network.py
-----------------
Flexible MLP regressor for fluorescent-protein property prediction.

Key differences from the notebook version that help reduce high MSE
--------------------------------------------------------------------
1. Variable depth (n_layers) AND width (hidden_sz) in one constructor.
2. BatchNorm1d after every linear layer → stabilises activations,
   allows higher learning rates, speeds convergence.
3. Dropout between layers → regularises heavily-overparameterised runs.
4. Kaiming weight initialisation → better gradient flow from epoch 1.
5. Per-target MSE logged separately → shows which output drives the error.
6. ReduceLROnPlateau scheduler → automatically reduces lr when stuck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class FluorescenceNet(pl.LightningModule):
    """
    Fully-connected MLP regressor with configurable depth and width.

    Parameters
    ----------
    input_dim    : number of input features (2050 for ESM, 1026 for T-scales)
    hidden_sz    : neurons per hidden layer
    n_layers     : number of hidden layers (1, 2, or 3 recommended)
    dropout      : dropout probability applied after each hidden layer
    lr           : initial Adam learning rate
    weight_decay : L2 regularisation coefficient
    """

    def __init__(
        self,
        input_dim:    int,
        hidden_sz:    int   = 128,
        n_layers:     int   = 2,
        dropout:      float = 0.3,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── Build hidden layers dynamically ──────────────────────────────────
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_sz),
                nn.BatchNorm1d(hidden_sz),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_sz

        layers.append(nn.Linear(in_dim, 2))   # output: [emission_wl, brightness]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ── Shared loss computation ───────────────────────────────────────────────
    def _step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y   = batch
        preds  = self(x)

        mse_wl = F.mse_loss(preds[:, 0], y[:, 0])   # emission wavelength
        mse_br = F.mse_loss(preds[:, 1], y[:, 1])   # brightness
        loss   = mse_wl + mse_br
        rmse   = torch.sqrt(loss)

        on_step = (stage == "train")
        self.log(f"{stage}_loss",   loss,   prog_bar=True,  on_epoch=True, on_step=on_step)
        self.log(f"{stage}_mse_wl", mse_wl, on_epoch=True,  on_step=False)
        self.log(f"{stage}_mse_br", mse_br, on_epoch=True,  on_step=False)
        self.log(f"{stage}_rmse",   rmse,   prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self,   batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")
    def test_step(self,       batch, _): return self._step(batch, "test")

    # ── Optimiser + LR scheduler ──────────────────────────────────────────────
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=8
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }