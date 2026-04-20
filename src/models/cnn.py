import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAveragePrecision


class LitConvolutionalNetwork(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        in_channels: int = 4,
        filters1: int = 32,
        kernel_size1: int = 4,
        pool_size1: int = 4,
        filters2: int = 64,
        kernel_size2: int = 4,
        pool_size2: int = 2,
        dense_units: int = 64,
        dropout: float = 0.5,
        learning_rate: float = 1e-3,
    ):
        """Initialize the instance."""
        super().__init__()
        self.save_hyperparameters()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, self.hparams.filters1, self.hparams.kernel_size1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.hparams.pool_size1),
            nn.Conv1d(
                self.hparams.filters1, self.hparams.filters2, self.hparams.kernel_size2
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=self.hparams.pool_size2),
        )

        with torch.no_grad():
            dummy_input = torch.randn(
                1, self.hparams.in_channels, self.hparams.sequence_length
            )
            output_dim = self.features(dummy_input).nelement()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=output_dim, out_features=self.hparams.dense_units),
            nn.ReLU(),
            nn.Dropout(p=self.hparams.dropout),
            nn.Linear(in_features=self.hparams.dense_units, out_features=1),
        )

        self.accuracy = Accuracy(task="binary")
        self.val_auprc = BinaryAveragePrecision()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Permute from (N, L, C) to (N, C, L) for Conv1D
        """Handle forward."""
        x = x.permute(0, 2, 1)
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    def _common_step(self, batch, batch_idx):
        """Handle common step."""
        x, y = batch
        logits = self(x)
        logits = torch.nan_to_num(logits)
        squeezed_logits = logits.squeeze(dim=-1)
        y_float = y.float().view(-1)

        if squeezed_logits.shape != y_float.shape:
            raise ValueError(
                "Shape mismatch between logits and targets: "
                f"{tuple(squeezed_logits.shape)} vs {tuple(y_float.shape)}"
            )

        loss = F.binary_cross_entropy_with_logits(squeezed_logits, y_float)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected in {self.__class__.__name__}: {loss.item()}"
            )

        return loss, logits, y

    def training_step(self, batch, batch_idx):
        """Handle training step."""
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Handle validation step."""
        loss, logits, y = self._common_step(batch, batch_idx)

        self.log("val_loss", loss, prog_bar=True)

        if logits.dim() > 1:
            logits_squeezed = logits.squeeze(-1)
        else:
            logits_squeezed = logits

        targets = y.int()
        probs = torch.sigmoid(logits_squeezed)
        self.val_auprc(probs, targets)
        self.log(
            "val_auprc", self.val_auprc, on_step=False, on_epoch=True, prog_bar=True
        )

        self.accuracy(logits_squeezed, targets)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Handle test step."""
        loss, logits, y = self._common_step(batch, batch_idx)
        self.log("test_loss", loss)

        if logits.dim() > 1:
            logits_squeezed = logits.squeeze(-1)
        else:
            logits_squeezed = logits

        self.accuracy(logits_squeezed, y)
        self.log("test_acc", self.accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Predict step."""
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        """Handle configure optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.01
        )
        return optimizer
