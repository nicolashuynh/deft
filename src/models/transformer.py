import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import BinaryAveragePrecision


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize the instance."""
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        odd_dim_count = pe[:, 0, 1::2].shape[1]
        pe[:, 0, 1::2] = torch.cos(position * div_term[:odd_dim_count])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """Handle forward."""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class LitTransformer(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        n_nucleotides: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        dropout: float,
        learning_rate: float = 1e-3,
    ):
        """Initialize the instance."""
        super().__init__()
        self.save_hyperparameters()

        self.input_embedding = nn.Linear(n_nucleotides, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, n_head, d_model * 4, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.classifier = nn.Linear(d_model, 1)

        self.accuracy = Accuracy(task="binary")
        self.val_auprc = BinaryAveragePrecision()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Handle forward."""
        x = self.input_embedding(x)

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)

        x = self.transformer_encoder(x)

        cls_token_output = x[:, 0, :]

        logits = self.classifier(cls_token_output)
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
        self.log("train_loss", loss)
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

        self.accuracy(logits_squeezed, y.int())
        self.log("test_acc", self.accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Predict step."""
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        """Handle configure optimizers."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
