from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaselineModel(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the baseline model."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (n_samples, n_classes)."""

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a stable model name for logging and outputs."""
