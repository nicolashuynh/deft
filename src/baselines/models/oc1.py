from __future__ import annotations

import numpy as np

from src.baselines.base import BaselineModel
from src.external.Oblique_Classifier_1 import ObliqueClassifier1


class OC1Baseline(BaselineModel):
    def __init__(
        self,
        max_depth: int,
        num_tries: int = 20,
        min_samples_split: int = 2,
        min_samples_leaf: int | None = None,
        min_samples_leaf_proportion: float = 0.01,
        random_state: int = 42,
    ) -> None:
        """Initialize the instance."""
        self.max_depth = max_depth
        self.num_tries = num_tries
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_proportion = min_samples_leaf_proportion
        self.random_state = random_state
        self.model: ObliqueClassifier1 | None = None

    def _resolve_min_samples_leaf(self, n_samples: int) -> int:
        """Resolve min samples leaf."""
        if self.min_samples_leaf is not None:
            return self.min_samples_leaf
        return max(1, int(self.min_samples_leaf_proportion * n_samples))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit values."""
        min_samples_leaf = self._resolve_min_samples_leaf(len(y))
        self.model = ObliqueClassifier1(
            max_depth=self.max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=self.min_samples_split,
            num_tries=self.num_tries,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict values."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict proba."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def get_model_name(self) -> str:
        """Get model name."""
        return "oc1"
