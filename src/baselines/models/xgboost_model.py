from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from src.baselines.base import BaselineModel


class XGBoostBaseline(BaselineModel):
    def __init__(
        self,
        max_depth: int,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        min_child_weight: int | None = None,
        min_child_weight_proportion: float = 0.01,
        eval_metric: str = "logloss",
        random_state: int = 42,
    ) -> None:
        """Initialize the instance."""
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.min_child_weight_proportion = min_child_weight_proportion
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.model: XGBClassifier | None = None

    def _resolve_min_child_weight(self, n_samples: int) -> int:
        """Resolve min child weight."""
        if self.min_child_weight is not None:
            return self.min_child_weight
        return max(1, int(self.min_child_weight_proportion * n_samples))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit values."""
        min_child_weight = self._resolve_min_child_weight(len(y))
        self.model = XGBClassifier(
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=min_child_weight,
            eval_metric=self.eval_metric,
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
        return "xgboost"
