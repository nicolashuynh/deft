from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.dataset_plugins import instantiate_dataset_plugin


def _index_rows(X: Any, indices: np.ndarray) -> Any:
    """Handle index rows."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[indices]
    arr = np.asarray(X)
    return arr[indices]


def _to_1d_numpy(y: Any) -> np.ndarray:
    """Handle to 1d numpy."""
    if isinstance(y, (pd.DataFrame, pd.Series)):
        return y.to_numpy().ravel()
    return np.asarray(y).ravel()


def _subsample_train(
    X_train: Any, y_train: Any, subsample_proportion: float, seed: int
) -> tuple[Any, Any]:
    """Subsample train."""
    if subsample_proportion >= 1.0:
        return X_train, y_train

    n_train = len(X_train)
    n_keep = max(1, int(n_train * subsample_proportion))
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_train, n_keep, replace=False)

    X_sub = _index_rows(X_train, indices)
    y_sub = _index_rows(y_train, indices)
    return X_sub, y_sub


def load_dataset_for_baseline(cfg: Any, seed: int) -> Dict[str, Any]:
    """Load train/test data while preserving dataset-specific legacy behavior."""

    featurizer_name = str(cfg.featurizer.name)
    plugin = instantiate_dataset_plugin(cfg.dataset)
    X_train, X_test, y_train, y_test = plugin.load_for_baseline(
        featurizer_name=featurizer_name,
        seed=int(seed),
    )

    X_train, y_train = _subsample_train(
        X_train,
        y_train,
        float(getattr(cfg.dataset, "subsample_proportion", 1.0)),
        seed,
    )

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": _to_1d_numpy(y_train),
        "y_test": _to_1d_numpy(y_test),
    }
