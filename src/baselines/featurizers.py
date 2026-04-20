from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pandas as pd

from src.utils.dataset import extract_kmer_features


def _to_dataframe_with_raw_sequence(X: Any) -> pd.DataFrame:
    """Handle to dataframe with raw sequence."""
    if isinstance(X, pd.DataFrame):
        if "raw_sequence" not in X.columns:
            raise ValueError("Expected a 'raw_sequence' column for k-mer featurization.")
        return X

    arr = np.asarray(X)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array of raw DNA strings for k-mer featurization.")
    return pd.DataFrame(arr, columns=["raw_sequence"])


def _identity_transform(X: Any) -> np.ndarray:
    """Handle identity transform."""
    if isinstance(X, pd.DataFrame):
        out = X.to_numpy()
    elif isinstance(X, pd.Series):
        out = X.to_numpy().reshape(-1, 1)
    else:
        out = np.asarray(X)

    if out.dtype == object:
        raise ValueError(
            "Identity featurizer received non-numeric features. "
            "Use kmer_count for raw sequences."
        )
    return out


def transform_features(
    featurizer_cfg: Any, X_train_raw: Any, X_test_raw: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform train/test features according to the configured featurizer."""

    featurizer_name = str(featurizer_cfg.name).lower()

    if featurizer_name == "identity":
        return _identity_transform(X_train_raw), _identity_transform(X_test_raw)

    if featurizer_name == "kmer_count":
        k = int(featurizer_cfg.k)
        X_train_df = _to_dataframe_with_raw_sequence(X_train_raw)
        X_test_df = _to_dataframe_with_raw_sequence(X_test_raw)
        X_train, _ = extract_kmer_features(X_train_df, k=k)
        X_test, _ = extract_kmer_features(X_test_df, k=k)
        return X_train, X_test

    raise ValueError(f"Unsupported featurizer: {featurizer_cfg.name}")
