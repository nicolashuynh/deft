from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import instantiate

from src.data.non_tata_promoters import load_promoters
from src.data.polymerase_loader import load_polymerase


@contextmanager
def _temporary_cwd(path: Path):
    """Handle temporary cwd."""
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def _subset_rows(data: Any, indices: np.ndarray) -> Any:
    """Handle subset rows."""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[indices].reset_index(drop=True)
    return np.asarray(data)[indices]


def _subsample_training_data(
    X_train: Any, y_train: Any, subsample_proportion: float, random_state: int
) -> tuple[Any, Any]:
    """Subsample training data."""
    if subsample_proportion >= 1.0:
        return X_train, y_train

    n_keep = max(1, int(len(X_train) * subsample_proportion))
    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X_train), n_keep, replace=False)

    if len(indices) > 10:
        logging.info(
            "Selected %d random indices for training: %s...",
            len(indices),
            indices[:10],
        )
    else:
        logging.info("Selected random indices for training: %s", indices)

    logging.info(
        "Training set subsampled from %d to %d samples (%.1f%%)",
        len(X_train),
        len(indices),
        subsample_proportion * 100,
    )

    return _subset_rows(X_train, indices), _subset_rows(y_train, indices)


def _get_condition_details(condition_name: str) -> str:
    """Get condition details."""
    parts = condition_name.split("_")
    cell_line_map = {"k562": "K562 blood cells", "hepg2": "HepG2 liver cells"}
    promoter_map = {"minp": "a minimal promoter", "sv40p": "a strong SV40 promoter"}
    cell_line = cell_line_map.get(parts[0], "an unknown cell line")
    promoter = promoter_map.get(parts[1], "an unknown promoter")
    return f"This specific task is for {cell_line} using {promoter}."


class BaseDatasetPlugin(ABC):
    """
    Pluggable dataset interface used by DEFT and baseline runners.

    To support a new dataset, add one class implementing this interface and
    reference it from `conf/dataset/<name>.yaml` under `plugin._target_`.
    """

    # Keep historical behavior: polymerase split order differs from others.
    dl_subsample_before_val_split = True

    @abstractmethod
    def load_for_deft(self) -> dict[str, Any]:
        """Return the DEFT training bundle."""

    @abstractmethod
    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        """Return (X_train, X_test, y_train, y_test) for classic baselines."""

    @abstractmethod
    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        """Return (X_train, X_test, y_train, y_test) for DL baselines."""


class PolymeraseDatasetPlugin(BaseDatasetPlugin):
    dl_subsample_before_val_split = False

    def __init__(
        self,
        use_raw_sequence: bool = True,
        type_dataset: str = "Plus",
        name_experiment: str = "",
        test_size: float = 0.2,
        random_state: int = 42,
        no_prior_knowledge: bool = False,
        size_left_window: int = 50,
    ):
        """Initialize the instance."""
        self.use_raw_sequence = bool(use_raw_sequence)
        self.type_dataset = str(type_dataset)
        self.name_experiment = str(name_experiment)
        self.test_size = float(test_size)
        self.random_state = int(random_state)
        self.no_prior_knowledge = bool(no_prior_knowledge)
        self.size_left_window = int(size_left_window)

    def _load(
        self,
        *,
        use_raw_sequence: bool,
        no_prior_knowledge: bool,
        one_hot_encoding: bool,
        random_state: int,
        use_external_test_set: bool,
    ) -> tuple[Any, Any, Any, Any]:
        """Load values."""
        kwargs = dict(
            use_raw_sequence=use_raw_sequence,
            type_dataset=self.type_dataset,
            no_prior_knowledge=no_prior_knowledge,
            test_size=self.test_size,
            random_state=random_state,
            size_left_window=self.size_left_window,
            one_hot_encoding=one_hot_encoding,
            use_external_test_set=use_external_test_set,
        )

        if use_external_test_set:
            # Keep backward-compatible execution context for polymerase baseline/DL loads.
            repo_root = Path(__file__).resolve().parents[2]
            with _temporary_cwd(repo_root / "experiments"):
                return load_polymerase(**kwargs)
        return load_polymerase(**kwargs)

    def load_for_deft(self) -> dict[str, Any]:
        """Load for deft."""
        no_prior = "no_prior_knowledge" in self.name_experiment
        X_train, _, y_train, _ = self._load(
            use_raw_sequence=self.use_raw_sequence,
            no_prior_knowledge=no_prior,
            one_hot_encoding=False,
            random_state=self.random_state,
            use_external_test_set=False,
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "tree_filename": "tree_polymerase.dill",
            "prompt_context": {},
        }

    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        """Load for baseline."""
        use_raw_sequence = str(featurizer_name) == "kmer_count"
        one_hot_encoding = not use_raw_sequence
        return self._load(
            use_raw_sequence=use_raw_sequence,
            no_prior_knowledge=self.no_prior_knowledge,
            one_hot_encoding=one_hot_encoding,
            random_state=int(seed),
            use_external_test_set=True,
        )

    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        """Load for dl."""
        return self._load(
            use_raw_sequence=self.use_raw_sequence,
            no_prior_knowledge=self.no_prior_knowledge,
            one_hot_encoding=True,
            random_state=int(seed),
            use_external_test_set=True,
        )


class PromotersDatasetPlugin(BaseDatasetPlugin):
    def __init__(
        self,
        random_state: int = 42,
        subsample_proportion: float = 1.0,
    ):
        """Initialize the instance."""
        self.random_state = int(random_state)
        self.subsample_proportion = float(subsample_proportion)

    def load_for_deft(self) -> dict[str, Any]:
        """Load for deft."""
        X_train, _, y_train, _ = load_promoters(use_raw_sequence=True)
        X_train, y_train = _subsample_training_data(
            X_train,
            y_train,
            self.subsample_proportion,
            self.random_state,
        )
        y_train = pd.DataFrame(np.asarray(y_train), columns=["is_promoter"])
        return {
            "X_train": X_train,
            "y_train": y_train,
            "tree_filename": "tree_human_non_tata_promoter.dill",
            "prompt_context": {},
        }

    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        """Load for baseline."""
        use_raw_sequence = str(featurizer_name) == "kmer_count"
        return load_promoters(use_raw_sequence=use_raw_sequence)

    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        """Load for dl."""
        return load_promoters(use_raw_sequence=False)


class MPRADatasetPlugin(BaseDatasetPlugin):
    def __init__(
        self,
        condition_name: str = "k562_minp_avg",
        one_hot_encode: bool = False,
        subsample_proportion: float = 1.0,
        random_state: int = 42,
    ):
        """Initialize the instance."""
        self.condition_name = str(condition_name)
        self.one_hot_encode = bool(one_hot_encode)
        self.subsample_proportion = float(subsample_proportion)
        self.random_state = int(random_state)

    def _load_mpra(
        self, *, one_hot_encode: bool, use_2D: bool
    ) -> tuple[Any, Any, Any, Any]:
        # Lazy import to avoid h5py dependency unless MPRA is used.
        """Load mpra."""
        from src.data.mpra import load_mpra

        return load_mpra(
            condition_name=self.condition_name,
            one_hot_encode=one_hot_encode,
            use_2D=use_2D,
        )

    def load_for_deft(self) -> dict[str, Any]:
        """Load for deft."""
        X_train, y_train, _, _ = self._load_mpra(
            one_hot_encode=self.one_hot_encode, use_2D=False
        )
        X_train, y_train = _subsample_training_data(
            X_train,
            y_train,
            self.subsample_proportion,
            self.random_state,
        )

        X_arr = np.asarray(X_train)
        if X_arr.ndim != 1:
            raise ValueError(
                "DEFT MPRA loader expects sequence strings. "
                "Set `one_hot_encode=False` in the dataset config."
            )

        X_train_df = pd.DataFrame(X_arr, columns=["raw_sequence"])
        y_train_df = pd.DataFrame(np.asarray(y_train), columns=["is_enhancer"])
        return {
            "X_train": X_train_df,
            "y_train": y_train_df,
            "tree_filename": (
                f"tree_{self.condition_name}_subsample_{self.subsample_proportion}.dill"
            ),
            "prompt_context": {
                "condition_details": _get_condition_details(self.condition_name),
            },
        }

    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        """Load for baseline."""
        if str(featurizer_name) == "kmer_count":
            X_train, y_train, X_test, y_test = self._load_mpra(
                one_hot_encode=False,
                use_2D=False,
            )
            X_train = pd.DataFrame(X_train, columns=["raw_sequence"])
            X_test = pd.DataFrame(X_test, columns=["raw_sequence"])
            return X_train, X_test, y_train, y_test

        X_train, y_train, X_test, y_test = self._load_mpra(
            one_hot_encode=True,
            use_2D=True,
        )
        return X_train, X_test, y_train, y_test

    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        """Load for dl."""
        X_train, y_train, X_test, y_test = self._load_mpra(
            one_hot_encode=True,
            use_2D=False,
        )
        return X_train, X_test, y_train, y_test


class PolymeraseBigDatasetPlugin(BaseDatasetPlugin):
    dl_subsample_before_val_split = False

    def __init__(
        self,
        data_dir: str = "data/polymerase_big/processed",
        train_filename: str = "train_polymerase_big.csv",
        test_filename: str = "test_polymerase_big.csv",
        sequence_column: str = "raw_sequence",
        label_column: str = "paused",
        subsample_proportion: float = 1.0,
        random_state: int = 42,
    ):
        """Initialize the instance."""
        repo_root = Path(__file__).resolve().parents[2]
        self.data_dir = (repo_root / data_dir).resolve()
        self.train_path = self.data_dir / train_filename
        self.test_path = self.data_dir / test_filename
        self.sequence_column = str(sequence_column)
        self.label_column = str(label_column)
        self.subsample_proportion = float(subsample_proportion)
        self.random_state = int(random_state)

    def _load_split(self, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load split."""
        if not path.exists():
            raise FileNotFoundError(
                f"Missing processed split file: {path}. "
                "Run notebooks/prepare_polymerase_big_split.ipynb first."
            )
        df = pd.read_csv(path)
        if self.sequence_column not in df.columns or self.label_column not in df.columns:
            raise KeyError(
                f"{path} must contain columns `{self.sequence_column}` and `{self.label_column}`."
            )
        X = pd.DataFrame(df[self.sequence_column].astype(str), columns=["raw_sequence"])
        y = pd.DataFrame(df[self.label_column].astype(int), columns=[self.label_column])
        return X, y

    @staticmethod
    def _one_hot_flatten(raw_sequences: pd.Series) -> pd.DataFrame:
        """Handle one hot flatten."""
        seq_df = pd.DataFrame(raw_sequences.apply(list).tolist())
        seq_df.columns = [f"seq_{i}" for i in range(seq_df.shape[1])]
        return pd.get_dummies(seq_df)

    def _load_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train test."""
        X_train, y_train = self._load_split(self.train_path)
        X_test, y_test = self._load_split(self.test_path)
        return X_train, X_test, y_train, y_test

    def load_for_deft(self) -> dict[str, Any]:
        """Load for deft."""
        X_train, _, y_train, _ = self._load_train_test()
        X_train, y_train = _subsample_training_data(
            X_train,
            y_train,
            self.subsample_proportion,
            self.random_state,
        )
        return {
            "X_train": X_train,
            "y_train": y_train,
            "tree_filename": "tree_polymerase_big.dill",
            "prompt_context": {},
        }

    def load_for_baseline(
        self, *, featurizer_name: str, seed: int
    ) -> tuple[Any, Any, Any, Any]:
        """Load for baseline."""
        X_train, X_test, y_train, y_test = self._load_train_test()

        if str(featurizer_name) == "kmer_count":
            return X_train, X_test, y_train, y_test

        train_encoded = self._one_hot_flatten(X_train["raw_sequence"])
        test_encoded = self._one_hot_flatten(X_test["raw_sequence"])
        aligned = pd.concat([train_encoded, test_encoded], axis=0).fillna(0)
        X_train_encoded = aligned.iloc[: len(train_encoded)].reset_index(drop=True)
        X_test_encoded = aligned.iloc[len(train_encoded) :].reset_index(drop=True)
        return X_train_encoded, X_test_encoded, y_train, y_test

    def load_for_dl(self, *, seed: int) -> tuple[Any, Any, Any, Any]:
        """Load for dl."""
        X_train, X_test, y_train, y_test = self._load_train_test()
        train_encoded = self._one_hot_flatten(X_train["raw_sequence"])
        test_encoded = self._one_hot_flatten(X_test["raw_sequence"])
        aligned = pd.concat([train_encoded, test_encoded], axis=0).fillna(0)
        X_train_encoded = aligned.iloc[: len(train_encoded)].reset_index(drop=True)
        X_test_encoded = aligned.iloc[len(train_encoded) :].reset_index(drop=True)
        return X_train_encoded, X_test_encoded, y_train, y_test


def instantiate_dataset_plugin(dataset_cfg: Any) -> BaseDatasetPlugin:
    """Instantiate dataset plugin."""
    plugin_cfg = None
    if isinstance(dataset_cfg, dict):
        plugin_cfg = dataset_cfg.get("plugin")
    elif hasattr(dataset_cfg, "get"):
        try:
            plugin_cfg = dataset_cfg.get("plugin")
        except Exception:
            plugin_cfg = None
    if plugin_cfg is None:
        plugin_cfg = getattr(dataset_cfg, "plugin", None)

    if plugin_cfg is None:
        raise ValueError(
            "Dataset config is missing required `plugin` block. "
            "Define `dataset.plugin._target_` in conf/dataset/<name>.yaml."
        )

    plugin = instantiate(plugin_cfg)
    if not isinstance(plugin, BaseDatasetPlugin):
        raise TypeError(
            "Dataset plugin config must instantiate a `BaseDatasetPlugin`, "
            f"got: {type(plugin)}"
        )
    return plugin
