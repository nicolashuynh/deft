from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset_plugins import instantiate_dataset_plugin

SUPPORTED_DATASETS = {"polymerase", "promoters", "mpra_enhancers", "mpra_easy", "mpra"}
SUPPORTED_MODEL_ALIASES = {"cnn", "transformer"}


def _to_1d_numpy(y: Any) -> np.ndarray:
    """Handle to 1d numpy."""
    if isinstance(y, (pd.DataFrame, pd.Series)):
        return y.to_numpy().ravel()
    return np.asarray(y).ravel()


def _index_rows(X: Any, indices: np.ndarray) -> Any:
    """Handle index rows."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[indices]
    return np.asarray(X)[indices]


def _subsample_train(
    X_train: Any,
    y_train: Any,
    subsample_proportion: float,
    seed: int,
) -> tuple[Any, Any]:
    """Subsample train."""
    if subsample_proportion >= 1.0:
        return X_train, y_train

    n_keep = max(1, int(len(X_train) * subsample_proportion))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X_train), n_keep, replace=False)
    return _index_rows(X_train, indices), _index_rows(y_train, indices)


def _load_dataset_splits(
    cfg: Any, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset splits."""
    plugin = instantiate_dataset_plugin(cfg.dataset)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = plugin.load_for_dl(seed=int(seed))

    subsample_proportion = float(cfg.dataset.subsample_proportion)
    if plugin.dl_subsample_before_val_split:
        X_train_raw, y_train_raw = _subsample_train(
            X_train_raw,
            y_train_raw,
            subsample_proportion,
            seed,
        )

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_train_raw,
        y_train_raw,
        test_size=cfg.dataset.val_size,
        random_state=seed,
        stratify=y_train_raw,
    )

    if not plugin.dl_subsample_before_val_split:
        X_train_raw, y_train_raw = _subsample_train(
            X_train_raw,
            y_train_raw,
            subsample_proportion,
            seed,
        )

    X_train = np.asarray(X_train_raw).reshape(
        -1, int(cfg.dataset.sequence_length), int(cfg.dataset.n_nucleotides)
    )
    X_val = np.asarray(X_val_raw).reshape(
        -1, int(cfg.dataset.sequence_length), int(cfg.dataset.n_nucleotides)
    )
    X_test = np.asarray(X_test_raw).reshape(
        -1, int(cfg.dataset.sequence_length), int(cfg.dataset.n_nucleotides)
    )

    y_train = _to_1d_numpy(y_train_raw)
    y_val = _to_1d_numpy(y_val_raw)
    y_test = _to_1d_numpy(y_test_raw)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Build loader."""
    return DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _get_predictions_and_scores(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Get predictions and scores."""
    logits_list = trainer.predict(model, dataloaders=loader)
    logits = torch.cat(logits_list)
    probabilities = torch.sigmoid(logits).squeeze().numpy()
    predictions = (probabilities > 0.5).astype(int)
    return predictions, probabilities


def _safe_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Handle safe auprc."""
    try:
        return float(average_precision_score(y_true, y_scores))
    except ValueError:
        return float("nan")


def _build_metrics_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
    is_train: bool,
    split: str,
    method_name: str,
    seed: int,
) -> Dict[str, Any]:
    """Build metrics row."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auprc": _safe_auprc(y_true, y_scores),
        "isTrain": bool(is_train),
        "split": split,
        "method": method_name,
        "random_seed": int(seed),
    }


def _filter_metric_columns(df: pd.DataFrame, metrics_to_keep: Iterable[str]) -> pd.DataFrame:
    """Filter metric columns."""
    keep = set(metrics_to_keep)
    all_metrics = {"accuracy", "f1", "precision", "recall", "auprc"}
    for col in all_metrics - keep:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _resolve_model_spec(cfg: Any, dataset_name: str, model_alias: str) -> Dict[str, Any]:
    """Resolve model spec."""
    if dataset_name not in cfg.model_registry:
        raise ValueError(
            f"No model registry found for dataset '{dataset_name}'. Available datasets: "
            f"{list(cfg.model_registry.keys())}"
        )

    if model_alias not in cfg.model_registry[dataset_name]:
        raise ValueError(
            f"No model alias '{model_alias}' configured for dataset '{dataset_name}'. "
            f"Available: {list(cfg.model_registry[dataset_name].keys())}"
        )

    spec = cfg.model_registry[dataset_name][model_alias]
    resolved = {
        "dl_model": str(spec.dl_model),
        "learning_rate": float(spec.learning_rate),
        "batch_size": int(spec.batch_size),
    }
    if "patience" in spec:
        resolved["patience"] = int(spec.patience)
    return resolved


def _load_model_cfg_from_name(
    project_root: str | Path, cfg: Any, model_name: str, learning_rate: float
) -> Any:
    """Load model cfg from name."""
    model_cfg_path = Path(project_root) / "conf" / "dl_model" / f"{model_name}.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"Unable to find dl_model config: {model_cfg_path}")

    raw_model_cfg = OmegaConf.load(model_cfg_path)
    merged = OmegaConf.merge(
        OmegaConf.create({"dataset": cfg.dataset, "learning_rate": learning_rate}),
        OmegaConf.create({"dl_model": raw_model_cfg}),
    )
    OmegaConf.resolve(merged)
    return merged.dl_model


def _build_output_path(project_root: str | Path, cfg: Any, model_class_name: str) -> Path:
    """Build output path."""
    output_dir = Path(project_root) / str(cfg.output_dir_template)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{model_class_name}.csv"


def _build_trainer(cfg: Any, patience: int | None = None) -> pl.Trainer:
    """Build trainer."""
    effective_patience = int(patience) if patience is not None else int(cfg.patience)
    callbacks = []
    if bool(cfg.early_stopping):
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="val_auprc",
                patience=effective_patience,
                mode="max",
                verbose=True,
            )
        )

    return pl.Trainer(
        max_epochs=int(cfg.max_epochs),
        accelerator="auto",
        deterministic=bool(getattr(cfg, "deterministic", True)),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(getattr(cfg, "enable_progress_bar", True)),
        callbacks=callbacks,
    )


def _run_single_model(
    cfg: Any,
    project_root: str | Path,
    model_alias: str,
) -> tuple[pd.DataFrame, Path, str]:
    """Run single model."""
    dataset_name = str(cfg.dataset.name)
    model_spec = _resolve_model_spec(cfg, dataset_name, model_alias)
    model_cfg = _load_model_cfg_from_name(
        project_root, cfg, model_spec["dl_model"], model_spec["learning_rate"]
    )

    model_class_name = str(model_cfg._target_).split(".")[-1]
    rows: List[Dict[str, Any]] = []

    for seed in cfg.seed_list:
        seed = int(seed)
        pl.seed_everything(seed, workers=True)

        X_train, X_val, X_test, y_train, y_val, y_test = _load_dataset_splits(cfg, seed)

        train_loader = _build_loader(X_train, y_train, model_spec["batch_size"], shuffle=True)
        val_loader = _build_loader(X_val, y_val, model_spec["batch_size"], shuffle=False)
        test_loader = _build_loader(X_test, y_test, model_spec["batch_size"], shuffle=False)
        full_train_eval_loader = _build_loader(
            X_train, y_train, model_spec["batch_size"], shuffle=False
        )

        model = hydra.utils.instantiate(model_cfg)
        trainer = _build_trainer(cfg, patience=model_spec.get("patience"))
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        y_pred_train, y_scores_train = _get_predictions_and_scores(
            trainer, model, full_train_eval_loader
        )
        y_pred_test, y_scores_test = _get_predictions_and_scores(trainer, model, test_loader)

        method_name = f"{model_class_name}_{dataset_name}"
        rows.append(
            _build_metrics_row(
                y_true=y_train,
                y_pred=y_pred_train,
                y_scores=y_scores_train,
                is_train=True,
                split="train",
                method_name=method_name,
                seed=seed,
            )
        )
        rows.append(
            _build_metrics_row(
                y_true=y_test,
                y_pred=y_pred_test,
                y_scores=y_scores_test,
                is_train=False,
                split="test",
                method_name=method_name,
                seed=seed,
            )
        )

    df_model = pd.DataFrame(rows)
    df_model = _filter_metric_columns(df_model, cfg.metrics)

    output_path = _build_output_path(project_root, cfg, model_class_name)
    df_model.to_csv(output_path, index=False)

    return df_model, output_path, model_class_name


def run_dl_baselines_experiment(
    cfg: Any,
    project_root: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """Run dl baselines experiment."""
    dataset_name = str(cfg.dataset.name)
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    model_aliases = [str(name).lower() for name in cfg.models_to_run]
    invalid_aliases = sorted(set(model_aliases) - SUPPORTED_MODEL_ALIASES)
    if invalid_aliases:
        raise ValueError(
            f"Unsupported model alias(es): {invalid_aliases}. "
            f"Supported aliases: {sorted(SUPPORTED_MODEL_ALIASES)}"
        )

    all_results: List[pd.DataFrame] = []
    output_paths: Dict[str, Path] = {}

    for model_alias in model_aliases:
        df_model, output_path, model_class_name = _run_single_model(
            cfg, project_root, model_alias
        )
        all_results.append(df_model)
        output_paths[model_class_name] = output_path

    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
    else:
        df_results = pd.DataFrame()

    return df_results, output_paths
