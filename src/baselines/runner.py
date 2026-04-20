from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.baselines.data_loading import load_dataset_for_baseline
from src.baselines.featurizers import transform_features


def expand_sweep_values(baseline_model_cfg: Any) -> List[float]:
    """Handle expand sweep values."""
    sweep_param = str(baseline_model_cfg.sweep_param)

    if sweep_param == "max_depth":
        max_depth = int(baseline_model_cfg.max_depth)
        return list(range(1, max_depth + 1))

    if sweep_param == "C" and "C_values" in baseline_model_cfg:
        return [float(v) for v in baseline_model_cfg.C_values]

    if "sweep_values" in baseline_model_cfg and baseline_model_cfg.sweep_values is not None:
        return [float(v) for v in baseline_model_cfg.sweep_values]

    if sweep_param in baseline_model_cfg:
        return [float(baseline_model_cfg[sweep_param])]

    raise ValueError(f"Unable to infer sweep values for sweep_param='{sweep_param}'")


def _build_model_instance(baseline_model_cfg: Any, seed: int, sweep_value: float) -> Any:
    """Build model instance."""
    excluded_keys = {
        "_target_",
        "name",
        "sweep_param",
        "sweep_values",
        "C_values",
        "random_state_from_seed",
    }

    init_kwargs = {}
    for key, value in baseline_model_cfg.items():
        if key in excluded_keys:
            continue
        init_kwargs[key] = value

    sweep_param = str(baseline_model_cfg.sweep_param)
    init_kwargs[sweep_param] = sweep_value

    use_seed_for_random_state = bool(
        baseline_model_cfg.get("random_state_from_seed", True)
    )
    if "random_state" in init_kwargs and use_seed_for_random_state:
        init_kwargs["random_state"] = seed

    model_cfg = {"_target_": baseline_model_cfg._target_, **init_kwargs}
    return instantiate(model_cfg)


def _safe_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Handle safe auprc."""
    try:
        return float(average_precision_score(y_true, y_scores))
    except ValueError:
        return float("nan")


def _positive_class_scores(y_proba: np.ndarray) -> np.ndarray:
    """Handle positive class scores."""
    arr = np.asarray(y_proba)
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr[:, 1]
    return arr.ravel()


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auprc": _safe_auprc(y_true, y_scores),
    }


def _add_optional_metadata(cfg: Any, row: Dict[str, Any]) -> Dict[str, Any]:
    """Handle add optional metadata."""
    if str(cfg.featurizer.name) == "kmer_count" and "k" in cfg.featurizer:
        row["kmer_size"] = int(cfg.featurizer.k)

    if str(cfg.baseline_model.name) == "oc1" and "num_tries" in cfg.baseline_model:
        row["num_tries"] = int(cfg.baseline_model.num_tries)

    if "subsample_proportion" in cfg.dataset:
        row["subsample_proportion"] = float(cfg.dataset.subsample_proportion)

    return row


def _filter_metric_columns(df: pd.DataFrame, metrics_to_keep: Iterable[str]) -> pd.DataFrame:
    """Filter metric columns."""
    keep = set(metrics_to_keep)
    all_metrics = {"accuracy", "f1", "precision", "recall", "auprc"}
    for col in all_metrics - keep:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def run_baseline_experiment(cfg: Any, project_root: str | Path) -> Tuple[pd.DataFrame, Path]:
    """Run baseline experiment."""
    rows: List[Dict[str, Any]] = []
    sweep_values = expand_sweep_values(cfg.baseline_model)

    for seed in cfg.seed_list:
        data = load_dataset_for_baseline(cfg, int(seed))

        X_train, X_test = transform_features(
            cfg.featurizer, data["X_train_raw"], data["X_test_raw"]
        )
        y_train = np.asarray(data["y_train"]).ravel()
        y_test = np.asarray(data["y_test"]).ravel()

        for sweep_value in sweep_values:
            model = _build_model_instance(cfg.baseline_model, int(seed), sweep_value)
            model.fit(X_train, y_train)

            y_pred_train = model.predict(X_train)
            y_prob_train = _positive_class_scores(model.predict_proba(X_train))
            y_pred_test = model.predict(X_test)
            y_prob_test = _positive_class_scores(model.predict_proba(X_test))

            metrics_train = _compute_metrics(y_train, y_pred_train, y_prob_train)
            metrics_test = _compute_metrics(y_test, y_pred_test, y_prob_test)

            shared = {
                "dataset": str(cfg.dataset.name),
                "model": str(cfg.baseline_model.name),
                "featurizer": str(cfg.featurizer.name),
                "seed": int(seed),
                "sweep_param": str(cfg.baseline_model.sweep_param),
                "sweep_value": float(sweep_value),
            }
            shared = _add_optional_metadata(cfg, shared)

            rows.append({**shared, "split": "train", **metrics_train})
            rows.append({**shared, "split": "test", **metrics_test})

    df = pd.DataFrame(rows)
    df = _filter_metric_columns(df, cfg.metrics)

    output_dir = Path(project_root) / str(cfg.output_dir_template)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(cfg.output_file_template)
    df.to_csv(output_path, index=False)

    return df, output_path
