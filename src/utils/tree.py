import gzip
import io
from copy import copy, deepcopy

import dill
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_feature_function(feature_fn, X):
    """Evaluate `feature_fn` on a defensive copy to prevent in-place mutations."""
    return feature_fn(X.copy(deep=True))


def get_leaf_data_with_history(node, X, y, path_to_leaf):
    """
    Gets the X and y data that corresponds to a specific leaf, along with the history of splits.
    """
    mask = np.ones(len(X), dtype=bool)
    current_node = node
    current_history = []

    for direction in path_to_leaf:
        if current_node._is_leaf():
            raise ValueError("Path extends beyond leaf node")

        left_mask, right_mask = _compute_split_masks(
            node=current_node,
            X_subset=X,
            right_is_complement=False,
            as_numpy=True,
        )

        # Update history before moving to next node
        direction_str = "smaller" if direction == "left" else "greater"
        current_history = current_history + [
            (copy(current_node.feature), direction_str)
        ]

        if direction == "left":
            mask &= left_mask
            current_node = current_node.left
        else:
            mask &= right_mask
            current_node = current_node.right

    return X[mask], y[mask], current_history


def predict_at_depth(root, X, depth, return_proba=False):
    """Predict at depth."""
    return predict_at_depth_vectorized(
        root=root,
        X=X,
        depth=depth,
        return_proba=return_proba,
    )


def _prediction_to_scalar(prediction):
    """Normalize node predictions to a scalar float."""
    if isinstance(prediction, pd.DataFrame):
        prediction = prediction.squeeze()
    if isinstance(prediction, pd.Series):
        if prediction.empty:
            return np.nan
        return float(prediction.iloc[0])

    arr = np.asarray(prediction)
    if arr.ndim == 0:
        return float(arr.item())
    if arr.size == 0:
        return np.nan
    return float(arr.reshape(-1)[0])


def feature_output_to_series(feature_output, index):
    """Convert a feature function output to a Series aligned with `index`."""
    if isinstance(feature_output, pd.DataFrame):
        if feature_output.shape[1] == 1:
            feature_output = feature_output.iloc[:, 0]
        else:
            feature_output = feature_output.squeeze()

    if isinstance(feature_output, pd.Series):
        if not feature_output.index.equals(index):
            try:
                feature_output = feature_output.reindex(index)
            except Exception:
                feature_output = pd.Series(feature_output.to_numpy(), index=index)
        return feature_output

    arr = np.asarray(feature_output)
    if arr.ndim == 0:
        return pd.Series(np.repeat(arr.item(), len(index)), index=index)

    arr = arr.reshape(-1)
    if arr.size != len(index):
        raise ValueError(
            "Feature function output size does not match number of input samples."
        )
    return pd.Series(arr, index=index)


def _compute_split_masks(
    node, X_subset, right_is_complement=True, as_numpy=False
):
    """
    Compute left/right split masks for a node on `X_subset`.
    """
    feature_values = evaluate_feature_function(node.feature.fn, X_subset)
    feature_values = feature_output_to_series(feature_values, X_subset.index)
    left_mask = feature_values <= node.feature.threshold

    if right_is_complement:
        right_mask = ~left_mask
    else:
        right_mask = feature_values > node.feature.threshold

    if as_numpy:
        return (
            left_mask.to_numpy(dtype=bool),
            right_mask.to_numpy(dtype=bool),
        )
    return left_mask, right_mask


def _finalize_predictions(predictions, return_proba):
    """Return probabilities or thresholded predictions with consistent behavior."""
    if return_proba:
        return predictions
    return predictions > 0.5


def predict_all_depths_vectorized(root, X, max_depth, return_proba=False):
    """
    Predict for depths 0..max_depth in one incremental, vectorized traversal.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    n_samples = len(X)
    predictions = np.zeros((max_depth + 1, n_samples), dtype=float)
    if n_samples == 0:
        return predictions if return_proba else predictions > 0.5

    # Frontier stores disjoint sample groups currently at each node for this depth.
    frontier = [(root, np.arange(n_samples, dtype=int))]

    for depth in range(max_depth + 1):
        # Current-depth predictions come directly from each frontier node.
        for node, indices in frontier:
            if len(indices) == 0:
                continue
            predictions[depth, indices] = _prediction_to_scalar(node.prediction)

        if depth == max_depth:
            break

        next_frontier = []
        for node, indices in frontier:
            if len(indices) == 0:
                continue

            # Leaf predictions stay unchanged at deeper depths.
            if node._is_leaf():
                next_frontier.append((node, indices))
                continue

            X_subset = X.iloc[indices]
            left_mask, right_mask = _compute_split_masks(
                node=node,
                X_subset=X_subset,
                right_is_complement=True,
                as_numpy=True,
            )

            if left_mask.size != len(indices):
                raise ValueError(
                    "Feature mask size does not match number of node samples."
                )

            left_indices = indices[left_mask]
            right_indices = indices[right_mask]

            if len(left_indices) > 0:
                next_frontier.append((node.left, left_indices))
            if len(right_indices) > 0:
                next_frontier.append((node.right, right_indices))

        frontier = next_frontier
        if not frontier:
            break

    return _finalize_predictions(predictions, return_proba)


def predict_at_depth_vectorized(root, X, depth, return_proba=False):
    """
    Vectorized single-depth prediction helper.
    """
    all_predictions = predict_all_depths_vectorized(
        root=root, X=X, max_depth=depth, return_proba=True
    )
    predictions = all_predictions[depth]
    return _finalize_predictions(predictions, return_proba)


def compress_node(node):
    # Deep copy to avoid modifying original
    """Handle compress node."""
    node_copy = deepcopy(node)

    # Recursively compress artifacts in all nodes
    def compress_artifacts_recursive(n):
        """Handle compress artifacts recursive."""
        if n is None:
            return

        if n.artifacts:
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9) as f:
                dill.dump(n.artifacts, f)
            n.artifacts = buffer.getvalue()

        compress_artifacts_recursive(n.left)
        compress_artifacts_recursive(n.right)

    compress_artifacts_recursive(node_copy)
    return node_copy


def decompress_node(compressed_node):
    """Handle decompress node."""
    if compressed_node is None:
        return None

    def decompress_recursive(node):
        """Handle decompress recursive."""
        if node is None:
            return

        if node.artifacts:
            buffer = io.BytesIO(node.artifacts)
            with gzip.GzipFile(fileobj=buffer, mode="rb") as f:
                node.artifacts = dill.load(f)

        decompress_recursive(node.left)
        decompress_recursive(node.right)

    node_copy = deepcopy(compressed_node)
    decompress_recursive(node_copy)
    return node_copy


def _compute_metrics(y_true, y_pred, y_prob):
    """Compute the classification metrics used in result reporting."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_prob),
    }


def _build_result_row(depth, metrics, is_train, name_method, random_seed):
    """Build a single row in the results table schema."""
    return {
        "depth": depth,
        "accuracy": metrics["accuracy"],
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "auprc": metrics["auprc"],
        "isTrain": is_train,
        "method": name_method,
        "random_seed": random_seed,
    }


def get_results(
    root, X_train, X_test, y_train, y_test, range_depth, name_method, random_seed="42"
):
    """Get results."""
    depths = list(range_depth)
    result_rows = []
    if not depths:
        return pd.DataFrame(result_rows)

    max_depth = max(depths)
    train_proba_all_depths = predict_all_depths_vectorized(
        root=root, X=X_train, max_depth=max_depth, return_proba=True
    )
    test_proba_all_depths = predict_all_depths_vectorized(
        root=root, X=X_test, max_depth=max_depth, return_proba=True
    )

    for depth in depths:
        y_prob_train = train_proba_all_depths[depth]
        y_prob_test = test_proba_all_depths[depth]
        y_pred_train = y_prob_train > 0.5
        y_pred_test = y_prob_test > 0.5

        train_metrics = _compute_metrics(y_train, y_pred_train, y_prob_train)
        test_metrics = _compute_metrics(y_test, y_pred_test, y_prob_test)

        result_rows.append(
            _build_result_row(depth, train_metrics, True, name_method, random_seed)
        )
        result_rows.append(
            _build_result_row(depth, test_metrics, False, name_method, random_seed)
        )

    return pd.DataFrame(result_rows)


def _annotate_tree_node_statistics(
    tree,
    X,
    y=None,
    annotate_sample_proportions=False,
    annotate_positives=False,
    annotate_negatives=False,
):
    """Shared traversal helper to annotate per-node statistics in-place."""
    if (annotate_positives or annotate_negatives) and y is None:
        raise ValueError(
            "y must be provided when annotating positive/negative node statistics."
        )

    total_samples = len(X)

    def _annotate_node(node, X_subset, y_subset):
        """Handle annotate node."""
        if annotate_sample_proportions:
            node.sample_proportion = len(X_subset) / total_samples
            node.number_samples = len(X_subset)
            node.number_samples_train = total_samples

        if annotate_positives:
            node.number_positives = np.sum(y_subset == 1)

        if annotate_negatives:
            node.number_negatives = np.sum(y_subset == 0)

        if node._is_leaf():
            return

        left_mask, right_mask = _compute_split_masks(
            node=node,
            X_subset=X_subset,
            right_is_complement=True,
            as_numpy=False,
        )

        left_y = y_subset[left_mask] if y_subset is not None else None
        right_y = y_subset[right_mask] if y_subset is not None else None

        _annotate_node(node.left, X_subset[left_mask], left_y)
        _annotate_node(node.right, X_subset[right_mask], right_y)

    _annotate_node(tree.root, X, y)
    return tree


def calculate_sample_proportions(tree, X):
    """
    Calculate the proportion of samples that go through each node in the tree.
    """
    return _annotate_tree_node_statistics(
        tree=tree,
        X=X,
        annotate_sample_proportions=True,
    )


def calculate_positive_proportions(tree, X, y):
    """
    Calculate the number of positive labels (y == 1) that go through each node.
    """
    return _annotate_tree_node_statistics(
        tree=tree,
        X=X,
        y=y,
        annotate_positives=True,
    )


def calculate_negative_proportions(tree, X, y):
    """
    Calculate the number of negative labels (y == 0) that go through each node.
    """
    return _annotate_tree_node_statistics(
        tree=tree,
        X=X,
        y=y,
        annotate_negatives=True,
    )
