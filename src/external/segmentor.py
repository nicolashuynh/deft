#
# Import all the packages
#
from abc import ABCMeta, abstractmethod

import numpy as np

#
#
#


class Twoing:
    """
    Twoing rule for impurity criterion
    """

    def __call__(self, left_label, right_label):
        """Run the instance as a callable."""
        sum = 0
        huge_val = np.inf
        left_len, right_len, n = (
            len(left_label),
            len(right_label),
            (len(left_label) + len(right_label)),
        )
        labels = list(left_label) + list(right_label)
        n_classes = np.unique(labels)
        if left_len != 0 & right_len != 0:
            for i in n_classes:
                idx = np.where(left_label == i)[0]
                li = len(idx) / left_len
                idx = np.where(right_label == i)[0]
                ri = len(idx) / right_len
                sum += np.abs(li - ri)
            twoing_value = ((left_len / n) * (right_len / n) * np.square(sum)) / 4

        elif left_len == 0:
            for i in n_classes:
                idx = np.where(right_label == i)[0]
                ri = len(idx) / right_len
                sum += ri
            twoing_value = ((left_len / n) * (right_len / n) * np.square(sum)) / 4

        else:
            for i in n_classes:
                idx = np.where(left_label == i)[0]
                li = len(idx) / left_len
                sum += li
            twoing_value = ((left_len / n) * (right_len / n) * np.square(sum)) / 4
        if twoing_value == 0:
            return huge_val
        else:
            return 1 / twoing_value


class MSE:
    """
    Mean squared error impurity criterion
    """

    def __call__(self, left_label, right_label):
        """Run the instance as a callable."""
        left_len, right_len = len(left_label), len(right_label)

        left_std = np.std(left_label)
        right_std = np.std(right_label)

        total = left_len + right_len

        return (left_len / total) * left_std + (right_len / total) * right_std


class SegmentorBase:
    """
    Abstract segmentor class. Segmentor called in nodes for find best split.


    Parameters
    -----------

    msl : int, optional (default=1)
    The minimum number of samples required to be at a leaf nodes.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def _split_generator(self, X):
        """
        Abstract method for split.


        Parameter
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.
        """
        pass

    def __init__(self, msl=2):
        """Initialize the instance."""
        self._min_samples_leaf = msl

    def __call__(self, X, y, impurity=MSE()):
        """
        Parameters
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.
        y : array-like, shape = [n_samples]
        The target values.
        impurity : object of impurity (default=MSE()).
        The name of criterion to measure the quality of a split.


        Returns
        -----------

        Tuple of the following elements:
        best_impurity : float.
        The best value of impurity.
        best_split_rule : tuple.
        The pair of feature and value.
        best_left_i : numpy.ndarray.
        The indexes of left node objects.
        best_right_i : numpy.ndarray.
        The indexes of right node objects.
        """
        best_impurity = float("inf")
        best_split_rule = None
        best_left_i = None
        best_right_i = None
        splits = self._split_generator(X)

        for left_i, right_i, split_rule in splits:
            if (
                left_i.size > self._min_samples_leaf
                and right_i.size > self._min_samples_leaf
            ):
                left_labels, right_labels = y[left_i], y[right_i]
                cur_impurity = impurity(left_labels, right_labels)
                if cur_impurity < best_impurity:
                    best_impurity = cur_impurity
                    best_split_rule = split_rule
                    best_left_i = left_i
                    best_right_i = right_i
        return (best_impurity, best_split_rule, best_left_i, best_right_i)


class MeanSegmentor(SegmentorBase):
    """
    Split based on mean value of each feature.
    """

    def _split_generator(self, X):
        """
        Parameters
        -----------

        X : array-like, shape = [n_samples, n_features]
        The training input samples.


        Returns
        -----------

        Tuple of the following elements:
        left_i : numpy.ndarray.
        The indexes of left node objects.
        right_i : numpy.ndarray.
        The indexes of right node objects.
        split_rule : tuple.
        The pair of feature and value.
        """
        for feature_i in range(X.shape[1]):
            feature_values = X[:, feature_i]
            mean = np.mean(feature_values)
            left_i = np.nonzero(feature_values < mean)[0]
            right_i = np.nonzero(feature_values >= mean)[0]
            split_rule = (feature_i, mean)
            yield (left_i, right_i, split_rule)


class ClassificationSegmentor(SegmentorBase):
    """
    Segmentor suitable for classification criteria like Gini, Entropy, or Twoing.
    Splits numeric features by testing thresholds between unique sorted values.
    Assumes features are numeric. For categorical features, different logic is needed.
    """

    def _split_generator(self, X):
        """
        Generates potential splits for each feature. For numeric features,
        it tests splits at the midpoints between unique sorted values.

        Parameters: See SegmentorBase._split_generator
        Yields: Tuples of (left_indices, right_indices, split_rule=(feature_idx, threshold))
        """
        n_samples = X.shape[0]
        if n_samples == 0:  # No samples, no splits
            return
        indices = np.arange(n_samples)

        for feature_i in range(X.shape[1]):
            feature_values = X[:, feature_i]

            try:
                # Find unique sorted values in the current feature column
                unique_values = np.unique(feature_values)
            except TypeError:
                # Skip feature if it contains non-sortable types (e.g., mixed str/int)
                continue

            # Need at least two unique values to make a split
            if unique_values.size <= 1:
                continue

            # Generate potential thresholds: midpoints between unique values
            # Example: unique values [1, 5, 10] -> thresholds [3, 7.5]
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0

            for threshold in thresholds:
                # Split data based on the current threshold
                # Using <= for left split is a common convention
                left_i = indices[feature_values <= threshold]
                right_i = indices[feature_values > threshold]

                # We must yield splits even if one side might be smaller than min_samples_leaf
                # The __call__ method in SegmentorBase handles the filtering.
                # However, we must ensure neither side is completely empty after a split.
                if left_i.size > 0 and right_i.size > 0:
                    split_rule = (feature_i, threshold)
                    yield (left_i, right_i, split_rule)
