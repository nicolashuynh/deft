from abc import ABC, abstractmethod

import numpy as np

"""
Note that we want to maximize the scores (hence the negative sign)    
"""


class SplitCriterion(ABC):
    @abstractmethod
    def __call__(self, y_left, y_right):
        """Run the instance as a callable."""
        pass


class InformationGainCriterion(SplitCriterion):
    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)

    def _entropy(self, y):
        """Handle entropy."""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def __call__(self, y_left, y_right):
        """Run the instance as a callable."""
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        p_left = n_left / n_total
        p_right = n_right / n_total

        entropy_left = self._entropy(y_left)
        entropy_right = self._entropy(y_right)

        return -p_left * entropy_left - p_right * entropy_right + self.offset


class GiniCriterion(SplitCriterion):
    def __init__(self, offset: float = 1.0):
        """Initialize the instance."""
        self.offset = float(offset)

    def __call__(self, y_left, y_right):
        """Run the instance as a callable."""
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        p_left = n_left / n_total
        p_right = n_right / n_total

        gini_left = 1 - np.sum((np.unique(y_left, return_counts=True)[1] / n_left) ** 2)
        gini_right = 1 - np.sum(
            (np.unique(y_right, return_counts=True)[1] / n_right) ** 2
        )

        return -p_left * gini_left - p_right * gini_right + self.offset
