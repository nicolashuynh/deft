import random
from abc import ABC, abstractmethod
from typing import List, TypeVar

import numpy as np

from src.utils.feature import FeatureInfo

T = TypeVar("T")


class Selector(ABC):
    """Abstract base class for selection strategies."""

    @abstractmethod
    def select(self, population: List[T], k: int = 1) -> List[T]:
        """
        Select k items from the population.

        Args:
            population: List of items to select from
            k: Number of items to select

        Returns:
            List of selected items
        """
        pass


class TournamentSelector(Selector):
    """Tournament selection implementation."""

    def __init__(self, tournament_size: int = 2):
        """Initialize the instance."""
        self.tournament_size = tournament_size

    def select(self, population: List[FeatureInfo], k: int = 1) -> List[FeatureInfo]:
        """Select values."""
        selected = []
        for _ in range(k):
            # Select random candidates for tournament
            candidates = random.sample(population, self.tournament_size)
            # Select the candidate with highest score
            winner = max(candidates, key=lambda x: x.score)
            selected.append(winner)
        return selected


class RankSelector(Selector):
    """Rank-based selection implementation."""

    def select(self, population: List[FeatureInfo], k: int = 1) -> List[FeatureInfo]:
        """Select values."""
        N = len(population)
        # Sort population by score in descending order
        sorted_pop = sorted(population, key=lambda x: x.score, reverse=True)

        # Calculate selection probabilities
        ranks = np.arange(1, N + 1)
        probabilities = 1 / (ranks + N)
        probabilities = probabilities / probabilities.sum()  # Normalize

        # Select k items based on rank probabilities
        selected_indices = np.random.choice(N, size=k, p=probabilities, replace=True)

        return [sorted_pop[i] for i in selected_indices]


class TopKSelector(Selector):
    """Select the top k items based on score."""

    def select(self, population: List[FeatureInfo], k: int = 1) -> List[FeatureInfo]:
        """Select values."""
        return sorted(population, key=lambda x: x.score, reverse=True)[:k]
