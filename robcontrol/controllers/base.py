from __future__ import annotations

import abc
from typing import Protocol

import numpy as np


class RobustController(abc.ABC):
    """Abstract interface for robust controller synthesis."""

    @abc.abstractmethod
    def synthesize(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Return feedback gain K (m x n) for given dynamics and cost matrices."""
        raise NotImplementedError
