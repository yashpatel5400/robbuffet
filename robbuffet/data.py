from __future__ import annotations

import abc
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


class BaseDataset(abc.ABC):
    """Abstract dataset with train/calibration/test splits."""

    @property
    @abc.abstractmethod
    def train(self) -> TensorDataset:
        ...

    @property
    @abc.abstractmethod
    def calibration(self) -> TensorDataset:
        ...

    @property
    @abc.abstractmethod
    def test(self) -> TensorDataset:
        ...


class OfflineDataset(BaseDataset):
    """
    Dataset wrapper for pre-loaded arrays/tensors. Splits into train/cal/test.
    """

    def __init__(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        train_frac: float = 0.6,
        cal_frac: float = 0.2,
        seed: int = 0,
        shuffle: bool = True,
    ):
        if abs(train_frac + cal_frac) > 1.0:
            raise ValueError("train_frac + cal_frac must be <= 1.0")
        X_t = torch.as_tensor(X)
        y_t = torch.as_tensor(y)
        if X_t.shape[0] != y_t.shape[0]:
            raise ValueError("X and y must have same number of rows.")
        n = X_t.shape[0]
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        X_t = X_t[idx]
        y_t = y_t[idx]
        n_train = int(train_frac * n)
        n_cal = int(cal_frac * n)
        n_test = n - n_train - n_cal
        if n_test <= 0:
            raise ValueError("Not enough samples for test split; adjust fractions.")

        self._train = TensorDataset(X_t[:n_train], y_t[:n_train])
        self._cal = TensorDataset(X_t[n_train : n_train + n_cal], y_t[n_train : n_train + n_cal])
        self._test = TensorDataset(X_t[-n_test:], y_t[-n_test:])

    @property
    def train(self) -> TensorDataset:
        return self._train

    @property
    def calibration(self) -> TensorDataset:
        return self._cal

    @property
    def test(self) -> TensorDataset:
        return self._test


class GeneratorDataset(BaseDataset):
    """
    Dataset wrapper around callables producing TensorDatasets for each split.

    Provide functions train_fn(), cal_fn(), test_fn() that return TensorDataset.
    """

    def __init__(
        self,
        train_fn: Callable[[], TensorDataset],
        cal_fn: Callable[[], TensorDataset],
        test_fn: Callable[[], TensorDataset],
    ):
        self._train_fn = train_fn
        self._cal_fn = cal_fn
        self._test_fn = test_fn
        self._train: Optional[TensorDataset] = None
        self._cal: Optional[TensorDataset] = None
        self._test: Optional[TensorDataset] = None

    @property
    def train(self) -> TensorDataset:
        if self._train is None:
            self._train = self._train_fn()
        return self._train

    @property
    def calibration(self) -> TensorDataset:
        if self._cal is None:
            self._cal = self._cal_fn()
        return self._cal

    @property
    def test(self) -> TensorDataset:
        if self._test is None:
            self._test = self._test_fn()
        return self._test


class SimulationDataset(BaseDataset):
    """
    Dataset generated on the fly via simulators:
      - x_sampler(n, rng) -> features/context X
      - y_sampler(X, rng) -> labels/targets Y (can sample from P(y|x))

    Splits are generated once and cached.
    """

    def __init__(
        self,
        x_sampler: Callable[[int, np.random.Generator], np.ndarray | torch.Tensor],
        y_sampler: Callable[[np.ndarray | torch.Tensor, np.random.Generator], np.ndarray | torch.Tensor],
        train_size: int,
        cal_size: int,
        test_size: int,
        seed: int = 0,
    ):
        self.x_sampler = x_sampler
        self.y_sampler = y_sampler
        self.train_size = train_size
        self.cal_size = cal_size
        self.test_size = test_size
        self.rng = np.random.default_rng(seed)
        self._train: Optional[TensorDataset] = None
        self._cal: Optional[TensorDataset] = None
        self._test: Optional[TensorDataset] = None

    def _generate(self, n: int) -> TensorDataset:
        X = self.x_sampler(n, self.rng)
        X_t = torch.as_tensor(X)
        Y = self.y_sampler(X, self.rng)
        Y_t = torch.as_tensor(Y)
        return TensorDataset(X_t, Y_t)

    @property
    def train(self) -> TensorDataset:
        if self._train is None:
            self._train = self._generate(self.train_size)
        return self._train

    @property
    def calibration(self) -> TensorDataset:
        if self._cal is None:
            self._cal = self._generate(self.cal_size)
        return self._cal

    @property
    def test(self) -> TensorDataset:
        if self._test is None:
            self._test = self._generate(self.test_size)
        return self._test
