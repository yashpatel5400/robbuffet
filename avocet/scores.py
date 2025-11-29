from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch

from .region import (
    EllipsoidRegion,
    L1BallRegion,
    L2BallRegion,
    LinfBallRegion,
    PredictionRegion,
)


class ScoreFunction(abc.ABC):
    """Base interface for score functions."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute nonconformity scores for a batch."""

    @abc.abstractmethod
    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        """Construct a prediction region given a point prediction and score quantile."""


class L2Score(ScoreFunction):
    """
    Standard residual score: s(x, y) = ||pred - y||_2.
    Induces an L2 ball prediction region around the point prediction.
    """

    @property
    def name(self) -> str:
        return "l2"

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(prediction - target, p=2, dim=-1)

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy()
        if center.ndim == 0:
            center = center.reshape(1)
        return L2BallRegion(center=center, radius=float(quantile))


class L1Score(ScoreFunction):
    """
    L1 residual score: s(x, y) = ||pred - y||_1.
    Induces an L1 ball prediction region around the point prediction.
    """

    @property
    def name(self) -> str:
        return "l1"

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(prediction - target, p=1, dim=-1)

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy()
        if center.ndim == 0:
            center = center.reshape(1)
        return L1BallRegion(center=center, radius=float(quantile))


class LinfScore(ScoreFunction):
    """
    Linf residual score: s(x, y) = ||pred - y||_inf.
    Induces an L-infinity (hypercube) prediction region around the point prediction.
    """

    @property
    def name(self) -> str:
        return "linf"

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(prediction - target, p=float("inf"), dim=-1)

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy()
        if center.ndim == 0:
            center = center.reshape(1)
        return LinfBallRegion(center=center, radius=float(quantile))


class MahalanobisScore(ScoreFunction):
    """
    Mahalanobis residual: s(x, y) = sqrt((pred - y)^T W (pred - y)).
    W must be positive definite; induces an ellipsoidal region.
    """

    def __init__(self, weight: np.ndarray):
        if weight.shape[0] != weight.shape[1]:
            raise ValueError("weight must be square")
        self.weight = weight

    @property
    def name(self) -> str:
        return "mahalanobis"

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        w = torch.as_tensor(self.weight, device=diff.device, dtype=diff.dtype)
        # batch quadratic form
        return torch.sqrt(torch.einsum("bi,ij,bj->b", diff, w, diff))

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy()
        if center.ndim == 0:
            center = center.reshape(1)
        return EllipsoidRegion(center=center, shape_matrix=self.weight, radius=float(quantile))


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    """
    Compute the split-conformal quantile with finite-sample correction.

    Quantile index k = ceil((n + 1) * (1 - alpha)) - 1 over sorted scores.
    """
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    n = scores.numel()
    if n == 0:
        raise ValueError("No calibration scores provided.")
    k = int(np.ceil((n + 1) * (1 - alpha)) - 1)
    k = min(max(k, 0), n - 1)
    sorted_scores = torch.sort(scores)[0]
    return float(sorted_scores[k].item())
