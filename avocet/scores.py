from __future__ import annotations

import abc
from typing import Any

import numpy as np
import torch

from .region import PredictionRegion, ScoreGeometry


class ScoreFunction(abc.ABC):
    """Base interface for score functions."""

    geometry: ScoreGeometry

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

    def __init__(self):
        self.geometry = ScoreGeometry(name="l2_ball", convex=True, union=False, params={"p": 2})

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.norm(prediction - target, p=2, dim=-1)

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy()
        if center.ndim == 0:
            center = center.reshape(1)
        return PredictionRegion.l2_ball(center=center, radius=float(quantile))


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
