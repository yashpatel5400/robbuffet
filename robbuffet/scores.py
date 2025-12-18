from __future__ import annotations

import abc
from typing import Any, Callable

import numpy as np
import torch

from .region import (
    EllipsoidRegion,
    L1BallRegion,
    L2BallRegion,
    LinfBallRegion,
    PredictionRegion,
    UnionRegion,
    OperatorNormBallRegion,
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
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
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
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
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
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
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
        diff = prediction
        if diff.dim() == 1:
            diff = diff.unsqueeze(-1)
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        diff = diff - target
        w = torch.as_tensor(self.weight, device=diff.device, dtype=diff.dtype)
        # batch quadratic form
        return torch.sqrt(torch.einsum("bi,ij,bj->b", diff, w, diff))

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy()
        if center.ndim == 0:
            center = center.reshape(1)
        return EllipsoidRegion(center=center, shape_matrix=self.weight, radius=float(quantile))


class OperatorNormScore(ScoreFunction):
    """
    Matrix spectral norm residual score for dynamics matrices C = [A, B].
    Expects flattened matrices of shape (batch, state_dim * (state_dim + control_dim)).
    """

    def __init__(self, state_dim: int, control_dim: int):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.mat_cols = state_dim + control_dim

    @property
    def name(self) -> str:
        return "operator_norm"

    def _reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor.view(tensor.shape[0], self.state_dim, self.mat_cols)

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mat = self._reshape(prediction)
        tgt_mat = self._reshape(target)
        diff = pred_mat - tgt_mat
        return torch.linalg.matrix_norm(diff, ord=2)

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        center = prediction.detach().cpu().numpy().reshape(self.state_dim, self.mat_cols)
        return OperatorNormBallRegion(center=center, radius=float(quantile))


class GPCPScore(ScoreFunction):
    """
    Generalized Probabilistic Conformal Prediction score:
    s(x, y) = min_{k} || sample_k(x) - y ||_2.

    The predictor is assumed to be a callable returning samples of shape (K, batch, d).
    """

    def __init__(self, sampler: Callable[[torch.Tensor], torch.Tensor]):
        """
        sampler: function x -> samples of shape (K, batch, d) or (K, d)
        """
        self.sampler = sampler

    @property
    def name(self) -> str:
        return "gpcp"

    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # prediction is ignored; sampler produces samples given target's x upstream
        samples = prediction  # reuse argument name for compatibility; caller passes samples
        if samples.dim() == 2:
            samples = samples.unsqueeze(-1)  # (K, b, 1)
        if target.dim() == 1:
            target_exp = target.unsqueeze(0)
            if target_exp.dim() < samples.dim():
                target_exp = target_exp.unsqueeze(-1)
        else:
            target_exp = target
        diffs = samples - target_exp
        norms = torch.norm(diffs, dim=-1)
        min_norm, _ = torch.min(norms, dim=0)
        return min_norm

    def build_region(self, prediction: torch.Tensor, quantile: float) -> PredictionRegion:
        centers = prediction.detach().cpu().numpy()
        # Handle batch dimension of size 1 if present
        if centers.ndim == 3 and centers.shape[1] == 1:
            centers = centers[:, 0, :]
        if centers.ndim == 2 and centers.shape[1] == 1:
            centers = centers.reshape(centers.shape[0], 1)
        if centers.ndim == 1:
            centers = centers.reshape(centers.shape[0], 1)
        regions = [L2BallRegion(center=c, radius=float(quantile)) for c in centers]
        if len(regions) == 1:
            return regions[0]
        return UnionRegion(regions)


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
