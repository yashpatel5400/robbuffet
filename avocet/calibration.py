from __future__ import annotations

from typing import Iterable, Optional, Protocol, Tuple

import torch

from .region import PredictionRegion
from .scores import ScoreFunction, conformal_quantile


class Predictor(Protocol):
    """Minimal predictor protocol."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


class SplitConformalCalibrator:
    """
    Split conformal calibration for PyTorch predictors.

    Usage:
        cal = SplitConformalCalibrator(predictor, score_fn, calibration_loader)
        cal.calibrate(alpha=0.1)
        region = cal.predict_region(x_new)
    """

    def __init__(
        self,
        predictor: Predictor,
        score_fn: ScoreFunction,
        calibration_data: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        device: Optional[torch.device] = None,
    ):
        self.predictor = predictor
        self.score_fn = score_fn
        self.calibration_data = calibration_data
        self.device = device or torch.device("cpu")
        self._quantile: Optional[float] = None

    @torch.no_grad()
    def compute_scores(self, data: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        scores = []
        for x, y in data:
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self.predictor(x)
            batch_scores = self.score_fn.score(preds, y)
            bs = batch_scores.detach().cpu()
            if bs.dim() == 0:
                bs = bs.unsqueeze(0)
            scores.append(bs)
        if not scores:
            raise ValueError("No data provided for score computation.")
        return torch.cat(scores).float()

    @torch.no_grad()
    def calibrate(self, alpha: float) -> float:
        all_scores = self.compute_scores(self.calibration_data)
        self._quantile = conformal_quantile(all_scores, alpha)
        return self._quantile

    @torch.no_grad()
    def predict_region(self, x: torch.Tensor) -> PredictionRegion:
        if self._quantile is None:
            raise RuntimeError("Calibrator not fitted. Call calibrate(alpha) first.")
        x = x.to(self.device)
        pred = self.predictor(x)
        # For GPCP-style scores, predictor may return samples directly
        # Ensure batch dimension is handled by squeezing if needed
        if isinstance(pred, torch.Tensor) and pred.dim() >= 2 and pred.shape[0] == 1:
            pred = pred.squeeze(0)
        return self.score_fn.build_region(pred, self._quantile)
