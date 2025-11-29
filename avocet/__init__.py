"""
Avocet: Conformal prediction and robust decision-making toolkit.

Exposes:
- ScoreFunction implementations and geometry-aware PredictionRegion classes.
- SplitConformalCalibrator for calibration and region generation.
- Region visualization utilities.
- Scenario-based or deterministic robust decision-making helpers.
"""

from .scores import L1Score, L2Score, LinfScore, MahalanobisScore, ScoreFunction
from .calibration import SplitConformalCalibrator
from .region import (
    PredictionRegion,
    L1BallRegion,
    L2BallRegion,
    LinfBallRegion,
    EllipsoidRegion,
    UnionRegion,
)
from .decision import (
    ScenarioRobustOptimizer,
    support_function,
    robustify_affine_objective,
    robustify_affine_leq,
)

__all__ = [
    "L1Score",
    "L2Score",
    "LinfScore",
    "MahalanobisScore",
    "ScoreFunction",
    "SplitConformalCalibrator",
    "PredictionRegion",
    "L1BallRegion",
    "L2BallRegion",
    "LinfBallRegion",
    "EllipsoidRegion",
    "UnionRegion",
    "ScenarioRobustOptimizer",
    "support_function",
    "robustify_affine_objective",
    "robustify_affine_leq",
]
