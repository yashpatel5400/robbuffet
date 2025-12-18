"""
Robbuffet: Conformal prediction and robust decision-making toolkit.

Exposes:
- ScoreFunction implementations and geometry-aware PredictionRegion classes.
- SplitConformalCalibrator for calibration and region generation.
- Region visualization utilities.
- Scenario-based or deterministic robust decision-making helpers.
"""

from .scores import L1Score, L2Score, LinfScore, MahalanobisScore, OperatorNormScore, ScoreFunction
from .calibration import SplitConformalCalibrator
from .region import (
    PredictionRegion,
    L1BallRegion,
    L2BallRegion,
    LinfBallRegion,
    EllipsoidRegion,
    UnionRegion,
    OperatorNormBallRegion,
)
from .data import BaseDataset, OfflineDataset, SimulationDataset
from .decision import (
    ScenarioRobustOptimizer,
    AffineRobustSolver,
    support_function,
    robustify_affine_objective,
    robustify_affine_leq,
    DanskinRobustOptimizer,
)
__all__ = [
    "L1Score",
    "L2Score",
    "LinfScore",
    "MahalanobisScore",
    "OperatorNormScore",
    "ScoreFunction",
    "SplitConformalCalibrator",
    "PredictionRegion",
    "L1BallRegion",
    "L2BallRegion",
    "LinfBallRegion",
    "EllipsoidRegion",
    "UnionRegion",
    "OperatorNormBallRegion",
    "BaseDataset",
    "OfflineDataset",
    "SimulationDataset",
    "SimulationDataset",
    "ScenarioRobustOptimizer",
    "AffineRobustSolver",
    "DanskinRobustOptimizer",
    "support_function",
    "robustify_affine_objective",
    "robustify_affine_leq",
]
