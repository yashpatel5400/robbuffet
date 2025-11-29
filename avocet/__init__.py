"""
Avocet: Conformal prediction and robust decision-making toolkit.

Exposes:
- ScoreFunction implementations and region geometry helpers.
- SplitConformalCalibrator for calibration and region generation.
- Region visualization utilities.
- Scenario-based robust decision-making helpers.
"""

from .scores import L2Score, ScoreFunction
from .calibration import SplitConformalCalibrator
from .region import PredictionRegion, ScoreGeometry
from .decision import ScenarioRobustOptimizer

__all__ = [
    "L2Score",
    "ScoreFunction",
    "SplitConformalCalibrator",
    "PredictionRegion",
    "ScoreGeometry",
    "ScenarioRobustOptimizer",
]
