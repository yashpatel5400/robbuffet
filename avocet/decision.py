from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np

from .region import PredictionRegion


ObjectiveFn = Callable[[cp.Variable, np.ndarray], cp.Expression]
ConstraintFn = Callable[[cp.Variable, np.ndarray], Sequence[cp.Constraint]]


class ScenarioRobustOptimizer:
    """
    Scenario-based robust optimization over conformal prediction regions.

    The user provides:
    - decision variable shape,
    - objective_fn(decision_var, theta_sample) -> cp.Expression,
    - constraints_fn(decision_var, theta_sample) -> list of cp.Constraint.

    The optimizer samples the region and minimizes the worst-case objective
    over those samples (epigraph formulation). This yields an inner approximation
    of the true robust counterpart; increase num_samples for tighter results.
    """

    def __init__(
        self,
        decision_shape: Tuple[int, ...],
        objective_fn: ObjectiveFn,
        constraints_fn: Optional[ConstraintFn] = None,
        num_samples: int = 128,
        seed: Optional[int] = None,
    ):
        self.decision_shape = decision_shape
        self.objective_fn = objective_fn
        self.constraints_fn = constraints_fn or (lambda w, theta: [])
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def build_problem(
        self, region: PredictionRegion, solver: Optional[str] = None
    ) -> cp.Problem:
        w = cp.Variable(self.decision_shape)
        t = cp.Variable()
        samples = region.sample(self.num_samples, rng=self.rng)
        constraints: List[cp.Constraint] = []
        for theta in samples:
            constraints.append(t >= self.objective_fn(w, theta))
            constraints.extend(self.constraints_fn(w, theta))
        problem = cp.Problem(cp.Minimize(t), constraints)
        if solver is not None:
            problem.solve(solver=solver)
        return problem
