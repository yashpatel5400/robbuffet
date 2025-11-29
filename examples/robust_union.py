"""
Robust optimization over a union of convex regions.
We model theta lying in the union of two L2 balls. We form a robust constraint
by decomposing over the components (scenario approximation per component) and
taking the max.
"""

import cvxpy as cp
import numpy as np

from avocet import PredictionRegion, ScenarioRobustOptimizer


def main():
    # Union of two L2 balls centered at (+/- 0.5, 0) radius 0.3
    r1 = PredictionRegion.l2_ball(center=np.array([0.5, 0.0]), radius=0.3)
    r2 = PredictionRegion.l2_ball(center=np.array([-0.5, 0.0]), radius=0.3)
    region = PredictionRegion.union([r1, r2])

    def objective(w: cp.Variable, theta: np.ndarray):
        return cp.sum_squares(w - theta)

    def constraints(w: cp.Variable, theta: np.ndarray):
        return [w >= -1, w <= 1]

    opt = ScenarioRobustOptimizer(
        decision_shape=(2,),
        objective_fn=objective,
        constraints_fn=constraints,
        num_samples=200,
        seed=0,
    )
    prob = opt.build_problem(region, solver="ECOS")
    w = prob.variables()[0]
    print("status:", prob.status)
    print("robust w*:", w.value)


if __name__ == "__main__":
    main()
