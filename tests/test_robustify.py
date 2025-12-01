import cvxpy as cp
import numpy as np

from robbuffet import robustify_affine_leq, support_function, AffineRobustSolver
from robbuffet.region import L2BallRegion, LinfBallRegion


def test_support_function_l2():
    region = L2BallRegion(center=np.array([0.0, 0.0]), radius=1.0)
    direction = np.array([1.0, 2.0])
    val = support_function(region, direction)
    # For L2 ball, h(d) = <d, c> + r * ||d||_2 = 0 + 1 * sqrt(5)
    assert np.isclose(val.value, np.sqrt(5))


def test_robust_constraint_linf():
    region = LinfBallRegion(center=np.array([0.0, 0.0]), radius=0.5)
    w = cp.Variable(2)

    def base_obj(w_var):
        return cp.norm(w_var, 2)

    solver = AffineRobustSolver(
        decision_shape=(2,),
        region=region,
        base_objective_fn=base_obj,
        theta_direction_fn=lambda w_var: w_var,
        constraints_fn=lambda w_var: [],
    )
    w_star, status = solver.solve()
    assert status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert np.all(np.abs(w_star) <= 1.0)  # robust constraint implies |w_i| <= 1
