# API Overview

## Calibration
- `SplitConformalCalibrator(predictor, score_fn, calibration_data, device=None)`
  - `calibrate(alpha: float) -> float`: fit quantile.
  - `predict_region(x: torch.Tensor) -> PredictionRegion`

## Scores
- `L2Score`, `L1Score`, `LinfScore`, `MahalanobisScore(weight)`
  - `score(prediction, target) -> torch.Tensor`
  - `build_region(prediction, quantile) -> PredictionRegion`

## Regions
- Factory methods: `PredictionRegion.l2_ball`, `.l1_ball`, `.linf_ball`, `.ellipsoid`, `.union([...])`
- Methods: `sample(n)`, `contains(y)`, `cvxpy_constraints(var)` (convex only), `is_convex()`

## Robust optimization helpers
- `support_function(region, direction)`: support of a convex region (union over-approximated by max of components).
- `robustify_affine_objective(base_obj, theta_direction, region)`: add worst-case linear term.
- `robustify_affine_leq(theta_direction, rhs, region)`: robust linear inequality.
- `ScenarioRobustOptimizer(decision_shape, objective_fn, constraints_fn=None, num_samples=128, seed=None)`
  - `build_problem(region, solver=None) -> cp.Problem`
