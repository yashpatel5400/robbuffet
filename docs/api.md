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
- Methods: `sample(n)`, `contains(y)`, `cvxpy_constraints(var)` (convex sets only), `is_convex()`
  - For unions, use support functions (`support_function`) or scenario-based optimization; no single convex constraint is provided.

## Robust optimization helpers
- `region.support_function(direction)`: support of a region (unions take max of component supports).
- `robustify_affine_objective(base_obj, theta_direction, region)`: add worst-case linear term.
- `robustify_affine_leq(theta_direction, rhs, region)`: robust linear inequality.
- `ScenarioRobustOptimizer(decision_shape, objective_fn, constraints_fn=None, num_samples=128, seed=None)`
  - `build_problem(region, solver=None) -> cp.Problem`
- `DanskinRobustOptimizer(region, inner_objective_fn, value_and_grad_fn=None, torch_value_fn=None, project_fn=None, solver="ECOS")`
  - `solve(w0, step_size=..., max_iters=..., tol=..., verbose=False) -> (w*, history)`
  - Either provide `value_and_grad_fn` (returns value, grad_w) or a PyTorch scalar `torch_value_fn(w_tensor, theta_tensor)` for autograd-based gradients.

## Metrics
- Regions expose `volume` (analytic where available) and `volume_mc(bounds, num_samples=...)` for Monte Carlo estimation.
