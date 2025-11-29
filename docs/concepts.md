# Conformal Scores and Regions

## Residual-based scores
- L2Score: \(s(x,y) = \|f(x) - y\|_2\) → L2 ball.
- L1Score: \(s(x,y) = \|f(x) - y\|_1\) → L1 ball.
- LinfScore: \(s(x,y) = \|f(x) - y\|_\infty\) → Linf ball.
- MahalanobisScore: \(s(x,y) = \sqrt{(f(x)-y)^\top W (f(x)-y)}\) → ellipsoid.

## GPCPScore (union of balls)
- Score: \(s(x,y) = \min_{k} \| \text{sample}_k(x) - y \|_2\).
- Predictor returns K samples; prediction region becomes a union of L2 balls centered at the samples with radius equal to the conformal quantile.
- Useful with generative models that can sample conditional outputs.

## Calibrators
- SplitConformalCalibrator runs split conformal quantile estimation given a predictor and score.
- For GPCPScore, predictor should return samples (shape K x batch x d); the score builds the union region.

## Robust optimization paths
- Deterministic (affine in uncertainty): use `robustify_affine_objective` / `robustify_affine_leq` with `region.support_function` (unions use max of component supports).
- Scenario-based: `ScenarioRobustOptimizer` samples from regions (works for unions).
- Gradient-based Danskin: `DanskinRobustOptimizer` solves inner maximization per component (works for unions) and updates w via gradients (optionally autograd).

## Region volume
- Regions provide `volume` when closed-form (L2/L1/Linf balls, ellipsoids); unions return None.
- Use `region.volume_mc(bounds)` to estimate volume for unions or complex regions.
