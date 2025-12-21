# Robbuffet

Conformal prediction + robust decision making with PyTorch predictors and CVXPY optimizers.

**What it does**
- Split conformal calibration with geometry-aware scores (L2, L1, Linf, Mahalanobis).
- Prediction regions (balls, ellipsoids, unions) that can be sampled, visualized, and used in downstream optimization.
- Deterministic robust counterparts for affine uncertainty, scenario-based robust optimization, and gradient-based (Danskin) solvers for unions.

**Install**
- `pip install robbuffet` (Python 3.10+). For development: `pip install -e .[dev]`.

**Getting started**
- Read the [Quickstart](quickstart.md).
- Browse runnable [Examples](examples.md).
- See [API](api.md) for core classes.
