# <img src="logo.png" alt="Avocet logo" width="64" height="64" style="vertical-align:middle; margin-right:8px;" /> avocet

Conformal prediction + robust decision making with PyTorch predictors and CVXPY optimizers.

**Docs:** https://ypatel.io/avocet/

## Install
- From PyPI (once published):
  ```bash
  pip install avocet
  ```
- From source:
  ```bash
  git clone https://github.com/your-org/avocet
  cd avocet
  pip install .
  ```
- Editable + dev extras:
  ```bash
  pip install -e .[dev]
  ```

## What this package does
- Calibrate PyTorch predictors with split conformal prediction and geometry-aware score functions.
- Produce prediction regions (convex or unions) that can be sampled, visualized (1D/2D), or passed to downstream optimizers.
- Build deterministic or scenario-based robust decision problems that respect conformal regions.

Supported scores/geometries with closed-form robustification:
- L2 residual (`L2Score`) → L2 ball.
- L1 residual (`L1Score`) → L1 ball.
- Linf residual (`LinfScore`) → Linf ball (hypercube).
- Mahalanobis residual (`MahalanobisScore`) → ellipsoid.

## Quickstart (split conformal, L2 residual score)
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from avocet import L2Score, SplitConformalCalibrator

# toy predictor
model = torch.nn.Linear(2, 2)

# calibration data loader
x_cal = torch.randn(200, 2)
y_cal = x_cal + 0.1 * torch.randn_like(x_cal)
cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=32)

cal = SplitConformalCalibrator(model, L2Score(), cal_loader)
alpha = 0.1
cal.calibrate(alpha=alpha)

x_new = torch.randn(1, 2)
region = cal.predict_region(x_new)
print("center:", region.center, "radius:", region.radius)
```

## Scenario-based robust decision making
```python
import cvxpy as cp
import numpy as np
from avocet import ScenarioRobustOptimizer, PredictionRegion

# pretend we already calibrated a 2D L2-ball region
region = PredictionRegion.l2_ball(center=np.array([0.0, 0.0]), radius=0.5)

def objective(w: cp.Variable, theta: np.ndarray):
    # linear loss that depends on uncertainty theta
    return cp.sum_squares(w - theta)

def constraints(w: cp.Variable, theta: np.ndarray):
    return [w >= -1, w <= 1]

optimizer = ScenarioRobustOptimizer(decision_shape=(2,), objective_fn=objective, constraints_fn=constraints, num_samples=256)
problem = optimizer.build_problem(region, solver="ECOS")
print("status:", problem.status, "w*:", problem.variables()[0].value)
```

## Visualization
```python
from avocet import vis
import matplotlib.pyplot as plt
vis.plot_region_2d(region, grid_limits=((-1, 1), (-1, 1)), resolution=200)
plt.show()
```

## Deterministic closed-form robustification (affine in the uncertainty)
For linear/affine dependence on the uncertain parameter `theta`, you can avoid sampling and use support functions:
```python
import cvxpy as cp
import numpy as np
from avocet import PredictionRegion, robustify_affine_leq, robustify_affine_objective

region = PredictionRegion.l2_ball(center=np.array([0.2, -0.1]), radius=0.3)
w = cp.Variable(2)

# Robust constraint: <w, theta> <= 1 for all theta in region
constr = robustify_affine_leq(theta_direction=w, rhs=1.0, region=region)

# Robust objective: minimize ||w||_2 + worst_case(<c, theta>)
c = w  # example direction depending on w
obj = robustify_affine_objective(base_obj=cp.norm(w, 2), theta_direction=c, region=region)

prob = cp.Problem(cp.Minimize(obj), [constr])
prob.solve(solver="ECOS")
print(w.value)
```

## Examples
- `examples/robust_l2.py` — L2 ball robust constraint.
- `examples/robust_l1.py` — L1 ball robust constraint.
- `examples/robust_linf.py` — Linf (hypercube) robust constraint.
- `examples/robust_ellipsoid.py` — ellipsoidal robust constraint.
- `examples/robust_supply_planning.py` — end-to-end: train + calibrate predictor, plot calibration curve, solve robust planning.

Run with `python examples/robust_l2.py` (similar for others).

## Extending
- Add new `ScoreFunction` implementations that expose their induced region geometry via `build_region`.
- For non-convex regions, return `PredictionRegion.union([...])` so optimizers can decompose or sample.
- Use the scenario optimizer as a default inner-approximation; for affine cases use the deterministic robustifiers above.

## Contributing
Please open issues for bugs/feature requests and PRs for fixes/additions. See CONTRIBUTING.md for guidelines.

## Citation
If you use Avocet in academic work, please cite:
```
@software{avocet,
  title = {Avocet: Conformal prediction and robust decision making},
  author = {Yash Patel},
  year = {2025},
  url = {https://github.com/yashpatel5400/avocet}
}
```
