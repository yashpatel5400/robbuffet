# avocet

Conformal prediction + robust decision making with PyTorch predictors and CVXPY optimizers.

## What this package does
- Calibrate PyTorch predictors with split conformal prediction and geometry-aware score functions.
- Produce prediction regions (convex or unions) that can be sampled, visualized (1D/2D), or passed to downstream optimizers.
- Build scenario-based robust decision problems that respect conformal regions.

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

## Extending
- Add new `ScoreFunction` implementations that expose their induced region geometry via `build_region`.
- For non-convex regions, return `PredictionRegion.union([...])` so optimizers can decompose or sample.
- Use the scenario optimizer as a default inner-approximation; more specialized robust counterparts can be layered on top when closed-form support functions are available.
