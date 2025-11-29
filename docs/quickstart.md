# Quickstart

## Install
```bash
pip install -e .[dev]
```

## Calibrate a predictor (L2 score)
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from avocet import L2Score, SplitConformalCalibrator

model = torch.nn.Linear(2, 2)
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

## Robust decision making (deterministic, L2 ball)
```python
import cvxpy as cp
import numpy as np
from avocet import PredictionRegion, robustify_affine_leq, robustify_affine_objective

region = PredictionRegion.l2_ball(center=np.array([0.0, 0.0]), radius=0.5)
w = cp.Variable(2)

constr = robustify_affine_leq(theta_direction=w, rhs=1.0, region=region)
obj = robustify_affine_objective(base_obj=cp.norm(w, 2), theta_direction=w, region=region)

prob = cp.Problem(cp.Minimize(obj), [constr])
prob.solve(solver="ECOS")
print("w*:", w.value)
```

## Robust decision making (scenario-based)
```python
import cvxpy as cp
import numpy as np
from avocet import ScenarioRobustOptimizer, PredictionRegion

region = PredictionRegion.l1_ball(center=np.array([0.0, 0.0]), radius=0.5)

def objective(w, theta):
    return cp.sum_squares(w - theta)

def constraints(w, theta):
    return [w >= -1, w <= 1]

opt = ScenarioRobustOptimizer(decision_shape=(2,), objective_fn=objective, constraints_fn=constraints, num_samples=256)
prob = opt.build_problem(region, solver="ECOS")
print("status:", prob.status, "w*:", prob.variables()[0].value)
```

## Visualization (2D region)
```python
import matplotlib.pyplot as plt
from avocet import vis
vis.plot_region_2d(region, grid_limits=((-1, 1), (-1, 1)), resolution=200)
plt.show()
```
