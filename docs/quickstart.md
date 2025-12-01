# Quickstart

## Install
```bash
pip install -e .[dev]
```

## Quickstart (split conformal, L2 residual score)
```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from robbuffet import L2Score, SplitConformalCalibrator

# toy predictor
model = torch.nn.Linear(2, 2)

# calibration data loader
torch.manual_seed(0)
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

## Visualization
```python
from robbuffet import vis
import matplotlib.pyplot as plt
vis.plot_region_2d(region, grid_limits=((-1, 1), (-1, 1)), resolution=200)
plt.show()
```

## Affine Robust Solver
For linear/affine dependence on the uncertain parameter `theta`, build a predictor + score and conformal region, then use support functions:
```python
import cvxpy as cp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from robbuffet import L2Score, SplitConformalCalibrator, AffineRobustSolver

# toy predictor
model = torch.nn.Linear(2, 2)
torch.manual_seed(0)
x_cal = torch.randn(200, 2)
y_cal = x_cal + 0.1 * torch.randn_like(x_cal)
cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=32)

cal = SplitConformalCalibrator(model, L2Score(), cal_loader)
q = cal.calibrate(alpha=0.1)
region = cal.predict_region(torch.zeros(1, 2))  # example point

def base_obj(w):
    return cp.norm(w, 2)

def theta_dir(w):
    return w

def robust_constraints(w):
    # Example affine constraint <w, theta> <= 0.5 for all theta in region
    return [(w, 0.5)]

solver = AffineRobustSolver(
    decision_shape=(2,),
    region=region,
    base_objective_fn=base_obj,
    theta_direction_fn=theta_dir,
    constraints_fn=lambda w: [],
    robust_constraints_fn=robust_constraints,
)
w_star, status = solver.solve()
print("status:", status, "w*:", w_star)
```

`AffineRobustSolver` assumes the uncertain parameter enters the problem **affinely**. The robustified optimization has the form:

$\min_{w} \quad g(w) + \sup_{\theta \in \mathcal{C}(x)} \langle d(w), \theta \rangle$  
$\text{s.t. } h_i(w) \le 0, \quad \langle a_j(w), \theta \rangle \le b_j(w) \quad\quad \forall \theta \in \mathcal{C}(x).$

Here:  
- $g(w)$ is `base_objective_fn(w)`; $h_i(w)$ and $b_j(w)$ come from `constraints_fn`.  
- The dependence on $\theta$ is affine: `theta_direction_fn(w)` corresponds to $d(w)$ in the objective, and each pair $(a_j(w), b_j(w))$ comes from `robust_constraints_fn`.  
- $\mathcal{C}(x)$ is the conformal region returned by `cal.predict_region(...)`.  

`AffineRobustSolver` replaces the affine $\theta$ terms with support functions $h_{\mathcal{C}}(\cdot)$; non-affine $\theta$ dependence is **not** supported. Use the Danskin or sampling-based approaches when the uncertainty enters non-affinely or the region is nonconvex/union and you prefer gradient-based optimization.

## Gradient-Based (Danskin) Solver
For nonconvex or union regions, get a conformal region from a predictor/score, then use the Danskin optimizer:
```python
import numpy as np
import torch
from robbuffet import DanskinRobustOptimizer, SplitConformalCalibrator
from robbuffet.scores import GPCPScore
from torch.utils.data import DataLoader, TensorDataset

# toy sampler predictor: returns K samples (K, batch, d)
def sampler(x):
    base = torch.randn(5, x.shape[0], 2)
    return base

score_fn = GPCPScore(sampler)
cal = SplitConformalCalibrator(sampler, score_fn, DataLoader(TensorDataset(torch.zeros(10, 1), torch.zeros(10, 1)), batch_size=2))
q = cal.calibrate(alpha=0.1)
region = cal.predict_region(torch.zeros(1, 1))

def inner(theta_var, w_np):
    return theta_var @ w_np

def value_and_grad(w_np, theta_np):
    return float(theta_np @ w_np), np.array(theta_np, dtype=float)

project = lambda w_vec: np.clip(w_vec, -1, 1)
opt = DanskinRobustOptimizer(region, nom_obj=inner, value_and_grad_fn=value_and_grad, project_fn=project)
w_star, _ = opt.solve(w0=np.zeros(2), step_size=0.1, max_iters=100)
print("Danskin w*:", w_star)
```
