# <img src="logo.png" alt="Avocet logo" width="48" height="64" style="vertical-align:middle; margin-right:8px;" /> robbuffet

[![PyPI](https://img.shields.io/pypi/v/avocet-cp.svg)](https://pypi.org/project/avocet-cp/)
[![Python](https://img.shields.io/pypi/pyversions/avocet-cp.svg)](https://pypi.org/project/avocet-cp/)
[![CI](https://github.com/yashpatel5400/robbuffet/actions/workflows/ci.yml/badge.svg)](https://github.com/yashpatel5400/robbuffet/actions/workflows/ci.yml)
[![Docs](https://github.com/yashpatel5400/robbuffet/actions/workflows/docs.yml/badge.svg)](https://ypatel.io/robbuffet/)

Conformal prediction + robust decision making with PyTorch predictors and CVXPY optimizers.

## Install
- From PyPI (once published):
  ```bash
  pip install avocet-cp
  ```
- From source:
  ```bash
  git clone https://github.com/yashpatel5400/robbuffet
  cd robbuffet
  pip install .
  ```
- Editable + dev extras:
  ```bash
  pip install -e .[dev]
  ```

## Submodules
This repo uses the `DCRNN_PyTorch` submodule for the METR-LA shortest-path example. Clone with:
```bash
git clone --recurse-submodules https://github.com/yashpatel5400/robbuffet
```
or, if already cloned:
```bash
git submodule update --init --recursive
```
For the METR-LA example, generate predictions via:
```bash
cd examples/DCRNN_PyTorch
python run_demo_pytorch.py --config_filename=data/model/pretrained/METR-LA/config.yaml
cd ../..
```
This writes `data/dcrnn_predictions_pytorch.npz` that the example consumes.

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
from robbuffet import L2Score, SplitConformalCalibrator

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
from robbuffet import ScenarioRobustOptimizer, PredictionRegion

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
from robbuffet import vis
import matplotlib.pyplot as plt
vis.plot_region_2d(region, grid_limits=((-1, 1), (-1, 1)), resolution=200)
plt.show()
```

## Deterministic closed-form robustification (affine in the uncertainty)
For linear/affine dependence on the uncertain parameter `theta`, you can avoid sampling and use support functions:
```python
import cvxpy as cp
import numpy as np
from robbuffet import PredictionRegion, robustify_affine_leq, robustify_affine_objective

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
- `examples/robust_shortest_path_metrla.py` — robust shortest path on METR-LA with conformalized DCRNN_PyTorch forecasts (needs `examples/DCRNN_PyTorch` submodule + predictions NPZ).
- `examples/robust_bike_newsvendor.py` — conformal calibration on UCI Bike Sharing demand + robust newsvendor decisions.

Run with `python examples/<script>.py`. The METR-LA script assumes you have generated `examples/DCRNN_PyTorch/data/dcrnn_predictions_pytorch.npz` (see Submodules above).

## Trial runner
Use `scripts/run_trials.py` to run an example multiple times, cache results, and compare robust vs nominal:
```bash
# absolute objectives (no normalization)
python scripts/run_trials.py --example robust_bike_newsvendor.py --trials 5 --alpha 0.1

# relative gaps (requires avg_cost_oracle from the example)
python scripts/run_trials.py --example robust_bike_newsvendor.py --trials 5 --alpha 0.1 --relative
```
Outputs mean/std and a paired t-test (robust < nominal) when scipy is available. Caches results in `.cache/run_trials.json`.

## Extending
- Add new `ScoreFunction` implementations that expose their induced region geometry via `build_region`.
- For non-convex regions, return `PredictionRegion.union([...])` so optimizers can decompose or sample.
- Use the scenario optimizer as a default inner-approximation; for affine cases use the deterministic robustifiers above.

## Contributing
Please open issues for bugs/feature requests and PRs for fixes/additions. See CONTRIBUTING.md for guidelines.

## Citation
If you use Avocet in academic work, please cite:
```
@software{robbuffet,
  title = {Robbuffet: Conformal prediction and robust decision making},
  author = {Yash Patel},
  year = {2025},
  url = {https://github.com/yashpatel5400/robbuffet}
}
```
