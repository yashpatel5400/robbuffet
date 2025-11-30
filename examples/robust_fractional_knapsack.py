"""
Robust fractional knapsack using SBIBM's two_moons simulator and a Neural Spline Flow
posterior estimator to generate samples for GPCP.
"""

import argparse
import os
import numpy as np
import torch
from torch import nn
from functools import partial
from torch.utils.data import DataLoader, TensorDataset

# Guard for environments where numpy might lack np.dtypes
if not hasattr(np, "dtypes"):
    np.dtypes = np.dtype  # type: ignore

from robbuffet import SplitConformalCalibrator
from robbuffet.scores import GPCPScore, conformal_quantile

try:
    import sbibm
except ImportError as e:  # pragma: no cover - optional dependency
    sbibm = None
    _sbibm_import_error = e
else:
    _sbibm_import_error = None
try:
    from pyknos.nflows import distributions as distributions_
    from pyknos.nflows import flows, transforms
    from pyknos.nflows.nn import nets
    from sbi.utils.sbiutils import standardizing_net, standardizing_transform, z_score_parser
    from sbi.utils.torchutils import create_alternating_binary_mask
    from torch import relu, tensor, uint8
except ImportError as e:  # pragma: no cover
    distributions_ = flows = transforms = nets = None
    _nflows_import_error = e
else:
    _nflows_import_error = None


class ContextSplineMap(nn.Module):
    """Conditioner for 1D spline when using context as input (from SBI utils)."""

    def __init__(self, in_features: int, out_features: int, hidden_features: int, context_features: int, hidden_layers: int):
        super().__init__()
        self.hidden_features = hidden_features
        layers = [nn.Linear(context_features, hidden_features), nn.ReLU()]
        layers += [nn.Linear(hidden_features, hidden_features), nn.ReLU()] * hidden_layers
        layers += [nn.Linear(hidden_features, out_features)]
        self.spline_predictor = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, context: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.spline_predictor(context)


def build_nsf(
    batch_theta: torch.Tensor,
    batch_x: torch.Tensor,
    z_score_theta: str = "independent",
    z_score_x: str = "independent",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
) -> nn.Module:
    """Build Neural Spline Flow q(theta | x)."""
    theta_numel = batch_theta[0].numel()
    x_numel = batch_x[0].numel()

    # embedding for context x
    embedding_net = nn.Identity()
    y_dim = embedding_net(batch_x[:1]).numel() if hasattr(embedding_net, "__call__") else x_numel

    def mask_in_layer(i):
        return create_alternating_binary_mask(features=theta_numel, even=(i % 2 == 0))

    if theta_numel == 1:
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_dim,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_dim,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i) if theta_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        if theta_numel > 1:
            block.append(transforms.LULinear(theta_numel, identity_init=True))
        transform_list += block

    z_theta, structured_theta = z_score_parser(z_score_theta)
    if z_theta:
        transform_list = [standardizing_transform(batch_theta, structured_theta)] + transform_list

    z_x, structured_x = z_score_parser(z_score_x)
    if z_x:
        embedding_net = nn.Sequential(standardizing_net(batch_x, structured_x), embedding_net)

    base = distributions_.StandardNormal((theta_numel,))
    transform = transforms.CompositeTransform(transform_list)
    flow = flows.Flow(transform, base, embedding_net)
    return flow


def solve_nominal(values, weights, capacity):
    # Fractional knapsack nominal: take ratio ordering.
    ratio = values / weights
    order = np.argsort(ratio)[::-1]
    remaining = capacity
    x = np.zeros_like(values)
    for idx in order:
        take = min(1.0, remaining / weights[idx])
        x[idx] = take
        remaining -= take * weights[idx]
        if remaining <= 1e-6:
            break
    return x, values @ x


def solve_robust(values_centers, radius, weights, capacity):
    # Heuristic robust greedy: use worst-case per-item value proxy (min center minus radius)
    worst_vals = values_centers.min(axis=0) - radius
    ratio = worst_vals / weights
    order = np.argsort(ratio)[::-1]
    remaining = capacity
    x = np.zeros_like(weights)
    for idx in order:
        take = min(1.0, remaining / weights[idx])
        x[idx] = take
        remaining -= take * weights[idx]
        if remaining <= 1e-6:
            break
    worst_case_value = values_centers @ x
    worst_case = worst_case_value.min() - radius * np.linalg.norm(x)
    return x, worst_case


def run_experiment(alpha=0.1, K=8, n_items=10, capacity=5.0, seed=0):
    if sbibm is None:
        raise ImportError("sbibm is required for this example. Install with `pip install sbibm`.") from _sbibm_import_error
    if flows is None:
        raise ImportError("pyknos/nflows is required. Install with `pip install pyknos sbi`.") from _nflows_import_error

    task = sbibm.get_task("two_moons")
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Generate data
    train_theta = prior(num_samples=800)
    train_x = simulator(train_theta)
    cal_theta = prior(num_samples=200)
    cal_x = simulator(cal_theta)
    test_theta = prior(num_samples=200)
    test_x = simulator(test_theta)

    cache_path = "two_moons_flow.pt"
    if os.path.exists(cache_path):
        flow = torch.load(cache_path, map_location="cpu")
    else:
        flow = build_nsf(train_theta, train_x, z_score_theta="independent", z_score_x="independent")
        opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
        flow.train()
        train_ds = TensorDataset(train_x, train_theta)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        for epoch in range(200):
            for xb, yb in train_loader:
                opt.zero_grad()
                loss = -flow.log_prob(inputs=yb, context=xb).mean()
                loss.backward()
                opt.step()
        torch.save(flow, cache_path)

    cal_ds = TensorDataset(cal_x, cal_theta)
    cal_loader = DataLoader(cal_ds, batch_size=64, shuffle=False)
    test_ds = TensorDataset(test_x, test_theta)

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            samples = flow.sample(num_samples=K, context=xb)
        return samples  # (K, batch, theta_dim)

    score_fn = GPCPScore(sampler)
    calibrator = SplitConformalCalibrator(sampler, score_fn, cal_loader)
    q = calibrator.calibrate(alpha=alpha)

    # Calibration curve (optional)
    cal_scores = calibrator.compute_scores(cal_loader).numpy()
    test_scores = calibrator.compute_scores(DataLoader(test_ds, batch_size=64)).numpy()
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = [float(np.mean(test_scores <= conformal_quantile(torch.tensor(cal_scores), a))) for a in alphas]

    # Evaluate on first test point
    x_test, y_test = test_ds.tensors
    x0 = x_test[0:1]
    y0 = y_test[0].numpy()
    values_true = y0[:n_items]
    weights_true = np.abs(values_true) + 0.5

    region = calibrator.predict_region(x0)
    centers = np.stack([r.center for r in region.as_union()])
    radius = q

    mean_pred = centers.mean(axis=0)
    values_pred = mean_pred[:n_items]
    weights_pred = np.abs(values_pred) + 0.5

    # Robust and nominal solutions
    x_nom, nominal_obj_pred = solve_nominal(values_pred, weights_pred, capacity)
    x_rob, robust_obj_pred = solve_robust(centers[:, :n_items], radius, weights_pred, capacity)

    true_nominal_obj = float(values_true @ x_nom)
    true_robust_obj = float(values_true @ x_rob)

    print("Fractional knapsack results (single test instance):")
    print(f"  Nominal objective (true values): {true_nominal_obj:.4f}")
    print(f"  Robust objective  (true values): {true_robust_obj:.4f}")
    print(f"  Calibrated radius q: {q:.4f}")

    print("\nTable:")
    print(f"{'method':<12}{'true_obj':>12}")
    print(f"{'nominal':<12}{true_nominal_obj:>12.4f}")
    print(f"{'robust':<12}{true_robust_obj:>12.4f}")

    return {
        "avg_cost_nominal": -true_nominal_obj,  # negative so lower is better for t-test
        "avg_cost_robust": -true_robust_obj,
        "avg_cost_oracle": None,
        "q_calibrated": q,
        "coverage_alphas": alphas,
        "coverage": coverages,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--n-items", type=int, default=10)
    parser.add_argument("--capacity", type=float, default=5.0)
    args = parser.parse_args()
    run_experiment(alpha=args.alpha, K=args.K, n_items=args.n_items, capacity=args.capacity)
