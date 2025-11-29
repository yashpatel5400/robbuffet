"""
End-to-end example:
1) Train a PyTorch predictor on synthetic features -> demand vectors.
2) Calibrate with split conformal (L2 score), produce empirical calibration curve on held-out test data.
3) Form an L2-ball prediction region and solve a deterministic robust planning problem in CVXPY.
"""

import math
from typing import Tuple

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from avocet import (
    L2Score,
    PredictionRegion,
    SplitConformalCalibrator,
    robustify_affine_objective,
)
from avocet.scores import conformal_quantile


def make_synthetic_data(
    n: int = 1000,
    d_x: int = 5,
    d_theta: int = 2,
    noise_scale: float = 0.2,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d_x, generator=rng)
    A = torch.randn(d_x, d_theta, generator=rng)
    theta = x @ A + noise_scale * torch.randn(n, d_theta, generator=rng)
    return x, theta


class MLP(nn.Module):
    def __init__(self, d_x: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_x, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, d_out),
        )

    def forward(self, x):
        return self.net(x)


def train_model(model: nn.Module, loader: DataLoader, lr: float = 1e-2, epochs: int = 20):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()


def calibration_curve(
    model: nn.Module,
    score_fn: L2Score,
    x_cal: torch.Tensor,
    y_cal: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    alphas: np.ndarray,
):
    # collect calibration scores
    with torch.no_grad():
        cal_preds = model(x_cal)
        cal_scores = score_fn.score(cal_preds, y_cal).cpu()
        test_preds = model(x_test)
    coverages = []
    for alpha in alphas:
        q = conformal_quantile(cal_scores, alpha)
        # coverage on test
        residuals = torch.norm(test_preds - y_test, dim=-1).cpu().numpy()
        cov = float(np.mean(residuals <= q))
        coverages.append(cov)
    return np.array(coverages)


def robust_planning(region: PredictionRegion, lam: float = 0.1):
    """
    Solve: min_w lam*||w||_2^2 - min_theta <w, theta> s.t. w >= 0, sum w <= 1
    Robustified deterministically using the support function over the region.
    """
    w = cp.Variable(2)
    base_obj = lam * cp.sum_squares(w)
    obj = robustify_affine_objective(base_obj=base_obj, theta_direction=-w, region=region)
    constraints = [w >= 0, cp.sum(w) <= 1]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver="ECOS")
    return w.value, prob.status


def evaluate_solution(w: np.ndarray, region: PredictionRegion, lam: float = 0.1, n_samples: int = 5000):
    samples = region.sample(n_samples)
    obj_vals = []
    for theta in samples:
        obj_vals.append(lam * np.sum(w**2) - np.dot(w, theta))
    return np.max(obj_vals), np.mean(obj_vals)


def main():
    # data
    x, theta = make_synthetic_data()
    x_train, y_train = x[:700], theta[:700]
    x_cal, y_cal = x[700:900], theta[700:900]
    x_test, y_test = x[900:], theta[900:]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    # train predictor
    model = MLP(d_x=x.shape[1], d_out=theta.shape[1])
    train_model(model, train_loader)

    # calibrate
    score_fn = L2Score()
    cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=64)
    calibrator = SplitConformalCalibrator(model, score_fn, cal_loader)
    alpha = 0.1
    q = calibrator.calibrate(alpha=alpha)
    print(f"Calibrated L2 radius (alpha={alpha}): {q:.4f}")

    # calibration curve on test
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = calibration_curve(model, score_fn, x_cal, y_cal, x_test, y_test, alphas)
    plt.figure()
    plt.plot(alphas, coverages, marker="o", label="empirical coverage")
    plt.plot(alphas, 1 - alphas, "--", label="target 1-alpha")
    plt.xlabel("alpha")
    plt.ylabel("coverage")
    plt.title("Calibration curve on test data")
    plt.legend()
    plt.tight_layout()
    plt.savefig("calibration_curve.png", dpi=150)
    print("Saved calibration_curve.png")

    # pick a new feature vector and region
    x_new = torch.zeros(1, x.shape[1])
    region = calibrator.predict_region(x_new)

    # robust planning vs nominal
    lam = 0.1
    robust_w, status = robust_planning(region, lam=lam)
    print("Robust status:", status, "w*:", robust_w)

    nominal_region = PredictionRegion.l2_ball(center=region.center, radius=0.0)
    nominal_w, _ = robust_planning(nominal_region, lam=lam)
    print("Nominal w*:", nominal_w)

    worst_case, avg_case = evaluate_solution(robust_w, region, lam=lam)
    print(f"Worst-case objective (robust w): {worst_case:.4f}, mean: {avg_case:.4f}")


if __name__ == "__main__":
    main()
