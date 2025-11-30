"""
End-to-end example:
1) Train a PyTorch predictor on synthetic features -> demand vectors.
2) Calibrate with split conformal (L2 score), produce empirical calibration curve on held-out test data.
3) Form an L2-ball prediction region and solve a deterministic robust planning problem in CVXPY.
"""

import math
from typing import Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from avocet import (
    L2Score,
    L1Score,
    LinfScore,
    PredictionRegion,
    SplitConformalCalibrator,
    robustify_affine_objective,
)
from avocet.scores import conformal_quantile
from avocet import vis


def load_bike_demand(num_products: int = 10):
    df = pd.read_csv("data/day.csv")
    feature_cols = ["season", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
    x = df[feature_cols].values.astype(np.float32)
    rng = np.random.default_rng(0)
    # generate num_products demands as positive random mixtures of rider counts
    base = df[["casual", "registered"]].values.astype(np.float32) / 100.0  # keep magnitudes reasonable
    W = np.abs(rng.normal(size=(base.shape[1], num_products)).astype(np.float32)) + 0.1
    y = base @ W  # shape (n, num_products)
    y = y + 0.5  # ensure positivity
    return torch.tensor(x), torch.tensor(y)


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
    score_fn,
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
        test_scores = score_fn.score(test_preds, y_test).cpu()
    coverages = []
    for alpha in alphas:
        q = conformal_quantile(cal_scores, alpha)
        cov = float(np.mean(test_scores.numpy() <= q))
        coverages.append(cov)
    return np.array(coverages)


def robust_planning(region: PredictionRegion, lam: float = 0.1):
    """
    Solve: min_w lam*||w||_2^2 - min_theta <w, theta> s.t. w >= 0, sum w <= 1
    Robustified deterministically using the support function over the region.
    """
    d = region.center.shape[-1]
    w = cp.Variable(d)
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
    num_products = 10
    x, theta = load_bike_demand(num_products=num_products)
    n = len(x)
    idx = torch.randperm(n)
    x = x[idx]
    theta = theta[idx]
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    x_train, y_train = x[:n_train], theta[:n_train]
    x_cal, y_cal = x[n_train : n_train + n_cal], theta[n_train : n_train + n_cal]
    x_test, y_test = x[n_train + n_cal :], theta[n_train + n_cal :]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)

    # train predictor
    model = MLP(d_x=x.shape[1], d_out=theta.shape[1])
    train_model(model, train_loader)

    # evaluate multiple scores
    scores = [("L2", L2Score()), ("L1", L1Score()), ("Linf", LinfScore())]
    alpha = 0.1
    cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=64)

    # pick a realistic feature vector and region (use a held-out test point)
    x_new = x_test[0:1]
    y_true = y_test[0].numpy()
    nominal_pred = model(x_new).detach().cpu().numpy().squeeze()

    rows = []
    for name, score_fn in scores:
        calibrator = SplitConformalCalibrator(model, score_fn, cal_loader)
        q = calibrator.calibrate(alpha=alpha)
        region = calibrator.predict_region(x_new)
        pred_center = region.center

        lam = 0.1
        robust_w, status = robust_planning(region, lam=lam)
        nominal_region = PredictionRegion.l2_ball(center=region.center, radius=0.0) if name == "L2" else region
        nominal_w, _ = robust_planning(nominal_region, lam=lam)

        worst_case, avg_case = evaluate_solution(robust_w, region, lam=lam)
        nominal_obj = lam * np.sum(nominal_w**2) - np.dot(nominal_w, region.center)
        robust_obj = lam * np.sum(robust_w**2) - np.dot(robust_w, region.center)
        true_obj_nominal = lam * np.sum(nominal_w**2) - np.dot(nominal_w, y_true)
        true_obj_robust = lam * np.sum(robust_w**2) - np.dot(robust_w, y_true)

        rows.append(
            {
                "score": name,
                "q": q,
                "worst_case": worst_case,
                "mean_case": avg_case,
                "nominal_obj_center": nominal_obj,
                "robust_obj_center": robust_obj,
                "true_obj_nominal": true_obj_nominal,
                "true_obj_robust": true_obj_robust,
                "||w_nom||1": float(np.sum(np.abs(nominal_w))),
                "||w_rob||1": float(np.sum(np.abs(robust_w))),
            }
        )

    # print table
    headers = ["score", "q", "true_obj_nominal", "true_obj_robust"]
    widths = [10, 8, 20, 20]
    def fmt_row(values):
        return "".join(str(v).ljust(w) for v, w in zip(values, widths))

    print(fmt_row(headers))
    for r in rows:
        print(
            fmt_row(
                [
                    r["score"],
                    f"{r['q']:.3f}",
                    f"{r['true_obj_nominal']:.3f}",
                    f"{r['true_obj_robust']:.3f}",
                ]
            )
        )

    # calibration curves for each score (shared target coverage)
    alphas = np.linspace(0.05, 0.5, num=10)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for name, score_fn in scores:
        cov = calibration_curve(model, score_fn, x_cal, y_cal, x_test, y_test, alphas)
        vis.plot_calibration_curve(alphas, cov, title="Calibration curve (supply planning)", ax=ax, label=name)
    ax.legend()
    fig.tight_layout()
    fig.savefig("calibration_curve.png", dpi=150)
    print("Saved calibration_curve.png")


if __name__ == "__main__":
    main()
