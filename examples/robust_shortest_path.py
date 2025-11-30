"""
End-to-end conformal predict-then-optimize for a shortest-path/flow problem.

We:
1) Train a simple generative predictor to map features -> edge cost vector (mean + Gaussian noise).
2) Calibrate with a GPCP-style score: s(x, y) = min_k ||y - sample_k(x)||_2.
3) Build a union-of-balls prediction region for a new context.
4) Solve a robust shortest-path LP with the union region by enforcing worst-case cost across components.
"""

import numpy as np
import cvxpy as cp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from avocet import SplitConformalCalibrator, vis
from avocet.scores import GPCPScore, conformal_quantile


def make_graph():
    # Small directed graph
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (4, 3),
    ]
    V = 5
    E = len(edges)
    A = np.zeros((V, E))
    for idx, (u, v) in enumerate(edges):
        A[u, idx] = 1.0
        A[v, idx] = -1.0
    b = np.zeros(V)
    s, t = 0, 3
    b[s] = 1.0
    b[t] = -1.0
    return edges, A, b


class GenerativePredictor(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, d_out),
        )
        self.log_std = nn.Parameter(torch.zeros(d_out))

    def forward(self, x, n_samples: int = 1):
        mean = self.net(x)
        std = torch.exp(self.log_std)
        eps = torch.randn(n_samples, *mean.shape, device=mean.device)
        samples = mean.unsqueeze(0) + eps * std
        return samples, mean


def train_model(model, loader, epochs=50, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            _, mean = model(xb, n_samples=1)
            loss = mse(mean.squeeze(0), yb)
            loss.backward()
            opt.step()


def robust_shortest_path(A, b, centers, radius):
    E = A.shape[1]
    w = cp.Variable(E)
    t = cp.Variable()
    constraints = [A @ w == b, w >= 0, w <= 1]
    for c in centers:
        constraints.append(t >= c @ w + radius * cp.norm(w, 2))
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve(solver="ECOS")
    return w.value, prob.status


def nominal_shortest_path(A, b, mean_cost):
    E = A.shape[1]
    w = cp.Variable(E)
    prob = cp.Problem(cp.Minimize(mean_cost @ w), [A @ w == b, w >= 0, w <= 1])
    prob.solve(solver="ECOS")
    return w.value, prob.status


def main():
    # Data
    edges, A, b = make_graph()
    d_out = len(edges)
    d_in = 6
    n_train, n_cal, n_test = 200, 100, 100
    rng = np.random.default_rng(0)
    x_all = torch.tensor(rng.normal(size=(n_train + n_cal + n_test, d_in)), dtype=torch.float32)
    true_W = torch.tensor(rng.normal(size=(d_in, d_out)), dtype=torch.float32)
    base_costs = x_all @ true_W + 0.1 * torch.randn(n_train + n_cal + n_test, d_out)

    x_train, y_train = x_all[:n_train], base_costs[:n_train]
    x_cal, y_cal = x_all[n_train : n_train + n_cal], base_costs[n_train : n_train + n_cal]
    x_test, y_test = x_all[n_train + n_cal :], base_costs[n_train + n_cal :]

    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    model = GenerativePredictor(d_in=d_in, d_out=d_out)
    train_model(model, loader, epochs=200, lr=5e-3)

    # Calibration using GPCPScore + SplitConformalCalibrator
    K = 8

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        samples, _ = model(xb, n_samples=K)
        return samples

    score_fn = GPCPScore(sampler)

    # For GPCPScore, predictor should return samples directly
    def predictor(xb: torch.Tensor) -> torch.Tensor:
        return sampler(xb)

    cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=32)
    calibrator = SplitConformalCalibrator(predictor, score_fn, cal_loader)
    alpha = 0.1
    q = calibrator.calibrate(alpha=alpha)
    print(f"Calibrated GPCP radius q: {q:.4f}")

    # Calibration curve
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = []
    with torch.no_grad():
        cal_scores = []
        for xb, yb in DataLoader(TensorDataset(x_cal, y_cal), batch_size=32):
            cal_scores.append(score_fn.score(predictor(xb), yb).cpu())
        cal_scores = torch.cat(cal_scores).numpy()
        test_scores = []
        for xb, yb in DataLoader(TensorDataset(x_test, y_test), batch_size=32):
            preds = predictor(xb)
            diffs = preds - yb.unsqueeze(0)
            norms = torch.norm(diffs, dim=-1).min(dim=0)[0].cpu().numpy()
            test_scores.append(norms)
        test_scores = np.concatenate(test_scores)
    for a in alphas:
        q_a = conformal_quantile(torch.tensor(cal_scores), a)
        cov = float(np.mean(test_scores <= q_a))
        coverages.append(cov)
    vis.plot_calibration_curve(alphas, coverages, title="Calibration curve (shortest path)")
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig("shortest_path_calibration_curve.png", dpi=150)
    print("Saved shortest_path_calibration_curve.png")

    # Pick a test point
    x_new = x_test[0:1]
    y_true = y_test[0].numpy()
    region = calibrator.predict_region(x_new)
    centers = np.stack([r.center for r in region.as_union()]) if hasattr(region, "as_union") else np.array([region.center])
    mean_pred = centers.mean(axis=0)
    radius = q

    # Robust vs nominal
    w_robust, status_r = robust_shortest_path(A, b, centers, radius)
    w_nom, status_n = nominal_shortest_path(A, b, mean_pred)
    print("Robust status:", status_r)
    print("Nominal status:", status_n)

    # Evaluate on true cost
    robust_cost = y_true @ w_robust
    nominal_cost = y_true @ w_nom
    print(f"True cost (robust): {robust_cost:.4f}")
    print(f"True cost (nominal): {nominal_cost:.4f}")
    print("Edge mapping:", edges)
    print("Robust flow:", w_robust)
    print("Nominal flow:", w_nom)


if __name__ == "__main__":
    main()
