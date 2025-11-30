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

from avocet import SplitConformalCalibrator, vis, L2Score
from avocet.scores import conformal_quantile


def load_siouxfalls():
    net_url = "https://raw.githubusercontent.com/bstabler/TransportationNetworks/master/SiouxFalls/SiouxFalls_net.tntp"
    path = "data/SiouxFalls_net.tntp"
    import pathlib, urllib.request

    pathlib.Path("data").mkdir(exist_ok=True)
    if not pathlib.Path(path).exists():
        urllib.request.urlretrieve(net_url, path)
    edges = []
    travel_time = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("~") or line.strip().startswith("<") or line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            tail, head, cap, t0 = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
            edges.append((tail - 1, head - 1))
            travel_time.append(t0)
    V = max(max(u, v) for u, v in edges) + 1
    E = len(edges)
    A = np.zeros((V, E))
    for idx, (u, v) in enumerate(edges):
        A[u, idx] = 1.0
        A[v, idx] = -1.0
    b = np.zeros(V)
    s, t = 0, V - 1
    b[s] = 1.0
    b[t] = -1.0
    base_cost = np.array(travel_time, dtype=np.float32)
    return edges, A, b, base_cost


class GenerativePredictor(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(-2.0), requires_grad=False)  # fixed small noise

    def forward(self, x, n_samples: int = 1):
        mean = self.net(x).squeeze(-1)
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
            samples, mean = model(xb, n_samples=1)
            pred = mean.squeeze(0)
            if pred.shape != yb.shape:
                yb = yb.view_as(pred)
            loss = mse(pred, yb)
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
    edges, A, b, base_cost = load_siouxfalls()
    d_out = len(edges)

    # Load real contextual data (auto MPG) to forecast a scalar multiplier
    data_path = "data/mpg.csv"
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
    import pathlib, urllib.request, pandas as pd

    pathlib.Path("data").mkdir(exist_ok=True)
    if not pathlib.Path(data_path).exists():
        urllib.request.urlretrieve(url, data_path)
    df = pd.read_csv(data_path).dropna()
    feature_cols = ["horsepower", "weight", "acceleration", "model_year"]
    X = df[feature_cols].values.astype(np.float32)
    # target: mpg scaled; lower mpg -> higher multiplier
    mpg = df["mpg"].values.astype(np.float32)
    mpg_norm = (mpg - mpg.min()) / (mpg.max() - mpg.min())
    y = (1.0 - mpg_norm).astype(np.float32)

    n = len(X)
    rng = np.random.default_rng(0)
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
    X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

    # Standardize features and target
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-6
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-6
    X_train = (X_train - x_mean) / x_std
    X_cal = (X_cal - x_mean) / x_std
    X_test = (X_test - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    y_cal = (y_cal - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std

    x_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    x_cal_t = torch.tensor(X_cal)
    y_cal_t = torch.tensor(y_cal)
    x_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=64, shuffle=True)

    model = GenerativePredictor(d_in=X.shape[1])
    train_model(model, loader, epochs=100, lr=5e-3)

    # GPCP score over sampled multipliers
    K = 1

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        samples, _ = model(xb, n_samples=K)
        return samples

    from avocet.scores import GPCPScore

    score_fn = GPCPScore(sampler)

    def predictor(xb: torch.Tensor) -> torch.Tensor:
        return sampler(xb)

    cal_loader = DataLoader(TensorDataset(x_cal_t, y_cal_t), batch_size=64)
    calibrator = SplitConformalCalibrator(predictor, score_fn, cal_loader)
    alpha = 0.1
    q = calibrator.calibrate(alpha=alpha)
    print(f"Calibrated GPCP radius q: {q:.4f}")

    # Calibration curve
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = []
    with torch.no_grad():
        cal_scores = calibrator.compute_scores(DataLoader(TensorDataset(x_cal_t, y_cal_t), batch_size=32)).numpy()
        test_scores = calibrator.compute_scores(DataLoader(TensorDataset(x_test_t, y_test_t), batch_size=32)).numpy()
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
    x_new = x_test_t[0:1]
    y_true_scalar = float(y_test[0])
    region = calibrator.predict_region(x_new)
    # region is union of balls over sampled multipliers
    centers_mult_scaled = np.array([r.center for r in region.as_union()]).reshape(-1)
    centers_mult = centers_mult_scaled * y_std + y_mean
    centers = np.array([m * base_cost for m in centers_mult])
    radius = q * y_std
    mean_mult = centers_mult.mean()
    mean_pred = mean_mult * base_cost

    # Robust vs nominal
    w_robust, status_r = robust_shortest_path(A, b, centers, radius)
    w_nom, status_n = nominal_shortest_path(A, b, mean_pred)
    print("Robust status:", status_r)
    print("Nominal status:", status_n)

    # Evaluate on true cost
    # For a single test scalar, construct a synthetic true multiplier
    true_mult = y_true_scalar * y_std + y_mean
    true_cost_vec = true_mult * base_cost
    robust_cost = true_cost_vec @ w_robust
    nominal_cost = true_cost_vec @ w_nom
    print(f"True cost (robust): {robust_cost:.4f}")
    print(f"True cost (nominal): {nominal_cost:.4f}")
    print("Edge mapping:", edges)
    print("Robust flow:", w_robust)
    print("Nominal flow:", w_nom)


if __name__ == "__main__":
    main()
