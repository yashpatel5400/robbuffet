"""
Predict-then-optimize example: robust newsvendor on bike rental demand.

Pipeline:
1) Load UCI Bike Sharing day-level data.
2) Train a PyTorch predictor for daily demand (cnt).
3) Calibrate with split conformal (L2 score), plot calibration curve.
4) For test days, build L2-ball regions and solve a robust newsvendor problem
   using the worst-case over interval endpoints; compare to nominal (plug-in) decisions.
5) Compute oracle (true) decision minimizing average cost over the test set, and report
   relative suboptimality of robust/nominal vs oracle.
"""

import os
from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from avocet import L2Score, SplitConformalCalibrator, PredictionRegion, vis
from avocet.scores import conformal_quantile


DATA_PATH = Path("data/day.csv")


def load_bike_data():
    import pandas as pd

    if not DATA_PATH.exists():
        raise FileNotFoundError("data/day.csv not found. Run data download first.")
    df = pd.read_csv(DATA_PATH)
    # features
    feature_cols = ["season", "mnth", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["cnt"].values.astype(np.float32)
    return X, y


class MLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_model(model, loader, epochs=50, lr=1e-3):
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


def calibration_curve(model, score_fn, x_cal, y_cal, x_test, y_test, alphas):
    cal_loader = DataLoader(TensorDataset(x_cal, y_cal), batch_size=64)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=64)
    cal = SplitConformalCalibrator(model, score_fn, cal_loader)
    cal_scores = cal.compute_scores(cal_loader).numpy()
    with torch.no_grad():
        test_scores = cal.compute_scores(test_loader).numpy()
    coverages = []
    quantiles = []
    for alpha in alphas:
        q = conformal_quantile(torch.tensor(cal_scores), alpha)
        quantiles.append(q)
        cov = float(np.mean(test_scores <= q))
        coverages.append(cov)
    return np.array(coverages), np.array(quantiles)


def newsvendor_cost(q, demand, cu=5.0, co=1.0):
    over = np.maximum(q - demand, 0.0)
    under = np.maximum(demand - q, 0.0)
    return co * over + cu * under


def robust_newsvendor(region: PredictionRegion, cu=5.0, co=1.0):
    center = float(np.array(region.center).squeeze())
    r = float(region.radius)
    lb = center - r
    ub = center + r
    q_var = cp.Variable()
    t = cp.Variable()
    cost_lb = co * cp.pos(q_var - lb) + cu * cp.pos(lb - q_var)
    cost_ub = co * cp.pos(q_var - ub) + cu * cp.pos(ub - q_var)
    prob = cp.Problem(cp.Minimize(t), [t >= cost_lb, t >= cost_ub, q_var >= 0])
    prob.solve()
    return float(q_var.value)


def feasibility_newsvendor(q: float) -> str:
    return f"q={q:.2f} (nonneg={q>=0})"


def run_experiment(alpha: float = 0.1, cu: float = 5.0, co: float = 1.0, num_test: int = 100, seed: int = 0):
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data/day.csv. Download Bike Sharing dataset first.")

    X, y = load_bike_data()
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    X = X[idx]
    y = y[idx]
    n_train = int(0.6 * n)
    n_cal = int(0.2 * n)
    X_train, y_train = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train : n_train + n_cal], y[n_train : n_train + n_cal]
    X_test, y_test = X[n_train + n_cal :], y[n_train + n_cal :]

    x_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    x_cal_t = torch.tensor(X_cal)
    y_cal_t = torch.tensor(y_cal)
    x_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=64, shuffle=True)

    model = MLP(d_in=X.shape[1])
    train_model(model, train_loader, epochs=150, lr=2e-3)

    score_fn = L2Score()
    cal_loader = DataLoader(TensorDataset(x_cal_t, y_cal_t), batch_size=64)
    calibrator = SplitConformalCalibrator(model, score_fn, cal_loader)
    q = calibrator.calibrate(alpha=alpha)

    alphas = np.linspace(0.05, 0.5, 10)
    coverages, quantiles = calibration_curve(model, score_fn, x_cal_t, y_cal_t, x_test_t, y_test_t, alphas)
    fig, ax = plt.subplots()
    vis.plot_calibration_curve(alphas, coverages, ax=ax, title="Calibration curve (Bike Sharing)", label="L2")
    ax.legend()
    fig.tight_layout()
    fig.savefig("bike_calibration_curve.png", dpi=150)

    # Evaluate robust vs nominal decisions on a subset of test points
    robust_costs = []
    nominal_costs = []
    qs_robust = []
    qs_nom = []
    selected = min(num_test, len(X_test))
    for i in range(selected):
        x_i = x_test_t[i : i + 1]
        y_true = float(y_test[i])
        region = calibrator.predict_region(x_i)
        q_robust = robust_newsvendor(region, cu=cu, co=co)
        q_nom = float(model(x_i).detach().cpu().numpy().squeeze())
        if i == 0:
            print("Feasibility check (first test point):")
            print(f"  robust : {feasibility_newsvendor(q_robust)}")
            print(f"  nominal: {feasibility_newsvendor(q_nom)}")
        robust_costs.append(newsvendor_cost(q_robust, y_true, cu=cu, co=co))
        nominal_costs.append(newsvendor_cost(q_nom, y_true, cu=cu, co=co))
        qs_robust.append(q_robust)
        qs_nom.append(q_nom)

    # Oracle decision: minimize average cost over test set with true demands known.
    q_star = cp.Variable(nonneg=True)
    y_vec = y_test[:selected]
    over = cp.pos(q_star - y_vec)
    under = cp.pos(y_vec - q_star)
    avg_cost_expr = cp.sum(co * over + cu * under) / selected
    prob = cp.Problem(cp.Minimize(avg_cost_expr))
    prob.solve()
    q_oracle = float(q_star.value)
    oracle_cost = float(avg_cost_expr.value)

    results = {
        "alpha": alpha,
        "q_calibrated": q,
        "avg_cost_robust": float(np.mean(robust_costs)),
        "avg_cost_nominal": float(np.mean(nominal_costs)),
        "avg_cost_oracle": oracle_cost,
        "q_oracle": q_oracle,
        "q_robust_first": qs_robust[0] if qs_robust else None,
        "q_nominal_first": qs_nom[0] if qs_nom else None,
    }

    # Relative suboptimality vs oracle
    if oracle_cost != 0:
        results["rel_robust"] = (results["avg_cost_robust"] - oracle_cost) / oracle_cost
        results["rel_nominal"] = (results["avg_cost_nominal"] - oracle_cost) / oracle_cost
    else:
        results["rel_robust"] = np.inf
        results["rel_nominal"] = np.inf

    print(f"Calibrated L2 radius q (alpha={alpha}): {q:.2f}")
    print("Saved bike_calibration_curve.png")
    print(f"Avg cost (robust): {results['avg_cost_robust']:.2f}")
    print(f"Avg cost (nominal): {results['avg_cost_nominal']:.2f}")
    print(f"Avg cost (oracle): {results['avg_cost_oracle']:.2f}")
    print(f"Relative suboptimality vs oracle - robust: {results['rel_robust']}, nominal: {results['rel_nominal']}")

    plt.figure()
    plt.hist(robust_costs, bins=20, alpha=0.6, label="robust")
    plt.hist(nominal_costs, bins=20, alpha=0.6, label="nominal")
    plt.xlabel("cost")
    plt.ylabel("count")
    plt.title("Robust vs nominal newsvendor costs (Bike Sharing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("bike_cost_hist.png", dpi=150)
    print("Saved bike_cost_hist.png")
    return results


if __name__ == "__main__":
    run_experiment()
