"""
Synthetic LQR dynamics datasets and a starter pipeline to train + conformalize
a dynamics predictor using the operator-norm score.

Each task samples a design vector theta, maps it to continuous-time dynamics
matrices (A_c, B_c), discretizes them, rolls out trajectories under an LQR
controller for that sample, and performs least-squares system ID to recover
estimated (A_hat, B_hat). The resulting tuples (theta, A_true, B_true, A_hat,
B_hat) can be fed to a dynamics predictor and conformal calibrator.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robbuffet import OperatorNormScore, SplitConformalCalibrator


Array = np.ndarray


@dataclass
class LQRTask:
    name: str
    theta_dim: int
    state_dim: int
    control_dim: int
    sample_theta: Callable[[np.random.Generator], Array]
    continuous_dynamics: Callable[[Array], Tuple[Array, Array]]
    q: Array
    r: Array
    dt: float = 0.02


def _forward_euler_discretize(A_c: Array, B_c: Array, dt: float) -> Tuple[Array, Array]:
    """Simple zero-order hold using forward Euler; sufficient for small dt."""
    n = A_c.shape[0]
    return np.eye(n) + dt * A_c, dt * B_c


def _solve_discrete_lqr(
    A: Array, B: Array, Q: Array, R: Array, max_iters: int = 500, tol: float = 1e-8
) -> Array:
    """
    Iterative solution to the discrete-time algebraic Riccati equation using only numpy.
    Returns feedback gain K (m x n) such that u = -K x.
    """
    P = Q.copy()
    for _ in range(max_iters):
        BT_P = B.T @ P
        G = R + BT_P @ B
        K = np.linalg.solve(G, BT_P @ A)
        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
        if np.linalg.norm(P_next - P, ord="fro") < tol:
            P = P_next
            break
        P = P_next
    return K


def simulate_rollout(
    A: Array,
    B: Array,
    K: Array,
    horizon: int,
    x0: Array,
    process_noise_std: float,
    control_noise_std: float,
    rng: np.random.Generator,
) -> Tuple[Array, Array]:
    """Simulate x_{t+1} = A x_t + B u_t + w_t with u_t = -K x_t + v_t."""
    n, m = A.shape[0], B.shape[1]
    xs = np.zeros((horizon + 1, n))
    us = np.zeros((horizon, m))
    xs[0] = x0
    for t in range(horizon):
        control_noise = rng.normal(scale=control_noise_std, size=m)
        u = -K @ xs[t] + control_noise
        us[t] = u
        process_noise = rng.normal(scale=process_noise_std, size=n)
        xs[t + 1] = A @ xs[t] + B @ u + process_noise
    return xs, us


def estimate_linear_dynamics(xs: Array, us: Array, ridge: float = 1e-6) -> Tuple[Array, Array]:
    """
    Least squares estimate of (A, B) from trajectories.
    xs: (T+1, n), us: (T, m). Uses one-step regression x_{t+1} = A x_t + B u_t.
    """
    X = xs[:-1].T  # (n, T)
    X_next = xs[1:].T  # (n, T)
    U = us.T  # (m, T)
    Z = np.vstack([X, U])  # (n+m, T)
    regularizer = ridge * np.eye(Z.shape[0])
    theta_hat = X_next @ Z.T @ np.linalg.pinv(Z @ Z.T + regularizer)
    n = X.shape[0]
    A_hat = theta_hat[:, :n]
    B_hat = theta_hat[:, n:]
    return A_hat, B_hat


def rollout_cost(
    A: Array,
    B: Array,
    K: Array,
    Q: Array,
    R: Array,
    horizon: int,
    rng: np.random.Generator,
    x0: Array | None = None,
    process_noise_std: float = 0.0,
    control_noise_std: float = 0.0,
) -> float:
    """Simulate and return finite-horizon LQR cost."""
    n = A.shape[0]
    if x0 is None:
        x0 = rng.normal(scale=0.1, size=n)
    xs, us = simulate_rollout(
        A,
        B,
        K,
        horizon=horizon,
        x0=x0,
        process_noise_std=process_noise_std,
        control_noise_std=control_noise_std,
        rng=rng,
    )
    cost = 0.0
    for t in range(horizon):
        x = xs[t]
        u = us[t]
        cost += float(x.T @ Q @ x + u.T @ R @ u)
    return cost


# --- Task definitions ----------------------------------------------------- #


def _sample_msd_theta(rng: np.random.Generator) -> Array:
    # Mass m, damping b, spring stiffness k
    m = rng.uniform(0.5, 3.0)
    b = rng.uniform(0.1, 2.0)
    k = rng.uniform(5.0, 50.0)
    return np.array([m, b, k], dtype=float)


def _msd_dynamics(theta: Array) -> Tuple[Array, Array]:
    m, b, k = theta
    A_c = np.array([[0.0, 1.0], [-k / m, -b / m]])
    B_c = np.array([[0.0], [1.0 / m]])
    return A_c, B_c


def _sample_cartpole_theta(rng: np.random.Generator) -> Array:
    # Cart mass M, pole mass m, pole length l, friction c
    M = rng.uniform(0.5, 5.0)
    m = rng.uniform(0.05, 0.5)
    l = rng.uniform(0.2, 1.0)
    c = rng.uniform(0.0, 0.2)
    return np.array([M, m, l, c], dtype=float)


def _cartpole_dynamics(theta: Array) -> Tuple[Array, Array]:
    M, m, l, c = theta
    g = 9.81
    A_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -c / M, (m * g) / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -c / (M * l), (M + m) * g / (M * l), 0.0],
        ]
    )
    B_c = np.array([[0.0], [1.0 / M], [0.0], [1.0 / (M * l)]])
    return A_c, B_c


def _sample_dc_motor_theta(rng: np.random.Generator) -> Array:
    # Inertia J, damping b, torque constant k_t
    J = rng.uniform(0.01, 0.1)
    b = rng.uniform(0.001, 0.02)
    k_t = rng.uniform(0.05, 0.2)
    return np.array([J, b, k_t], dtype=float)


def _dc_motor_dynamics(theta: Array) -> Tuple[Array, Array]:
    J, b, k_t = theta
    # States: [angle, angular velocity], input: voltage scaled -> torque
    A_c = np.array([[0.0, 1.0], [0.0, -b / J]])
    B_c = np.array([[0.0], [k_t / J]])
    return A_c, B_c


TASKS: Dict[str, LQRTask] = {
    "mass_spring_damper": LQRTask(
        name="mass_spring_damper",
        theta_dim=3,
        state_dim=2,
        control_dim=1,
        sample_theta=_sample_msd_theta,
        continuous_dynamics=_msd_dynamics,
        q=np.diag([10.0, 1.0]),
        r=np.array([[0.1]]),
    ),
    "cartpole": LQRTask(
        name="cartpole",
        theta_dim=4,
        state_dim=4,
        control_dim=1,
        sample_theta=_sample_cartpole_theta,
        continuous_dynamics=_cartpole_dynamics,
        q=np.diag([2.0, 0.5, 20.0, 1.0]),
        r=np.array([[0.2]]),
    ),
    "dc_motor": LQRTask(
        name="dc_motor",
        theta_dim=3,
        state_dim=2,
        control_dim=1,
        sample_theta=_sample_dc_motor_theta,
        continuous_dynamics=_dc_motor_dynamics,
        q=np.diag([5.0, 0.5]),
        r=np.array([[0.05]]),
    ),
}


# --- Public dataset builder ---------------------------------------------- #


def generate_lqr_dataset(
    task_name: str,
    num_samples: int,
    horizon: int = 200,
    process_noise_std: float = 0.01,
    control_noise_std: float = 0.0,
    seed: int | None = None,
) -> List[Dict[str, Array]]:
    """
    Build a dataset of (theta, A, B, A_hat, B_hat, xs, us) for a given task.

    Returns a list of dicts; xs and us are included to enable downstream inspection.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS)}")
    task = TASKS[task_name]
    rng = np.random.default_rng(seed)
    records: List[Dict[str, Array]] = []

    for _ in range(num_samples):
        theta = task.sample_theta(rng)
        A_c, B_c = task.continuous_dynamics(theta)
        A_d, B_d = _forward_euler_discretize(A_c, B_c, task.dt)
        K = _solve_discrete_lqr(A_d, B_d, task.q, task.r)

        x0 = rng.normal(scale=0.05, size=task.state_dim)
        xs, us = simulate_rollout(
            A_d,
            B_d,
            K,
            horizon=horizon,
            x0=x0,
            process_noise_std=process_noise_std,
            control_noise_std=control_noise_std,
            rng=rng,
        )
        A_hat, B_hat = estimate_linear_dynamics(xs, us)
        records.append(
            {
                "theta": theta,
                "A_true": A_d,
                "B_true": B_d,
                "A_hat": A_hat,
                "B_hat": B_hat,
                "xs": xs,
                "us": us,
            }
        )
    return records


# --- Pipeline helpers ----------------------------------------------------- #


def _stack_records(records: List[Dict[str, Array]]) -> Tuple[Array, Array]:
    thetas = np.stack([r["theta"] for r in records], axis=0)
    Cs = np.stack([np.hstack([r["A_true"], r["B_true"]]) for r in records], axis=0)
    return thetas, Cs


def _make_loaders(
    thetas: Array, Cs: Array, batch_size: int = 64
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    n = thetas.shape[0]
    idx_train = int(0.6 * n)
    idx_cal = int(0.8 * n)
    theta_t = torch.tensor(thetas, dtype=torch.float32)
    C_flat = torch.tensor(Cs.reshape(n, -1), dtype=torch.float32)
    train_ds = TensorDataset(theta_t[:idx_train], C_flat[:idx_train])
    cal_ds = TensorDataset(theta_t[idx_train:idx_cal], C_flat[idx_train:idx_cal])
    test_ds = TensorDataset(theta_t[idx_cal:], C_flat[idx_cal:])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(cal_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=1, shuffle=False),
    )


class DynamicsMLP(nn.Module):
    def __init__(self, theta_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(theta_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_predictor(model: nn.Module, loader: DataLoader, epochs: int = 300, lr: float = 1e-3, device: str = "cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        for theta, C_flat in loader:
            theta = theta.to(device)
            C_flat = C_flat.to(device)
            pred = model(theta)
            loss = loss_fn(pred, C_flat)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.to("cpu")


def compute_calibration_curve(
    model: nn.Module,
    score_fn: OperatorNormScore,
    cal_loader: Iterable,
    test_loader: Iterable,
    alphas: Array,
) -> Tuple[Array, Array]:
    cal = SplitConformalCalibrator(model, score_fn, cal_loader)
    coverages: List[float] = []
    mat_shape = (score_fn.state_dim, score_fn.state_dim + score_fn.control_dim)
    for alpha in alphas:
        cal.calibrate(alpha=float(alpha))
        hits = 0
        total = 0
        for theta, C_flat in test_loader:
            region = cal.predict_region(theta)
            C_true = C_flat.numpy().reshape(mat_shape)
            if region.contains(C_true):
                hits += 1
            total += 1
        coverages.append(hits / max(total, 1))
    return alphas, np.array(coverages)


def evaluate_controllers(
    model: nn.Module,
    task: LQRTask,
    test_loader: Iterable,
    horizon: int = 200,
    rollouts: int = 5,
    process_noise_std: float = 0.0,
    control_noise_std: float = 0.0,
    seed: int | None = None,
) -> Dict[str, float]:
    """
    For each test theta, synthesize:
      - K_true: optimal for true (A, B) via ARE.
      - K_hat: optimal for predicted (Â,  B̂) from the model.
    Evaluate both on true dynamics and report mean costs.
    """
    rng = np.random.default_rng(seed)
    mat_shape = (task.state_dim, task.state_dim + task.control_dim)
    costs_true = []
    costs_hat_on_true = []
    costs_hat_on_hat = []
    model.eval()
    with torch.no_grad():
        for theta, C_flat in test_loader:
            theta_np = theta.numpy()
            C_true = C_flat.numpy().reshape(mat_shape)
            A_true = C_true[:, : task.state_dim]
            B_true = C_true[:, task.state_dim :]
            # Predict dynamics and synthesize nominal controller
            C_pred = model(theta).detach().numpy().reshape(mat_shape)
            A_hat = C_pred[:, : task.state_dim]
            B_hat = C_pred[:, task.state_dim :]
            K_true = _solve_discrete_lqr(A_true, B_true, task.q, task.r)
            K_hat = _solve_discrete_lqr(A_hat, B_hat, task.q, task.r)
            # Rollout multiple times to mitigate noise variance
            run_costs_true = []
            run_costs_hat_on_true = []
            for _ in range(rollouts):
                x0 = rng.normal(scale=0.1, size=task.state_dim)
                run_costs_true.append(
                    rollout_cost(
                        A_true,
                        B_true,
                        K_true,
                        task.q,
                        task.r,
                        horizon=horizon,
                        rng=rng,
                        x0=x0,
                        process_noise_std=process_noise_std,
                        control_noise_std=control_noise_std,
                    )
                )
                run_costs_hat_on_true.append(
                    rollout_cost(
                        A_true,
                        B_true,
                        K_hat,
                        task.q,
                        task.r,
                        horizon=horizon,
                        rng=rng,
                        x0=x0,
                        process_noise_std=process_noise_std,
                        control_noise_std=control_noise_std,
                    )
                )
            costs_true.append(np.mean(run_costs_true))
            costs_hat_on_true.append(np.mean(run_costs_hat_on_true))
    return {
        "mean_cost_true_opt_on_true": float(np.mean(costs_true)),
        "mean_cost_hat_on_true": float(np.mean(costs_hat_on_true)),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate LQR data, train predictor, and plot calibration curve.")
    parser.add_argument("--task", default="mass_spring_damper", choices=list(TASKS.keys()))
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.1, help="Display quantile for this alpha (calibration curve sweeps a range).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="calibration_curve.png")
    parser.add_argument("--eval", action="store_true", help="Also evaluate controllers (true vs predicted) on test set.")
    args = parser.parse_args()

    records = generate_lqr_dataset(
        args.task,
        num_samples=args.num_samples,
        horizon=args.horizon,
        seed=args.seed,
    )
    thetas, Cs = _stack_records(records)
    task = TASKS[args.task]
    train_loader, cal_loader, test_loader = _make_loaders(thetas, Cs)

    out_dim = Cs.shape[1] * Cs.shape[2]
    model = DynamicsMLP(theta_dim=task.theta_dim, out_dim=out_dim)
    train_predictor(model, train_loader)

    score_fn = OperatorNormScore(state_dim=task.state_dim, control_dim=task.control_dim)
    alphas = np.linspace(0.01, 0.4, 8)
    alphas, coverages = compute_calibration_curve(model, score_fn, cal_loader, test_loader, alphas)

    fig, ax = plt.subplots()
    from robbuffet import vis

    vis.plot_calibration_curve(alphas, coverages, ax=ax, label=args.task, title=f"Calibration ({args.task})")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved calibration curve to {args.out}")

    if args.eval:
        stats = evaluate_controllers(
            model,
            task,
            test_loader,
            horizon=args.horizon,
            rollouts=5,
            process_noise_std=0.0,
            control_noise_std=0.0,
            seed=args.seed + 1,
        )
        print("Controller evaluation (mean finite-horizon cost):")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
