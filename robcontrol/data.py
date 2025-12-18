"""
Data generation utilities for LQR tasks.

Generates (theta, A_true, B_true, A_hat, B_hat) datasets and saves/loads
compressed NPZ artifacts keyed by task name.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robcontrol.utils import solve_discrete_lqr


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
    n = A_c.shape[0]
    return np.eye(n) + dt * A_c, dt * B_c


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
    """Least squares estimate of (A, B) from trajectories."""
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


def _sample_msd_theta(rng: np.random.Generator) -> Array:
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
    J = rng.uniform(0.01, 0.1)
    b = rng.uniform(0.001, 0.02)
    k_t = rng.uniform(0.05, 0.2)
    return np.array([J, b, k_t], dtype=float)


def _dc_motor_dynamics(theta: Array) -> Tuple[Array, Array]:
    J, b, k_t = theta
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


def generate_dataset(
    task_name: str,
    num_samples: int,
    horizon: int = 200,
    process_noise_std: float = 0.01,
    control_noise_std: float = 0.0,
    seed: int | None = None,
) -> Dict[str, Array]:
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS)}")
    task = TASKS[task_name]
    rng = np.random.default_rng(seed)
    thetas: List[Array] = []
    A_true: List[Array] = []
    B_true: List[Array] = []
    A_hat: List[Array] = []
    B_hat: List[Array] = []

    for _ in range(num_samples):
        theta = task.sample_theta(rng)
        A_c, B_c = task.continuous_dynamics(theta)
        A_d, B_d = _forward_euler_discretize(A_c, B_c, task.dt)
        K = solve_discrete_lqr(A_d, B_d, task.q, task.r)
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
        A_est, B_est = estimate_linear_dynamics(xs, us)
        thetas.append(theta)
        A_true.append(A_d)
        B_true.append(B_d)
        A_hat.append(A_est)
        B_hat.append(B_est)

    return {
        "thetas": np.stack(thetas, axis=0),
        "A_true": np.stack(A_true, axis=0),
        "B_true": np.stack(B_true, axis=0),
        "A_hat": np.stack(A_hat, axis=0),
        "B_hat": np.stack(B_hat, axis=0),
        "q": task.q,
        "r": task.r,
        "dt": task.dt,
        "task": task.name,
    }


def save_dataset(data: Dict[str, Array], path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_dataset(path: str) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate and save an LQR dynamics dataset.")
    parser.add_argument("--task", default="cartpole", choices=list(TASKS.keys()))
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--process-noise-std", type=float, default=0.01)
    parser.add_argument("--control-noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="robcontrol/artifacts/cartpole_dataset.npz")
    args = parser.parse_args()

    data = generate_dataset(
        args.task,
        num_samples=args.num_samples,
        horizon=args.horizon,
        process_noise_std=args.process_noise_std,
        control_noise_std=args.control_noise_std,
        seed=args.seed,
    )
    out_path = Path(args.out)
    if out_path.suffix == "":
        out_path = out_path.with_suffix(".npz")
    save_dataset(data, out_path)
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    main()
