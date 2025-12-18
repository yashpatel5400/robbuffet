from __future__ import annotations

import numpy as np


def simulate_rollout(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    horizon: int,
    x0: np.ndarray,
    process_noise_std: float,
    control_noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
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


def solve_discrete_lqr(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, max_iters: int = 500, tol: float = 1e-8
) -> np.ndarray:
    """
    Iterative discrete-time Riccati to return K (m x n) such that u = -K x.
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


def rollout_cost(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
    x0: np.ndarray | None = None,
    process_noise_std: float = 0.0,
    control_noise_std: float = 0.0,
) -> float:
    """Simulate finite-horizon LQR cost."""
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
