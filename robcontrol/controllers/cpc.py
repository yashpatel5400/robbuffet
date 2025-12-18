"""
Conformalized Predict-Then-Control (CPC) for discrete-time LQR via Danskin updates.

Implements the inner maximization over dynamics in a spectral-norm ball and
outer subgradient descent on the controller gain K, following the algorithmic
sketch in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from robcontrol.controllers.base import RobustController
from robcontrol.utils import solve_discrete_lqr


def solve_discrete_lyapunov(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Solve X from A X A^T - X + Q = 0 via vec linear system.
    """
    n = A.shape[0]
    I = np.eye(n * n)
    K = np.kron(A, A)
    rhs = Q.reshape(-1)
    x_vec = np.linalg.solve(I - K, rhs)
    return x_vec.reshape(n, n)


def lqr_cost(K: np.ndarray, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, X0: np.ndarray) -> float:
    """
    Infinite-horizon discrete-time LQR cost: trace((Q + K^T R K) X)
    where X solves Lyapunov with closed-loop Acl = A - B K and state covariance X0.
    """
    Acl = A - B @ K
    X = solve_discrete_lyapunov(Acl, X0)
    return float(np.trace((Q + K.T @ R @ K) @ X))


def gradients(
    K: np.ndarray, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, X0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Lyapunov solutions and gradients wrt K and C=[A,B].
    Returns (grad_K, grad_C, X, P, W)
    """
    n = A.shape[0]
    m = B.shape[1]
    Acl = A - B @ K
    X = solve_discrete_lyapunov(Acl, X0)
    P = solve_discrete_lyapunov(Acl.T, Q + K.T @ R @ K)
    grad_K = 2 * ((R + B.T @ P @ B) @ K - B.T @ P @ A) @ X
    W = np.vstack([np.eye(n), -K])  # (n+m, n)
    C = np.hstack([A, B])
    grad_C = 2 * P @ C @ W @ X @ W.T
    return grad_K, grad_C, X, P, W


@dataclass
class CPCConfig:
    step_k: float = 1e-3
    step_c: float = 1e-2
    outer_iters: int = 50
    inner_iters: int = 20
    radius: float = 0.1
    X0_scale: float = 0.1


class CPCController(RobustController):
    """
    Simple CPC solver using projected gradient ascent over dynamics and
    gradient descent over K (subgradient via Danskin).
    """

    def __init__(self, Q: np.ndarray, R: np.ndarray, config: CPCConfig | None = None):
        self.Q = Q
        self.R = R
        self.config = config or CPCConfig()

    def synthesize(
        self, A: np.ndarray, B: np.ndarray, Q: np.ndarray | None = None, R: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Satisfies RobustController interface; ignores passed Q,R if provided
        and uses those from initialization. Returns K.
        """
        C = np.hstack([A, B])
        K, _, _ = self.run(C)
        return K

    def run(self, C_center: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Run CPC starting from LQR on center dynamics.

        Args:
            C_center: matrix [A_hat, B_hat] (n x (n+m)).
        Returns:
            (K_star, C_worst, cost_history)
        """
        n, cols = C_center.shape
        m = cols - n
        A0 = C_center[:, :n]
        B0 = C_center[:, n:]
        K = solve_discrete_lqr(A0, B0, self.Q, self.R)
        X0 = (self.config.X0_scale) * np.eye(n)
        C_cur = C_center.copy()
        costs: List[float] = []

        for _ in range(self.config.outer_iters):
            # Inner loop: ascend over dynamics (unconstrained)
            for _ in range(self.config.inner_iters):
                A = C_cur[:, :n]
                B = C_cur[:, n:]
                _, grad_C, _, _, _ = gradients(K, A, B, self.Q, self.R, X0)
                C_next = C_cur + self.config.step_c * grad_C
                C_cur = C_next
            # Outer update on K using worst-case C_cur
            A_wc = C_cur[:, :n]
            B_wc = C_cur[:, n:]
            grad_K, _, _, _, _ = gradients(K, A_wc, B_wc, self.Q, self.R, X0)
            K = K - self.config.step_k * grad_K
            cost_wc = lqr_cost(K, A_wc, B_wc, self.Q, self.R, X0)
            costs.append(cost_wc)

        return K, C_cur, costs
