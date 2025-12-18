from __future__ import annotations

import numpy as np


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
