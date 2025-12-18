"""
Discrete-time H-infinity state-feedback synthesis via an LMI (bounded real lemma).

Returns a controller K for z = [Q^{1/2} x; R^{1/2} u] with ||Tzw|| < gamma.
Uses a simple feasibility solve; if infeasible, falls back to LQR.
"""

from __future__ import annotations

import numpy as np

import cvxpy as cp

from robcontrol.controllers.base import RobustController
from robcontrol.utils import solve_discrete_lqr


def _sqrt_psd(M: np.ndarray) -> np.ndarray:
    """Return a symmetric square root of a PSD matrix."""
    try:
        return np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(M)
        w_clipped = np.clip(w, 0.0, None)
        return v @ np.diag(np.sqrt(w_clipped)) @ v.T


class HInfinityController(RobustController):
    """
    Discrete-time H-infinity controller using bounded-real LMI.
    If infeasible, falls back to LQR gain.
    """

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def synthesize(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
        n, m = B.shape[0], B.shape[1]
        C = _sqrt_psd(Q)
        D = _sqrt_psd(R)

        P = cp.Variable((n, n), PSD=True)
        Y = cp.Variable((m, n))

        AP_BY = A @ P + B @ Y
        CP_DY = C @ P + D @ Y

        top = cp.hstack([P, AP_BY.T, CP_DY.T])
        mid = cp.hstack([AP_BY, P, np.zeros((n, C.shape[0]))])
        bot = cp.hstack([CP_DY, np.zeros((C.shape[0], n)), self.gamma * np.eye(C.shape[0])])
        big = cp.vstack([top, mid, bot])

        constraints = [P >> 1e-6 * np.eye(n), big >> 1e-6 * np.eye(big.shape[0])]
        prob = cp.Problem(cp.Minimize(0), constraints)
        # Try a sequence of solvers; any failure falls back to LQR.
        # Avoid Clarabel to sidestep upstream panics; prefer ECOS/SCS/OSQP/default.
        solvers = [cp.ECOS, cp.SCS, cp.OSQP, None]
        solved = False
        for solver in solvers:
            try:
                prob.solve(solver=solver, verbose=False)
            except Exception:
                continue
            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and P.value is not None:
                solved = True
                break
        if not solved:
            return solve_discrete_lqr(A, B, Q, R)

        P_val = np.array(P.value, dtype=float)
        Y_val = np.array(Y.value, dtype=float)
        try:
            K = Y_val @ np.linalg.inv(P_val)
        except np.linalg.LinAlgError:
            return solve_discrete_lqr(A, B, Q, R)
        return K
