from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import torch

from .region import (
    EllipsoidRegion,
    L1BallRegion,
    L2BallRegion,
    LinfBallRegion,
    PredictionRegion,
    UnionRegion,
)


ObjectiveFn = Callable[[cp.Variable, np.ndarray], cp.Expression]
ConstraintFn = Callable[[cp.Variable, np.ndarray], Sequence[cp.Constraint]]


def support_function(region: PredictionRegion, direction) -> cp.Expression:
    """
    Support function h_R(direction) = max_{theta in R} <direction, theta>.
    Assumes direction is a CVXPY expression or NumPy array.
    """
    return region.support_function(direction)


def robustify_affine_objective(
    base_obj: cp.Expression, theta_direction: cp.Expression, region: PredictionRegion
) -> cp.Expression:
    """
    Robustify an affine objective term g(w) + <theta_direction(w), theta>.
    Returns g(w) + h_R(theta_direction(w)).
    """
    return base_obj + support_function(region, theta_direction)


def robustify_affine_leq(
    theta_direction: cp.Expression, rhs: cp.Expression, region: PredictionRegion
) -> cp.Constraint:
    """
    Robustify a linear inequality <theta_direction(w), theta> <= rhs for all theta in region.
    Returns constraint: h_R(theta_direction(w)) <= rhs.
    """
    return support_function(region, theta_direction) <= rhs


class AffineRobustSolver:
    """
    Wrapper to build and solve a convex robust problem using analytic support functions.

    Interface:
      - base_objective_fn(w) -> cp.Expression (deterministic part)
      - theta_direction_fn(w) -> cp.Expression (affine theta direction) or None
      - constraints_fn(w) -> list of cp.Constraint (deterministic)
      - robust_constraints_fn(w) -> list of (theta_direction_expr, rhs_expr) to be robustified
    """

    def __init__(
        self,
        decision_shape: Tuple[int, ...],
        region: PredictionRegion,
        base_objective_fn: Callable[[cp.Variable], cp.Expression],
        theta_direction_fn: Optional[Callable[[cp.Variable], cp.Expression]] = None,
        constraints_fn: Optional[Callable[[cp.Variable], Sequence[cp.Constraint]]] = None,
        robust_constraints_fn: Optional[Callable[[cp.Variable], Sequence[Tuple[cp.Expression, cp.Expression]]]] = None,
        solver: Optional[str] = None,
    ):
        self.decision_shape = decision_shape
        self.region = region
        self.base_objective_fn = base_objective_fn
        self.theta_direction_fn = theta_direction_fn
        self.constraints_fn = constraints_fn or (lambda w: [])
        self.robust_constraints_fn = robust_constraints_fn or (lambda w: [])
        self.solver = solver

    def solve(self) -> Tuple[Optional[np.ndarray], str]:
        w = cp.Variable(self.decision_shape)
        base_obj = self.base_objective_fn(w)
        if self.theta_direction_fn is not None:
            theta_dir = self.theta_direction_fn(w)
            obj = robustify_affine_objective(base_obj, theta_dir, self.region)
        else:
            obj = base_obj

        constraints: List[cp.Constraint] = list(self.constraints_fn(w))
        for theta_dir, rhs in self.robust_constraints_fn(w):
            constraints.append(robustify_affine_leq(theta_dir, rhs, self.region))

        prob = cp.Problem(cp.Minimize(obj), constraints)
        status = None
        for solver in ([self.solver] if self.solver else [cp.CLARABEL, cp.ECOS, None]):
            try:
                prob.solve(solver=solver)
            except Exception:
                continue
            status = prob.status
            if w.value is not None:
                break
        return (np.array(w.value).astype(float) if w.value is not None else None), status or "failed"


class DanskinRobustOptimizer:
    """
    Gradient-based min-max solver using Danskin's Theorem.

    Assumes objective is convex in w and inner maximization over theta can be solved per region.
    The user provides:
      - inner_objective_fn(theta_var, w_np) -> cp.Expression (concave in theta).
      - value_and_grad_fn(w_np, theta_np) -> (value, grad_w_np).
      - project_fn: optional projection on w after each step.
    Supports unions by maximizing over components.
    """

    def __init__(
        self,
        region: PredictionRegion,
        nom_obj: Callable[[cp.Variable, np.ndarray], cp.Expression],
        value_and_grad_fn: Optional[Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]] = None,
        torch_value_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        project_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        solver: str = "ECOS",
    ):
        self.region = region
        self.nom_obj = nom_obj
        if value_and_grad_fn is None and torch_value_fn is None:
            raise ValueError("Provide either value_and_grad_fn or torch_value_fn.")
        if value_and_grad_fn is None and torch_value_fn is not None:
            self.value_and_grad_fn = self._make_torch_value_grad(torch_value_fn)
        else:
            assert value_and_grad_fn is not None
            self.value_and_grad_fn = value_and_grad_fn
        self.project_fn = project_fn
        self.solver = solver

    @staticmethod
    def _make_torch_value_grad(
        torch_value_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]:
        def wrapper(w_np: np.ndarray, theta_np: np.ndarray) -> tuple[float, np.ndarray]:
            w_t = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
            theta_t = torch.tensor(theta_np, dtype=torch.float64)
            val_t = torch_value_fn(w_t, theta_t)
            if val_t.dim() != 0:
                val_t = val_t.squeeze()
            val_t.backward()
            grad = w_t.grad.detach().cpu().numpy().astype(float)
            return float(val_t.item()), grad

        return wrapper

    def _argmax_theta(self, w: np.ndarray, region: PredictionRegion) -> tuple[np.ndarray, float]:
        if isinstance(region, UnionRegion):
            best_theta = None
            best_val = -np.inf
            for comp in region.regions:
                theta_candidate, val_candidate = self._argmax_theta(w, comp)
                if val_candidate > best_val:
                    best_val = val_candidate
                    best_theta = theta_candidate
            assert best_theta is not None
            return best_theta, best_val
        theta = cp.Variable(region.center.shape)  # type: ignore[attr-defined]
        obj = cp.Maximize(self.nom_obj(theta, w))
        constraints = region.cvxpy_constraints(theta)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=self.solver)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Inner maximization failed with status {prob.status}")
        theta_star = np.array(theta.value).astype(float)
        return theta_star, float(obj.value)

    def solve(
        self,
        w0: np.ndarray,
        step_size: float = 1e-1,
        max_iters: int = 200,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        w = np.array(w0, dtype=float)
        history: list[float] = []
        for it in range(max_iters):
            theta_star, _ = self._argmax_theta(w, self.region)
            val, grad_w = self.value_and_grad_fn(w, theta_star)
            history.append(val)
            w_next = w - step_size * grad_w
            if self.project_fn is not None:
                w_next = self.project_fn(w_next)
            if np.linalg.norm(w_next - w) < tol:
                w = w_next
                break
            w = w_next
            if verbose:
                print(f"iter {it}: val={val:.4f}, grad_norm={np.linalg.norm(grad_w):.4f}")
        return w, history


class ScenarioRobustOptimizer:
    """
    Scenario-based robust optimization over conformal prediction regions.

    The user provides:
    - decision variable shape,
    - objective_fn(decision_var, theta_sample) -> cp.Expression,
    - constraints_fn(decision_var, theta_sample) -> list of cp.Constraint.

    The optimizer samples the region and minimizes the worst-case objective
    over those samples (epigraph formulation). This yields an inner approximation
    of the true robust counterpart; increase num_samples for tighter results.
    """

    def __init__(
        self,
        decision_shape: Tuple[int, ...],
        objective_fn: ObjectiveFn,
        constraints_fn: Optional[ConstraintFn] = None,
        num_samples: int = 128,
        seed: Optional[int] = None,
    ):
        self.decision_shape = decision_shape
        self.objective_fn = objective_fn
        self.constraints_fn = constraints_fn or (lambda w, theta: [])
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

    def build_problem(
        self, region: PredictionRegion, solver: Optional[str] = None
    ) -> cp.Problem:
        w = cp.Variable(self.decision_shape)
        t = cp.Variable()
        constraints: List[cp.Constraint] = []
        if isinstance(region, UnionRegion):
            # solve per-component and enforce t >= max objective across components
            for comp in region.regions:
                samples = comp.sample(self.num_samples // len(region.regions) or 1, rng=self.rng)
                for theta in samples:
                    constraints.append(t >= self.objective_fn(w, theta))
                    constraints.extend(self.constraints_fn(w, theta))
        else:
            samples = region.sample(self.num_samples, rng=self.rng)
            for theta in samples:
                constraints.append(t >= self.objective_fn(w, theta))
                constraints.extend(self.constraints_fn(w, theta))
        problem = cp.Problem(cp.Minimize(t), constraints)
        if solver is not None:
            problem.solve(solver=solver)
        return problem
