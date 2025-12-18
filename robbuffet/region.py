from __future__ import annotations

import abc
import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - cvxpy may be optional at import time
    cp = None  # type: ignore


class PredictionRegion(abc.ABC):
    """Abstract prediction region."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @abc.abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        ...

    @abc.abstractmethod
    def contains(self, y: np.ndarray) -> bool:
        ...

    @abc.abstractmethod
    def cvxpy_constraints(
        self, theta_var: "cp.Variable", theta_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List["cp.Constraint"]:
        ...

    @abc.abstractmethod
    def is_convex(self) -> bool:
        ...

    def as_union(self) -> Sequence["PredictionRegion"]:
        """Return components if union; otherwise singleton list."""
        return [self]

    @property
    def volume(self) -> Optional[float]:
        """Exact volume if available; otherwise None."""
        return None

    def volume_mc(
        self, bounds: Tuple[np.ndarray, np.ndarray], num_samples: int = 20000, rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Monte Carlo volume estimate within bounding box (low, high).
        Useful for unions or shapes without closed-form volumes.
        """
        if rng is None:
            rng = np.random.default_rng()
        low, high = bounds
        low = np.asarray(low)
        high = np.asarray(high)
        if low.shape != high.shape:
            raise ValueError("bounds must have same shape")
        dim = low.shape[0]
        samples = rng.uniform(low=low, high=high, size=(num_samples, dim))
        mask = np.array([self.contains(p) for p in samples], dtype=float)
        vol_box = float(np.prod(high - low))
        return vol_box * float(mask.mean())

    @abc.abstractmethod
    def support_function(self, direction) -> "cp.Expression":
        """Support function h_R(direction) = max_{theta in R} <direction, theta>."""
        ...

class L2BallRegion(PredictionRegion):
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = float(radius)

    @property
    def name(self) -> str:
        return "l2_ball"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        dim = self.center.shape[-1] if self.center.ndim > 0 else 1
        raw = rng.normal(size=(n, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        directions = raw / norms
        radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
        return self.center + directions * radii

    def contains(self, y: np.ndarray) -> bool:
        return float(np.linalg.norm(y - self.center)) <= self.radius + 1e-8

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.norm(theta_var - self.center, 2) <= self.radius]

    def is_convex(self) -> bool:
        return True

    @property
    def volume(self) -> Optional[float]:
        d = self.center.shape[-1]
        return _ball_volume_unit(d) * (self.radius**d)

    def support_function(self, direction) -> "cp.Expression":
        if cp is None:
            raise ImportError("cvxpy is required for support_function.")
        if not isinstance(direction, cp.Expression):
            direction = cp.Constant(direction)
        base = direction @ self.center
        return base + self.radius * cp.norm(direction, 2)


class L1BallRegion(PredictionRegion):
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = float(radius)

    @property
    def name(self) -> str:
        return "l1_ball"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        dim = self.center.shape[-1] if self.center.ndim > 0 else 1
        exp_samples = rng.exponential(scale=1.0, size=(n, dim))
        signs = rng.choice([-1.0, 1.0], size=(n, dim))
        directions = signs * exp_samples
        l1_norms = np.sum(np.abs(directions), axis=1, keepdims=True)
        l1_norms[l1_norms == 0] = 1.0
        directions = directions / l1_norms
        radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
        return self.center + directions * radii

    def contains(self, y: np.ndarray) -> bool:
        return float(np.linalg.norm(y - self.center, ord=1)) <= self.radius + 1e-8

    def cvxpy_constraints(
        self, theta_var: "cp.Variable", theta_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.norm1(theta_var - self.center) <= self.radius]

    def is_convex(self) -> bool:
        return True

    @property
    def volume(self) -> Optional[float]:
        d = self.center.shape[-1]
        return (2**d / math.factorial(d)) * (self.radius**d)

    def support_function(self, direction) -> "cp.Expression":
        if cp is None:
            raise ImportError("cvxpy is required for support_function.")
        if not isinstance(direction, cp.Expression):
            direction = cp.Constant(direction)
        base = direction @ self.center
        return base + self.radius * cp.norm(direction, "inf")


class LinfBallRegion(PredictionRegion):
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = float(radius)

    @property
    def name(self) -> str:
        return "linf_ball"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        dim = self.center.shape[-1] if self.center.ndim > 0 else 1
        offsets = rng.uniform(low=-self.radius, high=self.radius, size=(n, dim))
        return self.center + offsets

    def contains(self, y: np.ndarray) -> bool:
        return float(np.linalg.norm(y - self.center, ord=np.inf)) <= self.radius + 1e-8

    def cvxpy_constraints(
        self, theta_var: "cp.Variable", theta_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.norm(theta_var - self.center, "inf") <= self.radius]

    def is_convex(self) -> bool:
        return True

    @property
    def volume(self) -> Optional[float]:
        d = self.center.shape[-1]
        return (2 * self.radius) ** d

    def support_function(self, direction) -> "cp.Expression":
        if cp is None:
            raise ImportError("cvxpy is required for support_function.")
        if not isinstance(direction, cp.Expression):
            direction = cp.Constant(direction)
        base = direction @ self.center
        return base + self.radius * cp.norm(direction, 1)


class EllipsoidRegion(PredictionRegion):
    def __init__(self, center: np.ndarray, shape_matrix: np.ndarray, radius: float = 1.0):
        self.center = center
        self.shape_matrix = shape_matrix
        self.radius = float(radius)

    @property
    def name(self) -> str:
        return "ellipsoid"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        dim = self.center.shape[-1]
        raw = rng.normal(size=(n, dim))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        directions = raw / norms
        radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
        unit_ball_samples = directions * radii
        eigvals, eigvecs = np.linalg.eigh(self.shape_matrix)
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
        transformed = unit_ball_samples @ inv_sqrt.T
        return self.center + transformed

    def contains(self, y: np.ndarray) -> bool:
        diff = y - self.center
        return float(diff.T @ self.shape_matrix @ diff) <= self.radius**2 + 1e-8

    def cvxpy_constraints(
        self, theta_var: "cp.Variable", theta_bounds: Optional[tuple[np.ndarray, np.ndarray]] = None
    ) -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.quad_form(theta_var - self.center, self.shape_matrix) <= self.radius**2]

    def is_convex(self) -> bool:
        return True

    @property
    def volume(self) -> Optional[float]:
        d = self.center.shape[-1]
        det = np.linalg.det(self.shape_matrix)
        if det <= 0:
            return None
        return _ball_volume_unit(d) * (self.radius**d) / math.sqrt(det)

    def support_function(self, direction) -> "cp.Expression":
        if cp is None:
            raise ImportError("cvxpy is required for support_function.")
        if not isinstance(direction, cp.Expression):
            direction = cp.Constant(direction)
        base = direction @ self.center
        w_inv = np.linalg.inv(self.shape_matrix)
        quad = cp.quad_form(direction, w_inv)
        return base + self.radius * cp.sqrt(quad)


class UnionRegion(PredictionRegion):
    def __init__(self, regions: Sequence[PredictionRegion]):
        self.regions = list(regions)

    @property
    def name(self) -> str:
        return "union"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if len(self.regions) == 0:
            return np.empty((0, 0))
        counts = rng.multinomial(n, [1 / len(self.regions)] * len(self.regions))
        samples = []
        for cnt, region in zip(counts, self.regions):
            if cnt > 0:
                samples.append(region.sample(cnt, rng))
        return np.vstack(samples) if samples else np.empty((0, 0))

    def contains(self, y: np.ndarray) -> bool:
        return any(region.contains(y) for region in self.regions)

    def cvxpy_constraints(
        self, theta_var: "cp.Variable", theta_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List["cp.Constraint"]:
        raise ValueError("Union regions have no single convex CVXPY constraint; use support functions or sampling.")

    def is_convex(self) -> bool:
        return False

    def as_union(self) -> Sequence["PredictionRegion"]:
        return list(self.regions)

    def support_function(self, direction) -> "cp.Expression":
        if cp is None:
            raise ImportError("cvxpy is required for support_function.")
        comps = [r.support_function(direction) for r in self.regions]
        return cp.maximum(*comps)


def _ball_volume_unit(d: int) -> float:
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


class OperatorNormBallRegion(PredictionRegion):
    """
    Spectral-norm ball around a matrix center.
    Useful for dynamics matrices C = [A, B].
    """

    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = float(radius)

    @property
    def name(self) -> str:
        return "operator_norm_ball"

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        samples = []
        for _ in range(n):
            raw = rng.standard_normal(size=self.center.shape)
            u, _, vh = np.linalg.svd(raw, full_matrices=False)
            direction = u @ vh  # spectral norm 1
            scale = rng.random() ** (1.0 / np.prod(self.center.shape))
            samples.append(self.center + self.radius * scale * direction)
        return np.stack(samples, axis=0)

    def contains(self, y: np.ndarray) -> bool:
        return float(np.linalg.norm(y - self.center, ord=2)) <= self.radius + 1e-8

    def cvxpy_constraints(
        self, theta_var: "cp.Variable", theta_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.norm(theta_var - self.center, 2) <= self.radius]

    def is_convex(self) -> bool:
        return True

    def support_function(self, direction) -> "cp.Expression":
        if cp is None:
            raise ImportError("cvxpy is required for support_function.")
        if not isinstance(direction, cp.Expression):
            direction = cp.Constant(direction)
        base = cp.sum(cp.multiply(direction, self.center))
        # Dual norm of spectral is nuclear
        return base + self.radius * cp.norm(direction, "nuc")
