from __future__ import annotations

import abc
from typing import List, Optional, Sequence

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
    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        ...

    @abc.abstractmethod
    def is_convex(self) -> bool:
        ...

    def as_union(self) -> Sequence["PredictionRegion"]:
        """Return components if union; otherwise singleton list."""
        return [self]

    @staticmethod
    def l2_ball(center: np.ndarray, radius: float) -> "PredictionRegion":
        return L2BallRegion(center=center, radius=radius)

    @staticmethod
    def l1_ball(center: np.ndarray, radius: float) -> "PredictionRegion":
        return L1BallRegion(center=center, radius=radius)

    @staticmethod
    def linf_ball(center: np.ndarray, radius: float) -> "PredictionRegion":
        return LinfBallRegion(center=center, radius=radius)

    @staticmethod
    def ellipsoid(center: np.ndarray, shape_matrix: np.ndarray, radius: float = 1.0) -> "PredictionRegion":
        return EllipsoidRegion(center=center, shape_matrix=shape_matrix, radius=radius)

    @staticmethod
    def union(regions: Sequence["PredictionRegion"]) -> "PredictionRegion":
        return UnionRegion(list(regions))


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

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.norm1(theta_var - self.center) <= self.radius]

    def is_convex(self) -> bool:
        return True


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

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.norm(theta_var - self.center, "inf") <= self.radius]

    def is_convex(self) -> bool:
        return True


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

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        return [cp.quad_form(theta_var - self.center, self.shape_matrix) <= self.radius**2]

    def is_convex(self) -> bool:
        return True


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

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        raise ValueError("Union regions require decomposition; no single convex constraint.")

    def is_convex(self) -> bool:
        return False

    def as_union(self) -> Sequence["PredictionRegion"]:
        return list(self.regions)
