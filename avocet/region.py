from __future__ import annotations

import dataclasses
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cvxpy as cp
except Exception:  # pragma: no cover - cvxpy may be optional at import time
    cp = None  # type: ignore


@dataclasses.dataclass
class ScoreGeometry:
    """
    Metadata describing the shape of prediction regions induced by a score function.

    Supported types:
    - name: "l2_ball" (convex), with params {"p": 2}
    - name: "union": union of convex subregions
    - name: "unknown": geometry not available
    """

    name: str
    convex: bool
    union: bool = False
    params: Optional[dict] = None

    def supports_cvxpy(self) -> bool:
        return cp is not None and self.convex


class PredictionRegion:
    """Prediction region abstraction with geometry-aware utilities."""

    def __init__(
        self,
        geometry: ScoreGeometry,
        center: Optional[np.ndarray] = None,
        radius: Optional[float] = None,
        components: Optional[Sequence["PredictionRegion"]] = None,
    ):
        self.geometry = geometry
        self.center = center
        self.radius = radius
        self.components = list(components) if components is not None else None

    @classmethod
    def l2_ball(cls, center: np.ndarray, radius: float) -> "PredictionRegion":
        geom = ScoreGeometry(name="l2_ball", convex=True, union=False, params={"p": 2})
        return cls(geometry=geom, center=center, radius=float(radius))

    @classmethod
    def union(cls, regions: Sequence["PredictionRegion"]) -> "PredictionRegion":
        geom = ScoreGeometry(name="union", convex=False, union=True)
        return cls(geometry=geom, components=regions)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw samples uniformly from the region (approximate for ball)."""
        if rng is None:
            rng = np.random.default_rng()
        if self.geometry.name == "l2_ball":
            assert self.center is not None and self.radius is not None
            dim = self.center.shape[-1] if self.center.ndim > 0 else 1
            # Sample from isotropic normal and project to the ball radius.
            raw = rng.normal(size=(n, dim))
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            directions = raw / norms
            radii = rng.random(size=(n, 1)) ** (1.0 / dim) * self.radius
            return self.center + directions * radii
        if self.geometry.name == "union" and self.components:
            counts = rng.multinomial(n, [1 / len(self.components)] * len(self.components))
            samples: List[np.ndarray] = []
            for cnt, region in zip(counts, self.components):
                if cnt > 0:
                    samples.append(region.sample(cnt, rng))
            return np.vstack(samples) if samples else np.empty((0, 0))
        raise NotImplementedError(f"Sampling not implemented for {self.geometry.name}")

    def contains(self, y: np.ndarray) -> bool:
        """Check membership for simple geometries."""
        if self.geometry.name == "l2_ball":
            assert self.center is not None and self.radius is not None
            return float(np.linalg.norm(y - self.center)) <= self.radius + 1e-8
        if self.geometry.name == "union" and self.components:
            return any(region.contains(y) for region in self.components)
        raise NotImplementedError(f"Containment not implemented for {self.geometry.name}")

    def cvxpy_constraints(self, theta_var: "cp.Variable") -> List["cp.Constraint"]:
        """
        Return constraints describing the region for CVXPY-based optimization.

        Only available for convex regions with CVXPY installed.
        """
        if cp is None:
            raise ImportError("cvxpy is required for constraint generation.")
        if self.geometry.name == "l2_ball":
            assert self.center is not None and self.radius is not None
            return [cp.norm(theta_var - self.center, 2) <= self.radius]
        if self.geometry.name == "union":
            raise ValueError("Union regions require handling via decomposition.")
        raise NotImplementedError(f"Constraints not implemented for {self.geometry.name}")

    def is_convex(self) -> bool:
        if self.geometry.name == "union":
            return False
        return self.geometry.convex

    def as_union(self) -> List["PredictionRegion"]:
        if self.geometry.name == "union" and self.components:
            return list(self.components)
        return [self]
