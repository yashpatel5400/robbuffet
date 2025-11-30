from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .region import PredictionRegion


def plot_region_1d(region: PredictionRegion, ax: Optional[plt.Axes] = None, color: str = "C0"):
    if ax is None:
        _, ax = plt.subplots()
    if region.name == "l2_ball":
        center = float(np.array(region.center).squeeze())
        r = float(region.radius)
        ax.axvspan(center - r, center + r, alpha=0.3, color=color, label="prediction region")
        ax.axvline(center, color=color, linestyle="--", label="prediction")
    elif region.name == "union":
        for idx, comp in enumerate(region.as_union()):
            plot_region_1d(comp, ax=ax, color=f"C{idx}")
    else:
        raise NotImplementedError(f"1D plot not available for {region.name}")
    ax.set_ylabel("Region indicator")
    ax.legend()
    return ax


def plot_region_2d(
    region: PredictionRegion,
    grid_limits: Tuple[Tuple[float, float], Tuple[float, float]],
    resolution: int = 100,
    ax: Optional[plt.Axes] = None,
    cmap: str = "Blues",
):
    if ax is None:
        _, ax = plt.subplots()
    x1 = np.linspace(*grid_limits[0], resolution)
    x2 = np.linspace(*grid_limits[1], resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    points = np.stack([xx1.ravel(), xx2.ravel()], axis=-1)
    mask = np.array([region.contains(p) for p in points]).reshape(xx1.shape)
    ax.contourf(xx1, xx2, mask, levels=1, cmap=cmap, alpha=0.5)
    ax.set_xlim(grid_limits[0])
    ax.set_ylim(grid_limits[1])
    ax.set_aspect("equal")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    return ax


def plot_calibration_curve(
    alphas,
    coverages,
    ax: Optional[plt.Axes] = None,
    label: str = "empirical coverage",
    title: str = "Calibration curve",
):
    if ax is None:
        _, ax = plt.subplots()
    x_axis = 1 - np.array(alphas)
    ax.plot(x_axis, coverages, marker="o", label=label)
    ax.plot(x_axis, x_axis, "--", label="target coverage")
    ax.set_xlabel("1 - alpha")
    ax.set_ylabel("coverage")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax