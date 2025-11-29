# Examples: Mathematical Formulations

This directory contains small scripts showing how to use Avocet for robust decision making. Below are the optimization problems each script solves.

## `robust_l2.py`
Uncertainty set: L2 ball `Theta = {theta : ||theta - c||_2 <= r}`.

Problem:
```
minimize    ||w||_2
subject to  <w, theta> <= 1   for all theta in Theta
```
Using the L2 support function, the robust constraint becomes:
`<w, c> + r * ||w||_2 <= 1`.

## `robust_l1.py`
Uncertainty set: L1 ball `Theta = {theta : ||theta - c||_1 <= r}`.

Problem:
```
minimize    ||w||_1
subject to  <w, theta> <= 1   for all theta in Theta
```
Robust constraint via support function:
`<w, c> + r * ||w||_∞ <= 1`.

## `robust_linf.py`
Uncertainty set: Linf ball (hypercube) `Theta = {theta : ||theta - c||_∞ <= r}`.

Problem:
```
minimize    ||w||_2
subject to  <w, theta> <= 1   for all theta in Theta
```
Robust constraint:
`<w, c> + r * ||w||_1 <= 1`.

## `robust_ellipsoid.py`
Uncertainty set: Ellipsoid `Theta = {theta : (theta - c)^T W (theta - c) <= r^2}`, with `W` PSD.

Problem:
```
minimize    ||w||_2
subject to  <w, theta> <= 1   for all theta in Theta
```
Robust constraint:
`<w, c> + r * sqrt(w^T W^{-1} w) <= 1`.

## `robust_union.py`
Uncertainty set: Union of two L2 balls `Theta = Theta_1 ∪ Theta_2`.

Problem:
```
minimize    ||w - theta||_2^2
subject to  w ∈ [-1, 1]^2   for all theta in Theta
```
Approach: decompose over components and approximate each with scenarios, taking the max objective across components.

## `robust_supply_planning.py`
Pipeline:
1. Train a PyTorch predictor `f(x)` for a 2D demand vector `theta`.
2. Split conformal calibration with L2 score produces an L2-ball region `Theta(x) = {theta : ||theta - f(x)||_2 <= q_alpha}`.
3. Robust planning problem for a new feature `x_new`:
```
Theta = Theta(x_new)
minimize    lam * ||w||_2^2  - min_{theta in Theta} <w, theta>
subject to  w >= 0,   sum(w) <= 1
```
Using the L2 support function:
`minimize lam * ||w||_2^2  - <w, c> + r * ||w||_2` with the same constraints, where `c` is the predicted center and `r` the conformal radius.

The script also plots empirical calibration curves on held-out test data.
