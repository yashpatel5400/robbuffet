# Examples: Mathematical Formulations

This directory contains small scripts showing how to use Avocet for robust decision making. Below are the optimization problems each script solves.

## `robust_l2.py`
Uncertainty set: L2 ball \( \Theta = \{\theta : \|\theta - c\|_2 \le r\} \).

Problem:
\[
\min_w \ \|w\|_2 \quad \text{s.t.} \quad \langle w, \theta \rangle \le 1 \ \forall \theta \in \Theta
\]
Robust constraint: \( \langle w, c \rangle + r \|w\|_2 \le 1 \).

## `robust_l1.py`
Uncertainty set: L1 ball \( \Theta = \{\theta : \|\theta - c\|_1 \le r\} \).

Problem:
\[
\min_w \ \|w\|_1 \quad \text{s.t.} \quad \langle w, \theta \rangle \le 1 \ \forall \theta \in \Theta
\]
Robust constraint: \( \langle w, c \rangle + r \|w\|_\infty \le 1 \).

## `robust_linf.py`
Uncertainty set: Linf ball (hypercube) \( \Theta = \{\theta : \|\theta - c\|_\infty \le r\} \).

Problem:
\[
\min_w \ \|w\|_2 \quad \text{s.t.} \quad \langle w, \theta \rangle \le 1 \ \forall \theta \in \Theta
\]
Robust constraint: \( \langle w, c \rangle + r \|w\|_1 \le 1 \).

## `robust_ellipsoid.py`
Uncertainty set: Ellipsoid \( \Theta = \{\theta : (\theta - c)^\top W (\theta - c) \le r^2\} \), \(W \succeq 0\).

Problem:
\[
\min_w \ \|w\|_2 \quad \text{s.t.} \quad \langle w, \theta \rangle \le 1 \ \forall \theta \in \Theta
\]
Robust constraint: \( \langle w, c \rangle + r \sqrt{w^\top W^{-1} w} \le 1 \).

## `robust_union.py`
Uncertainty set: Union of two L2 balls \( \Theta = \Theta_1 \cup \Theta_2 \).

Problem:
\[
\min_w \ \|w - \theta\|_2^2 \quad \text{s.t.} \quad w \in [-1, 1]^2 \ \forall \theta \in \Theta
\]
Approach: decompose over components and approximate each with scenarios, taking the max objective across components.

## `robust_supply_planning.py`
Pipeline:
1. Train a PyTorch predictor \( f(x) \) for a 2D demand vector \( \theta \).
2. Split conformal calibration with L2 score produces an L2-ball region \( \Theta(x) = \{\theta : \|\theta - f(x)\|_2 \le q_\alpha\} \).
3. Robust planning for new \( x_{\text{new}} \):
\[
\Theta = \Theta(x_{\text{new}}), \quad
\min_w \ \lambda \|w\|_2^2 - \min_{\theta \in \Theta} \langle w, \theta \rangle \quad
\text{s.t. } w \ge 0, \ \mathbf{1}^\top w \le 1
\]
Robust objective via support: \( \min_w \lambda \|w\|_2^2 - \langle w, c \rangle + r \|w\|_2 \) with the same constraints, where \(c\) is the predicted center and \(r\) the conformal radius. The script also plots empirical calibration curves on held-out test data.
