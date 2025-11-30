# Example: Predict-Then-Optimize with Conformal Calibration

This directory contains an end-to-end example showing how to conformalize a predictor and use the resulting region in a robust optimization problem.

## `robust_bike_newsvendor.py`
Bike rental demand (UCI Bike Sharing) with robust newsvendor:
- Predictor: daily demand $\hat{y}(x)$.
- Region: L2-ball $\mathcal{C}(x) = \{c : \|c - \hat{y}(x)\|_2 \le q\}$.
- Optimization:

$$
\min_{q \ge 0} \max_{c \in \mathcal{C}(x)} \; c_u\,\max(c - q, 0) + c_o\,\max(q - c, 0).
$$

We solve the inner max over interval endpoints and compare to the nominal plug-in $q = \hat{y}(x)$.

## `robust_shortest_path_metrla.py`
Robust shortest path on METR-LA with conformalized DCRNN forecasts:
- Predictor: DCRNN_PyTorch speeds $\hat{s}(x)$; costs are $c = 1/\hat{s}$.
- Region: union of L2-balls over sampled costs $\mathcal{C}(x) = \bigcup_{k} \{c : \|c - \hat{c}_k\|_2 \le q\}$.
- Optimization (flow vector $w$, incidence $A$, supply $b$):

$$
\min_{w} \; t \quad
\text{s.t. } A w = b,\; 0 \le w \le 1,\;
t \ge \langle c_k, w\rangle + q \|w\|_2 \;\; \forall k.
$$

Nominal solves $\min_w \langle \bar{c}, w\rangle$ with mean cost $\bar{c}$.
