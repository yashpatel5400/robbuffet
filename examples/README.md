# Example: Predict-Then-Optimize with Conformal Calibration

This directory contains an end-to-end example showing how to conformalize a predictor and use the resulting region in a robust optimization problem.

## `robust_bike_newsvendor.py`
Bike rental demand (UCI Bike Sharing) with robust newsvendor:
- Predictor: daily demand $\hat{y}(x)$.
- Region: L2-ball $\mathcal{C}(x) = \{c : \|c - \hat{y}(x)\|_2 \le q\}$.
- Optimization:

$$
\min_{q \ge 0} \max_{c \in C(x)} \quad c_u\,\max(c - q, 0) + c_o\,\max(q - c, 0).
$$

We solve the inner max over interval endpoints and compare to the nominal plug-in $q = \hat{y}(x)$.

## `robust_shortest_path_metrla.py`
Robust shortest path on METR-LA with conformalized DCRNN forecasts:
- Predictor: DCRNN_PyTorch speeds $\hat{s}(x)$; costs are $c = 1/\hat{s}$.
- Region: union of L2-balls over sampled costs $\mathcal{C}(x) = \bigcup_{k} \{c : \|c - \hat{c}_k\|_2 \le q\}$.
- Optimization (flow vector $w$, incidence $A$, supply $b$):

$$
\min_{w} \quad t \quad
\text{s.t. } A w = b,\quad 0 \le w \le 1,\quad
t \ge \langle c_k, w\rangle + q \|w\|_2 \quad\quad \forall k.
$$

Nominal solves $\min_w \langle \bar{c}, w\rangle$ with mean cost $\bar{c}$.

## `robust_fractional_knapsack.py`
Fractional knapsack with SBIBM two_moons simulator:
- Predictor: MLP forecasting item values/weights $(v, w)$ from simulated features $x$.
- Region: union of L2-balls over sampled value vectors $\mathcal{C}(x) = \bigcup_k \{v : \|v - \hat{v}_k\|_2 \le q\}$; weights fixed to nominal.
- Optimization (fractional decision $x \in [0,1]^m$, capacity $B$):

$$
\max_{x} \; \langle v, x\rangle \quad \text{s.t. } \langle w, x\rangle \le B,\; 0 \le x \le 1,
$$

with a robust variant maximizing the worst-case $\min_{v \in \mathcal{C}(x)} \langle v, x\rangle$ (approximated via worst-case per-item values).

## Empirical results (10 trials)

### Newsvendor (Bike Sharing)

| method  | mean objective | std    | paired t-test (robust < nominal) |
|---------|----------------|--------|----------------------------------|
| robust  | 2560.51        | 24.30  | t = -90.94, p = 5.958e-15        |
| nominal | 4370.20        | 83.10  | –                                |

### Shortest path (METR-LA)

| method  | mean objective | std      | paired t-test (robust < nominal) |
|---------|----------------|----------|----------------------------------|
| robust  | 109.58         | 15.56    | t = -9.52, p = 2.682e-06         |
| nominal | 12112.04       | 3780.02  | –                                |
