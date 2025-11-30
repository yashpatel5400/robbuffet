# Example: Predict-Then-Optimize with Conformal Calibration

This directory contains an end-to-end example showing how to conformalize a predictor and use the resulting region in a robust optimization problem.

## `robust_portfolio.py`
Pipeline:
1. Train a PyTorch predictor for multi-asset returns (synthetic, from Wine Quality features).
2. Split conformal calibration with multiple scores (L2, L1, Linf) produces regions in return space.
3. Robust portfolio: maximize worst-case return under budget/cap constraints vs. nominal plug-in.

## `robust_shortest_path.py`
Synthetic predict-then-optimize example for robust shortest path:
1. Train a generative predictor (Gaussian) for edge costs given features.
2. Calibrate with a GPCP score using K samples per point; region is a union of L2 balls.
3. Robust shortest path: minimize worst-case cost across sampled centers with an L2 buffer.
4. Compare robust vs nominal cost on a held-out true cost vector.

## `robust_bike_newsvendor.py`
Bike rental demand (UCI Bike Sharing):
1. Train a PyTorch predictor for daily bike demand.
2. Calibrate with split conformal (L2), plot calibration curve.
3. Robust newsvendor decision using scenario sampling over conformal L2-ball regions; compare to nominal decisions on test days.

## `robust_shortest_path_metrla.py`
Robust shortest path on METR-LA with conformalized DCRNN forecasts:
1. Use DCRNN pretrained on METR-LA to forecast edge speeds; derive costs.
2. Calibrate with GPCP over sampled forecasts to get a union-of-balls region.
3. Solve robust vs nominal flows and visualize.

## `robust_shortest_path.py`
Synthetic predict-then-optimize example for robust shortest path:
1. Train a generative predictor (Gaussian) for edge costs given features.
2. Calibrate with a GPCP score using K samples per point; region is a union of L2 balls.
3. Robust shortest path: minimize worst-case cost across sampled centers with an L2 buffer.
4. Compare robust vs nominal cost on a held-out true cost vector.
