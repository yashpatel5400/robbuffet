"""
Robust shortest path on the METR-LA traffic network with conformalized DCRNN predictions.

Pipeline:
- Load METR-LA graph (adjacency) and speed data (METR-LA).
- Use the DCRNN model (Liyaguang/DCRNN, pre-trained weights) to forecast future speeds.
  Attribution: https://github.com/liyaguang/DCRNN
- Treat speed forecasts as a generative model: sample K trajectories by adding calibrated noise.
- Conformalize with a GPCP score to get a union-of-balls region over edge costs.
- Solve robust vs nominal shortest-path flows; visualize calibration and flows.

Requirements:
- Clone https://github.com/liyaguang/DCRNN and install its dependencies (TensorFlow 1.x).
- Provide paths to METR-LA data and the pre-trained checkpoint (see DCRNN repo releases).
"""

import argparse
import os
import pickle
from typing import Tuple

import cvxpy as cp
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from avocet import SplitConformalCalibrator, vis
from avocet.scores import GPCPScore, conformal_quantile


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------


def load_adj(adj_path: str):
    with open(adj_path, "rb") as f:
        _, _, adj_mx = pickle.load(f, encoding="latin1")
    return adj_mx  # (N, N)


def build_graph_from_adj(adj_mx: np.ndarray) -> Tuple[nx.DiGraph, list]:
    G = nx.DiGraph()
    N = adj_mx.shape[0]
    edges = []
    for i in range(N):
        for j in range(N):
            if adj_mx[i, j] > 0:
                edges.append((i, j))
                G.add_edge(i, j, weight=adj_mx[i, j], idx=len(edges) - 1)
    return G, edges


def load_speed_data(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        # shape (num_samples, num_sensors)
        return f["speed"][:]


# ---------------------------------------------------------------------------
# DCRNN inference wrapper (expects DCRNN repo available)
# ---------------------------------------------------------------------------


def run_dcrnn_forecast(dcrnn_root: str, ckpt: str, data: np.ndarray, horizon: int) -> np.ndarray:
    """
    Run DCRNN inference to forecast horizon steps.
    Returns forecasts of shape (T, N) for next-step (we take horizon 1 here).
    """
    try:
        import sys

        sys.path.append(dcrnn_root)
        from dcrnn.model.dcrnn_supervisor import DCRNNSupervisor
    except Exception as e:
        raise ImportError("DCRNN not available. Install from https://github.com/liyaguang/DCRNN") from e

    config_path = os.path.join(dcrnn_root, "config", "metr-la.yaml")
    supervisor = DCRNNSupervisor(config_filename=config_path)
    # monkey patch to load provided checkpoint
    supervisor._trainable = False
    supervisor._ckpt_path = ckpt
    # build data loader style input: (batch, seq_len, N, 1)
    # use last 12 steps to predict next
    seq_len = supervisor._model_config.get("seq_len", 12)
    X_list = []
    for t in range(data.shape[0] - seq_len - horizon):
        X_list.append(data[t : t + seq_len])
    X = np.stack(X_list, axis=0)[..., np.newaxis]  # (B, seq_len, N, 1)
    y_hat = supervisor.model.predict(X)  # (B, horizon, N, 1)
    # take first horizon step
    return y_hat[:, 0, :, 0]


# ---------------------------------------------------------------------------
# Robust shortest path solver (flow LP)
# ---------------------------------------------------------------------------


def build_incidence(edges: list, num_nodes: int) -> np.ndarray:
    E = len(edges)
    A = np.zeros((num_nodes, E))
    for idx, (u, v) in enumerate(edges):
        A[u, idx] = 1.0
        A[v, idx] = -1.0
    return A


def robust_shortest_path(A, b, centers_costs, radius_cost, base_cost):
    E = A.shape[1]
    w = cp.Variable(E)
    t = cp.Variable()
    constraints = [A @ w == b, w >= 0, w <= 1]
    for c in centers_costs:
        constraints.append(t >= c @ w + radius_cost * cp.norm(cp.multiply(base_cost, w), 2))
    prob = cp.Problem(cp.Minimize(t), constraints)
    status = None
    for solver in [cp.CLARABEL, cp.ECOS, None]:
        try:
            prob.solve(solver=solver)
        except Exception:
            continue
        status = prob.status
        if w.value is not None:
            break
    return (w.value if w.value is not None else None), status if status else "failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcrnn-root", type=str, required=True, help="Path to cloned DCRNN repo")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to DCRNN METR-LA checkpoint (ckpt file)")
    parser.add_argument("--adj", type=str, default="data/adj_mx.pkl", help="Path to adj_mx.pkl")
    parser.add_argument("--metr", type=str, default="data/metr-la.h5", help="Path to metr-la.h5")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.adj) or not os.path.exists(args.metr):
        raise SystemExit("METR-LA adjacency or data missing. Please download adj_mx.pkl and metr-la.h5 from the DCRNN repo.")

    adj_mx = load_adj(args.adj)
    G, edges = build_graph_from_adj(adj_mx)
    A = build_incidence(edges, adj_mx.shape[0])
    b = np.zeros(adj_mx.shape[0])
    b[args.source] = 1.0
    b[args.target] = -1.0

    speed = load_speed_data(args.metr)  # (T, N)
    # Use last chunk for inference/calibration/test
    T = speed.shape[0]
    split_train = int(0.7 * T)
    split_cal = int(0.85 * T)
    train_speed = speed[:split_train]
    cal_speed = speed[split_train:split_cal]
    test_speed = speed[split_cal:]

    # Forecast next step speeds for cal/test
    cal_pred = run_dcrnn_forecast(args.dcrnn_root, args.ckpt, cal_speed, horizon=1)
    test_pred = run_dcrnn_forecast(args.dcrnn_root, args.ckpt, test_speed, horizon=1)

    # Estimate residual std for sampling
    resid = cal_speed[12 : 12 + cal_pred.shape[0]] - cal_pred  # align with forecast start
    resid_std = np.std(resid, axis=0, keepdims=True) + 1e-3

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        # xb is ignored; use precomputed predictions + Gaussian noise
        # To keep shapes consistent, xb carries indices into test_pred
        idxs = xb.squeeze().long().cpu().numpy()
        preds = test_pred[idxs]  # (batch, N)
        samples = []
        for p in preds:
            noise = np.random.normal(scale=resid_std.squeeze(), size=(args.K, p.shape[0]))
            samples.append(p[None, :] + noise)
        samples = np.stack(samples, axis=1)  # (K, batch, N)
        return torch.tensor(samples, dtype=torch.float32)

    # Build calibration/test datasets as index tensors
    cal_idx = torch.arange(min(500, cal_pred.shape[0]))  # use subset for speed
    test_idx = torch.arange(min(200, test_pred.shape[0]))
    cal_loader = DataLoader(TensorDataset(cal_idx, torch.zeros_like(cal_idx)), batch_size=64)

    score_fn = GPCPScore(sampler)
    calibrator = SplitConformalCalibrator(lambda idx: sampler(idx), score_fn, cal_loader)
    q = calibrator.calibrate(alpha=args.alpha)
    print(f"Calibrated GPCP radius q: {q:.4f}")

    # Calibration curve
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = []
    cal_scores = calibrator.compute_scores(cal_loader).numpy()
    test_scores = []
    with torch.no_grad():
        for xb, _ in DataLoader(TensorDataset(test_idx, torch.zeros_like(test_idx)), batch_size=64):
            test_scores.append(score_fn.score(sampler(xb), torch.zeros_like(xb)).cpu())
    test_scores = torch.cat(test_scores).numpy()
    for a in alphas:
        q_a = conformal_quantile(torch.tensor(cal_scores), a)
        coverages.append(float(np.mean(test_scores <= q_a)))
    fig, ax = plt.subplots()
    vis.plot_calibration_curve(alphas, coverages, title="Calibration curve (METR-LA)", ax=ax, label="GPCP")
    ax.legend()
    fig.tight_layout()
    fig.savefig("metrla_calibration_curve.png", dpi=150)
    plt.close(fig)
    print("Saved metrla_calibration_curve.png")

    # Pick a test point
    x_new = torch.tensor([0])  # first test index
    region = calibrator.predict_region(x_new)
    centers = np.stack([r.center for r in region.as_union()])
    radius = q
    mean_pred = centers.mean(axis=0)

    # Robust vs nominal
    w_robust, status_r = robust_shortest_path(A, b, centers, radius, base_cost=np.ones_like(mean_pred))
    w_nom, status_n = nominal_shortest_path(A, b, mean_pred)

    if w_robust is None:
        print("Robust solve failed.")
        return
    w_robust = np.asarray(w_robust).reshape(-1)
    w_nom = np.asarray(w_nom).reshape(-1)

    # Evaluate on true cost (use first test speed as inverse speed ~ cost)
    true_speed = test_speed[12]  # align to forecast horizon
    true_cost = 1.0 / np.maximum(true_speed, 1e-3)
    robust_cost = true_cost @ w_robust
    nominal_cost = true_cost @ w_nom
    print(f"True path cost (robust): {robust_cost:.4f}")
    print(f"True path cost (nominal): {nominal_cost:.4f}")

    # Visualize flows side by side
    pos = nx.spring_layout(G, seed=0, k=1 / np.sqrt(G.number_of_nodes()))

    def edge_widths(flow):
        return [max(flow[G[u][v]["idx"]] * 5, 0.1) for u, v in G.edges()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, flow, title in zip(axes, [w_nom, w_robust], ["Nominal flow", "Robust flow"]):
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightgray", ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths(flow), edge_color="C0", arrows=False, ax=ax)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("metrla_flows.png", dpi=150)
    plt.close(fig)
    print("Saved metrla_flows.png")


if __name__ == "__main__":
    main()
