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
import sys
from typing import Tuple

import cvxpy as cp
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


def maybe_download_metrla(h5_path: str):
    if os.path.exists(h5_path):
        return
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    file_id = "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        import urllib.request

        print(f"Downloading METR-LA data to {h5_path} ...")
        urllib.request.urlretrieve(url, h5_path)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download METR-LA data to {h5_path}. Please download manually from {url}") from e


def maybe_generate_adj(dcrnn_root: str, adj_path: str):
    if os.path.exists(adj_path):
        return
    sensor_ids = os.path.join(dcrnn_root, "data/sensor_graph/graph_sensor_ids.txt")
    if not os.path.exists(sensor_ids):
        raise RuntimeError("Missing sensor IDs file; ensure DCRNN submodule is initialized with data.")
    os.makedirs(os.path.dirname(adj_path), exist_ok=True)
    try:
        import subprocess
        cmd = [
            "python",
            "-m",
            "scripts.gen_adj_mx",
            f"--sensor_ids_filename={sensor_ids}",
            "--normalized_k=0.1",
            f"--output_pkl_filename={adj_path}",
        ]
        subprocess.run(cmd, check=True, cwd=dcrnn_root)
    except Exception as e:
        raise RuntimeError(f"Failed to generate adj_mx.pkl at {adj_path}") from e


def maybe_generate_dataset(dcrnn_root: str, metr_path: str, dataset_dir: str):
    """
    Use DCRNN's generate_training_data.py to create train/val/test NPZ splits
    that the library utilities expect.
    """
    train_npz = os.path.join(dataset_dir, "train.npz")
    if os.path.exists(train_npz):
        return
    os.makedirs(dataset_dir, exist_ok=True)
    try:
        import subprocess

        cmd = [
            "python",
            "scripts/generate_training_data.py",
            f"--output_dir={dataset_dir}",
            f"--traffic_df_filename={metr_path}",
        ]
        subprocess.run(cmd, check=True, cwd=dcrnn_root)
    except Exception as e:
        raise RuntimeError(f"Failed to generate train/val/test splits at {dataset_dir}") from e


# ---------------------------------------------------------------------------
# DCRNN inference wrapper (expects DCRNN repo available)
# ---------------------------------------------------------------------------


def dcrnn_val_test_predictions(
    dcrnn_root: str, ckpt_prefix: str, adj_path: str, dataset_dir: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the pre-trained DCRNN and return (val_pred, val_true, test_pred, test_true)
    for horizon-1 forecasts, all in original speed units.
    """
    import sys
    import tensorflow as tf
    import yaml

    sys.path.append(dcrnn_root)
    from lib.utils import load_graph_data
    from model.dcrnn_supervisor import DCRNNSupervisor

    config_path = os.path.join(dcrnn_root, "data/model/pretrained/METR-LA/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Point to local data/paths.
    config["data"]["dataset_dir"] = dataset_dir
    config["data"]["graph_pkl_filename"] = adj_path
    config["train"]["model_filename"] = ckpt_prefix

    _, _, adj_mx = load_graph_data(adj_path)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False

    def _predict_split(sess, supervisor, split: str):
        loader = supervisor._data[f"{split}_loader"]
        y_true = supervisor._data[f"y_{split}"]
        res = supervisor.run_epoch_generator(
            sess,
            supervisor._test_model,
            loader.get_iterator(),
            return_output=True,
            training=False,
        )
        y_preds = np.concatenate(res["outputs"], axis=0)  # (B, horizon, N, 1)
        scaler = supervisor._data["scaler"]
        preds_inv = scaler.inverse_transform(y_preds[: y_true.shape[0], 0, :, 0])
        truth_inv = scaler.inverse_transform(y_true[:, 0, :, 0])
        return preds_inv, truth_inv

    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, ckpt_prefix)
        val_pred, val_true = _predict_split(sess, supervisor, "val")
        test_pred, test_true = _predict_split(sess, supervisor, "test")
    return val_pred, val_true, test_pred, test_true


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


def nominal_shortest_path(A, b, cost):
    E = A.shape[1]
    w = cp.Variable(E)
    prob = cp.Problem(cp.Minimize(cost @ w), [A @ w == b, w >= 0, w <= 1])
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


def node_costs_to_edges(node_costs: np.ndarray, edges: list) -> np.ndarray:
    """Map per-node costs to per-edge by averaging endpoints."""
    return np.array([(node_costs[u] + node_costs[v]) / 2.0 for u, v in edges])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcrnn-root", type=str, default="examples/DCRNN", help="Path to cloned DCRNN repo")
    parser.add_argument("--ckpt", type=str, default="examples/DCRNN/data/model/pretrained/METR-LA/models-2.7422-24375", help="Path prefix to DCRNN METR-LA checkpoint (without extension)")
    parser.add_argument("--adj", type=str, default="examples/DCRNN/data/sensor_graph/adj_mx.pkl", help="Path to adj_mx.pkl")
    parser.add_argument("--metr", type=str, default="examples/DCRNN/data/metr-la.h5", help="Path to metr-la.h5")
    parser.add_argument("--dataset-dir", type=str, default="examples/DCRNN/data/METR-LA", help="Directory with DCRNN train/val/test NPZ files")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=10)
    args = parser.parse_args()

    dcrnn_root = os.path.abspath(args.dcrnn_root)
    adj_path = os.path.abspath(args.adj)
    metr_path = os.path.abspath(args.metr)
    dataset_dir = os.path.abspath(args.dataset_dir)
    ckpt_prefix = os.path.abspath(args.ckpt)

    maybe_generate_adj(dcrnn_root, adj_path)
    maybe_download_metrla(metr_path)
    maybe_generate_dataset(dcrnn_root, metr_path, dataset_dir)

    sys.path.append(dcrnn_root)
    from lib.utils import load_graph_data

    _, _, adj_mx = load_graph_data(adj_path)
    G, edges = build_graph_from_adj(adj_mx)
    A = build_incidence(edges, adj_mx.shape[0])
    b = np.zeros(adj_mx.shape[0])
    b[args.source] = 1.0
    b[args.target] = -1.0

    # DCRNN predictions (speed) on val/test splits
    val_pred, val_true, test_pred, test_true = dcrnn_val_test_predictions(
        dcrnn_root, ckpt_prefix, adj_path, dataset_dir
    )

    # Work in cost space (travel time ~ inverse speed)
    val_cost_pred = 1.0 / np.maximum(val_pred, 1e-3)
    val_cost_true = 1.0 / np.maximum(val_true, 1e-3)
    test_cost_pred = 1.0 / np.maximum(test_pred, 1e-3)
    test_cost_true = 1.0 / np.maximum(test_true, 1e-3)

    resid = val_cost_true - val_cost_pred
    resid_std = np.std(resid, axis=0, keepdims=True) + 1e-3

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        base = xb.cpu().numpy()
        noise = np.random.normal(scale=resid_std.squeeze(), size=(args.K, base.shape[0], base.shape[1]))
        samples = base[None, ...] + noise
        return torch.tensor(samples, dtype=torch.float32)

    cal_dataset = TensorDataset(
        torch.tensor(val_cost_pred, dtype=torch.float32), torch.tensor(val_cost_true, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_cost_pred, dtype=torch.float32), torch.tensor(test_cost_true, dtype=torch.float32)
    )
    cal_loader = DataLoader(cal_dataset, batch_size=32, shuffle=True)

    score_fn = GPCPScore(sampler)
    calibrator = SplitConformalCalibrator(sampler, score_fn, cal_loader)
    q = calibrator.calibrate(alpha=args.alpha)
    print(f"Calibrated GPCP radius q: {q:.4f}")

    # Calibration curve
    alphas = np.linspace(0.05, 0.5, num=10)
    coverages = []
    cal_scores = calibrator.compute_scores(cal_loader).numpy()
    test_scores = calibrator.compute_scores(DataLoader(test_dataset, batch_size=32)).numpy()
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
    x_new = torch.tensor(test_cost_pred[0:1], dtype=torch.float32)
    region = calibrator.predict_region(x_new)
    centers = np.stack([r.center for r in region.as_union()])
    radius = q
    mean_pred_nodes = centers.mean(axis=0)

    # Convert node-level costs to edge-level for optimization.
    centers_edges = np.stack([node_costs_to_edges(c, edges) for c in centers])
    mean_pred_edges = node_costs_to_edges(mean_pred_nodes, edges)

    # Robust vs nominal
    w_robust, status_r = robust_shortest_path(A, b, centers_edges, radius, base_cost=np.ones_like(mean_pred_edges))
    w_nom, status_n = nominal_shortest_path(A, b, mean_pred_edges)

    if w_robust is None:
        print("Robust solve failed.")
        return
    w_robust = np.asarray(w_robust).reshape(-1)
    w_nom = np.asarray(w_nom).reshape(-1)

    # Evaluate on true cost for the same test sample
    true_cost_nodes = test_cost_true[0]
    true_cost_edges = node_costs_to_edges(true_cost_nodes, edges)
    robust_cost = true_cost_edges @ w_robust
    nominal_cost = true_cost_edges @ w_nom
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
