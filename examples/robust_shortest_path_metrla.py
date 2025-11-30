"""
Robust shortest path on the METR-LA traffic network with conformalized DCRNN predictions.

Pipeline:
- Load METR-LA graph (adjacency) and precomputed DCRNN_PyTorch forecasts.
- Use the DCRNN model (https://github.com/chnsh/DCRNN_PyTorch) as the traffic predictor.
- Treat speed forecasts as a generative model: sample K trajectories by adding calibrated noise.
- Conformalize with a GPCP score to get a union-of-balls region over edge costs.
- Solve robust vs nominal shortest-path flows; visualize calibration and flows.

Requirements:
- Clone https://github.com/chnsh/DCRNN_PyTorch (we vendor it under examples/DCRNN_PyTorch).
- Have a precomputed predictions NPZ (run_demo_pytorch.py writes one at data/dcrnn_predictions_pytorch.npz).
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

from robbuffet import DanskinRobustOptimizer, SplitConformalCalibrator, vis
from robbuffet.region import L2BallRegion, UnionRegion
from robbuffet.scores import GPCPScore, conformal_quantile


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


def load_predictions(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed DCRNN predictions/truth from npz produced by run_demo_pytorch.
    Returns (pred, truth) for horizon 1 with shape (T, N).
    """
    data = np.load(npz_path, allow_pickle=True)
    preds = data["prediction"]
    truth = data["truth"]
    # Arrays are lists over horizon; take first step
    pred_h1 = np.array(preds[0])
    truth_h1 = np.array(truth[0])
    # Ensure shapes (T, N)
    if pred_h1.ndim == 3:
        pred_h1 = pred_h1[:, :, 0]
        truth_h1 = truth_h1[:, :, 0]
    if pred_h1.ndim == 1:
        pred_h1 = pred_h1[:, None]
        truth_h1 = truth_h1[:, None]
    return pred_h1, truth_h1


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


def feasibility_report(A: np.ndarray, b: np.ndarray, w: np.ndarray) -> str:
    res = np.linalg.norm(A @ w - b, ord=np.inf)
    return f"residual={res:.2e}, min={w.min():.3f}, max={w.max():.3f}"


def run_experiment(
    alpha: float = 0.1,
    K: int = 8,
    dcrnn_root: str = "examples/DCRNN_PyTorch",
    adj: str = "examples/DCRNN_PyTorch/data/sensor_graph/adj_mx.pkl",
    predictions: str = "examples/DCRNN_PyTorch/data/dcrnn_predictions_pytorch.npz",
    source: int = 0,
    target: int = 10,
    use_danskin: bool = False,
):
    dcrnn_root = os.path.abspath(dcrnn_root)
    adj_path = os.path.abspath(adj)
    pred_path = os.path.abspath(predictions)

    maybe_generate_adj(dcrnn_root, adj_path)

    sys.path.append(dcrnn_root)
    from lib.utils import load_graph_data

    _, _, adj_mx = load_graph_data(adj_path)
    G, edges = build_graph_from_adj(adj_mx)
    A = build_incidence(edges, adj_mx.shape[0])
    b = np.zeros(adj_mx.shape[0])
    b[source] = 1.0
    b[target] = -1.0

    # DCRNN predictions (speed) on held-out set
    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Predictions file not found at {pred_path}. Run run_demo_pytorch.py from the DCRNN_PyTorch repo to generate it."
        )
    speed_pred, speed_true = load_predictions(pred_path)

    # Work in cost space (travel time ~ inverse speed)
    cost_pred = 1.0 / np.maximum(speed_pred, 1e-3)
    cost_true = 1.0 / np.maximum(speed_true, 1e-3)

    # Split into calibration/test
    T = cost_pred.shape[0]
    split = int(0.6 * T)
    val_cost_pred, val_cost_true = cost_pred[:split], cost_true[:split]
    test_cost_pred, test_cost_true = cost_pred[split:], cost_true[split:]

    resid = val_cost_true - val_cost_pred
    resid_std = np.std(resid, axis=0, keepdims=True) + 1e-3

    def sampler(xb: torch.Tensor) -> torch.Tensor:
        base = xb.cpu().numpy()
        noise = np.random.normal(scale=resid_std.squeeze(), size=(K, base.shape[0], base.shape[1]))
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
    q = calibrator.calibrate(alpha=alpha)
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
    if use_danskin:
        l2_regions = [L2BallRegion(center=c, radius=radius) for c in centers_edges]
        union_region = UnionRegion(l2_regions)

        def inner_obj(theta_var, w_np):
            return theta_var @ w_np

        def value_and_grad(w_np, theta_np):
            return float(theta_np @ w_np), np.array(theta_np, dtype=float)

        # Project w onto flow constraints
        def project_flow(w_vec):
            E = A.shape[1]
            w_var = cp.Variable(E)
            prob = cp.Problem(cp.Minimize(cp.sum_squares(w_var - w_vec)), [A @ w_var == b, w_var >= 0, w_var <= 1])
            for solver in [cp.CLARABEL, cp.ECOS, None]:
                try:
                    prob.solve(solver=solver)
                except Exception:
                    continue
                if w_var.value is not None:
                    return w_var.value
            return w_vec

        w0 = mean_pred_edges
        danskin_opt = DanskinRobustOptimizer(
            region=union_region,
            inner_objective_fn=inner_obj,
            value_and_grad_fn=value_and_grad,
            project_fn=project_flow,
        )
        w_robust, _ = danskin_opt.solve(w0=np.asarray(w0).reshape(-1), step_size=0.1, max_iters=200)
        status_r = "danskin"
    else:
        w_robust, status_r = robust_shortest_path(A, b, centers_edges, radius, base_cost=np.ones_like(mean_pred_edges))
    w_nom, status_n = nominal_shortest_path(A, b, mean_pred_edges)

    if w_robust is None:
        print("Robust solve failed.")
        return {}
    w_robust = np.asarray(w_robust).reshape(-1)
    w_nom = np.asarray(w_nom).reshape(-1)

    print("Feasibility:")
    print(f"  nominal : {feasibility_report(A, b, w_nom)}")
    print(f"  analytic: {feasibility_report(A, b, w_robust)}")

    def worst_case_bound(w_vec):
        return float(np.max(centers_edges @ w_vec) + radius * np.linalg.norm(w_vec))

    print("Worst-case bound (analytic model):")
    print(f"  nominal : {worst_case_bound(w_nom):.4f}")
    print(f"  analytic: {worst_case_bound(w_robust):.4f}")

    # Evaluate on true cost for the same test sample
    true_cost_nodes = test_cost_true[0]
    true_cost_edges = node_costs_to_edges(true_cost_nodes, edges)
    robust_cost = true_cost_edges @ w_robust
    nominal_cost = true_cost_edges @ w_nom
    print(f"True path cost (nominal): {nominal_cost:.4f}")
    print(f"True path cost (analytic robust): {robust_cost:.4f}")

    # Visualize flows side by side
    pos = nx.spring_layout(G, seed=0, k=1 / np.sqrt(G.number_of_nodes()))

    def edge_widths(flow):
        return [max(flow[G[u][v]["idx"]] * 5, 0.1) for u, v in G.edges()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, flow, title in zip(
        axes,
        [w_nom, w_robust],
        ["Nominal flow", "Analytic robust flow"],
    ):
        nx.draw_networkx_nodes(G, pos, node_size=20, node_color="lightgray", ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths(flow), edge_color="C0", arrows=False, ax=ax)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("metrla_flows.png", dpi=150)
    plt.close(fig)
    print("Saved metrla_flows.png")
    return {
        "avg_cost_robust": float(robust_cost),
        "avg_cost_nominal": float(nominal_cost),
        "avg_cost_oracle": None,  # not available
        "q_calibrated": q,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcrnn-root", type=str, default="examples/DCRNN_PyTorch", help="Path to cloned DCRNN_PyTorch repo")
    parser.add_argument("--adj", type=str, default="examples/DCRNN_PyTorch/data/sensor_graph/adj_mx.pkl", help="Path to adj_mx.pkl")
    parser.add_argument("--predictions", type=str, default="examples/DCRNN_PyTorch/data/dcrnn_predictions_pytorch.npz", help="NPZ containing precomputed predictions/truth from run_demo_pytorch.py")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--use-danskin", action="store_true", help="Use Danskin gradient-based robust optimizer")
    args = parser.parse_args()
    run_experiment(
        alpha=args.alpha,
        K=args.K,
        dcrnn_root=args.dcrnn_root,
        adj=args.adj,
        predictions=args.predictions,
        source=args.source,
        target=args.target,
        use_danskin=args.use_danskin,
    )
