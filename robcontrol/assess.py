"""
Assessment script: load dataset + model, calibrate with operator-norm score,
and evaluate nominal vs CPC controllers using predicted dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robbuffet import OfflineDataset, OperatorNormScore, SplitConformalCalibrator
from robbuffet import vis
from robcontrol.controllers import CPCController
from robcontrol.data import TASKS, load_dataset
from robcontrol.utils import rollout_cost, solve_discrete_lqr


def build_loaders(thetas: np.ndarray, Cs: np.ndarray, batch_size: int, seed: int):
    n = thetas.shape[0]
    thetas = thetas.astype(np.float32)
    Cs = Cs.astype(np.float32)
    C_flat = Cs.reshape(n, -1)
    offline = OfflineDataset(thetas, C_flat, train_frac=0.6, cal_frac=0.2, seed=seed, shuffle=True)
    train_loader = DataLoader(offline.train, batch_size=batch_size, shuffle=True)
    cal_loader = DataLoader(offline.calibration, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(offline.test, batch_size=1, shuffle=False)
    return train_loader, cal_loader, test_loader


def load_model(model_path: Path, meta_path: Path) -> torch.nn.Module:
    from robcontrol.model import DynamicsMLP

    with open(meta_path, "r") as f:
        meta = json.load(f)
    model = DynamicsMLP(theta_dim=meta["theta_dim"], out_dim=meta["output_dim"])
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, meta


def compute_calibration_curve(model, score_fn, cal_loader, test_loader, mat_shape, alphas: np.ndarray):
    cal = SplitConformalCalibrator(model, score_fn, cal_loader)
    coverages: List[float] = []
    for alpha in alphas:
        cal.calibrate(alpha=float(alpha))
        hits = 0
        total = 0
        for theta, C_flat in test_loader:
            theta = theta.to(dtype=torch.float32)
            region = cal.predict_region(theta)
            C_true = C_flat.numpy().reshape(mat_shape)
            if region.contains(C_true):
                hits += 1
            total += 1
        coverages.append(hits / max(total, 1))
    return alphas, np.array(coverages)


def evaluate_once(
    model,
    task_meta,
    test_loader: Iterable,
    horizon: int,
    rollouts: int,
    process_noise_std: float = 0.0,
    control_noise_std: float = 0.0,
    seed: int = 0,
) -> Dict[str, float]:
    state_dim = task_meta["state_dim"]
    control_dim = task_meta["control_dim"]
    mat_shape = (state_dim, state_dim + control_dim)
    costs_true = []
    costs_hat_on_true = []
    costs_cpc_on_true = []
    cpc_ctrl = CPCController(np.array(task_meta["q"]), np.array(task_meta["r"]), config=None)
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for theta, C_flat in test_loader:
            theta = theta.to(dtype=torch.float32)
            C_true = C_flat.numpy().reshape(mat_shape)
            A_true = C_true[:, :state_dim]
            B_true = C_true[:, state_dim:]
            C_pred = model(theta).detach().numpy().reshape(mat_shape)
            A_hat = C_pred[:, :state_dim]
            B_hat = C_pred[:, state_dim:]
            K_true = solve_discrete_lqr(A_true, B_true, np.array(task_meta["q"]), np.array(task_meta["r"]))
            K_hat = solve_discrete_lqr(A_hat, B_hat, np.array(task_meta["q"]), np.array(task_meta["r"]))
            K_cpc = cpc_ctrl.synthesize(A_hat, B_hat, np.array(task_meta["q"]), np.array(task_meta["r"]))

            run_costs_true = []
            run_costs_hat_on_true = []
            run_costs_cpc_on_true = []
            for _ in range(rollouts):
                x0 = rng.normal(scale=0.1, size=state_dim)
                run_costs_true.append(
                    rollout_cost(
                        A_true,
                        B_true,
                        K_true,
                        np.array(task_meta["q"]),
                        np.array(task_meta["r"]),
                        horizon=horizon,
                        rng=rng,
                        x0=x0,
                        process_noise_std=process_noise_std,
                        control_noise_std=control_noise_std,
                    )
                )
                run_costs_hat_on_true.append(
                    rollout_cost(
                        A_true,
                        B_true,
                        K_hat,
                        np.array(task_meta["q"]),
                        np.array(task_meta["r"]),
                        horizon=horizon,
                        rng=rng,
                        x0=x0,
                        process_noise_std=process_noise_std,
                        control_noise_std=control_noise_std,
                    )
                )
                run_costs_cpc_on_true.append(
                    rollout_cost(
                        A_true,
                        B_true,
                        K_cpc,
                        np.array(task_meta["q"]),
                        np.array(task_meta["r"]),
                        horizon=horizon,
                        rng=rng,
                        x0=x0,
                        process_noise_std=process_noise_std,
                        control_noise_std=control_noise_std,
                    )
                )
            costs_true.append(np.mean(run_costs_true))
            costs_hat_on_true.append(np.mean(run_costs_hat_on_true))
            costs_cpc_on_true.append(np.mean(run_costs_cpc_on_true))

    return {
        "mean_cost_true_opt_on_true": float(np.mean(costs_true)),
        "mean_cost_hat_on_true": float(np.mean(costs_hat_on_true)),
        "mean_cost_cpc_on_true": float(np.mean(costs_cpc_on_true)),
    }


def evaluate(
    model,
    task_meta,
    test_loader: Iterable,
    horizon: int,
    rollouts: int,
    trials: int = 1,
    process_noise_std: float = 0.0,
    control_noise_std: float = 0.0,
) -> Dict[str, float]:
    robust_vals: List[float] = []
    nominal_vals: List[float] = []
    true_vals: List[float] = []
    for t in range(trials):
        stats = evaluate_once(
            model,
            task_meta,
            test_loader,
            horizon=horizon,
            rollouts=rollouts,
            process_noise_std=process_noise_std,
            control_noise_std=control_noise_std,
            seed=task_meta.get("seed", 0) + t,
        )
        true_vals.append(stats["mean_cost_true_opt_on_true"])
        nominal_vals.append(stats["mean_cost_hat_on_true"])
        robust_vals.append(stats["mean_cost_cpc_on_true"])
    metrics = {
        "true_opt_on_true": {"mean": float(np.mean(true_vals)), "std": float(np.std(true_vals))},
        "nominal_on_true": {"mean": float(np.mean(nominal_vals)), "std": float(np.std(nominal_vals))},
        "cpc_on_true": {"mean": float(np.mean(robust_vals)), "std": float(np.std(robust_vals))},
    }
    # Paired t-test (robust < nominal)
    try:
        from scipy import stats

        if trials > 1:
            t_stat, p_val = stats.ttest_rel(robust_vals, nominal_vals, alternative="less")
            metrics["paired_ttest_robust_lt_nominal"] = {"t": float(t_stat), "p": float(p_val)}
    except Exception:
        metrics["paired_ttest_robust_lt_nominal"] = {"t": None, "p": None}
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Assess nominal vs CPC controllers using predicted dynamics.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=5)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--process-noise-std", type=float, default=0.0)
    parser.add_argument("--control-noise-std", type=float, default=0.0)
    parser.add_argument("--calib-out", default=None, help="Path to save calibration plot.")
    parser.add_argument("--metrics-out", default=None, help="Path to save metrics JSON.")
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    thetas = data["thetas"]
    A_true = data["A_true"]
    B_true = data["B_true"]
    Cs = np.concatenate([A_true, B_true], axis=2)
    n = A_true.shape[1]
    m = B_true.shape[2]

    train_loader, cal_loader, test_loader = build_loaders(thetas, Cs, args.batch_size, args.seed)
    model, meta = load_model(Path(args.model), Path(args.meta))
    meta["q"] = data["q"]
    meta["r"] = data["r"]
    meta["seed"] = args.seed
    mat_shape = (n, n + m)
    score_fn = OperatorNormScore(state_dim=n, control_dim=m)
    alphas = np.linspace(0.01, 0.4, 8)
    alphas, coverages = compute_calibration_curve(model, score_fn, cal_loader, test_loader, mat_shape, alphas)

    fig, ax = plt.subplots()
    vis.plot_calibration_curve(alphas, coverages, ax=ax, label=data.get("task", "task"), title="Calibration")
    calib_out = Path(args.calib_out or Path(args.dataset).with_suffix("").as_posix() + "_calibration.png")
    calib_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(calib_out, bbox_inches="tight", dpi=150)
    print(f"Saved calibration plot to {calib_out}")

    metrics = evaluate(
        model,
        {**meta, "state_dim": n, "control_dim": m},
        test_loader,
        horizon=args.horizon,
        rollouts=args.rollouts,
        trials=args.trials,
        process_noise_std=args.process_noise_std,
        control_noise_std=args.control_noise_std,
    )
    metrics_out = Path(args.metrics_out or Path(args.dataset).with_suffix("").as_posix() + "_metrics.json")
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
