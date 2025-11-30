"""
Run an example script multiple times and summarize robust vs nominal performance.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple


def summarize(values: List[float]) -> Tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std


def cache_key(
    example: str,
    cwd: str,
    alpha: float,
    trials: int,
) -> str:
    return json.dumps(
        {
            "example": example,
            "cwd": os.path.abspath(cwd),
            "alpha": alpha,
            "trials": trials,
        },
        sort_keys=True,
    )


def load_cache(path: str) -> Dict[str, List[Dict[str, float]]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(path: str, data: Dict[str, List[Dict[str, float]]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_experiment_module(example: str, cwd: str, alpha: float) -> Tuple[float, float, float]:
    """
    Import the example module and call run_experiment() to get costs and oracle.
    """
    sys.path.insert(0, cwd)
    mod_path = os.path.join(cwd, "examples", example)
    spec = importlib.util.spec_from_file_location("example_run", mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module at {mod_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if not hasattr(module, "run_experiment"):
        raise AttributeError(f"{example} does not expose run_experiment()")
    results = module.run_experiment(alpha=alpha)
    oracle = results.get("avg_cost_oracle")
    robust = results.get("avg_cost_robust")
    nominal = results.get("avg_cost_nominal")
    if robust is None or nominal is None:
        raise ValueError("run_experiment did not return required cost fields")
    return float(robust), float(nominal), float(oracle) if oracle is not None else None


def main():
    parser = argparse.ArgumentParser(description="Run an example multiple times and report relative suboptimality stats.")
    parser.add_argument("--example", required=True, help="Name of example script under examples/ that exposes run_experiment()")
    parser.add_argument("--trials", type=int, default=5, help="Number of runs")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha to pass to run_experiment (if supported)")
    parser.add_argument("--relative", action="store_true", help="Report relative gaps using avg_cost_oracle when available")
    parser.add_argument("--cwd", default=".", help="Working directory for the command")
    parser.add_argument("--cache", default=".cache/run_trials.json", help="Cache file to reuse previous runs")
    args = parser.parse_args()

    if not os.path.isdir(args.cwd):
        raise SystemExit(f"Working directory does not exist: {args.cwd}")

    cache = load_cache(args.cache)
    key = cache_key(args.example, args.cwd, args.alpha, args.trials)
    cached_runs = cache.get(key, [])

    robust_vals: List[float] = [r["robust"] for r in cached_runs[: args.trials]]
    nominal_vals: List[float] = [r["nominal"] for r in cached_runs[: args.trials]]
    oracle_vals: List[Optional[float]] = [r.get("oracle") for r in cached_runs[: args.trials]]

    needed = max(args.trials - len(robust_vals), 0)
    if needed > 0:
        print(f"Running {needed} additional trial(s); {len(robust_vals)} cached.")
    for i in range(needed):
        try:
            r, n, _ = run_experiment_module(args.example, args.cwd, alpha=args.alpha)
        except Exception as e:
            raise SystemExit(f"Trial {len(robust_vals) + 1} failed: {e}")
        robust_vals.append(r)
        nominal_vals.append(n)
        oracle_vals.append(_)
        cached_runs.append({"robust": r, "nominal": n, "oracle": _})
        print(f"Trial {len(robust_vals)}: robust_rel={r:.4f}, nominal_rel={n:.4f}")

    cache[key] = cached_runs
    save_cache(args.cache, cache)

    # Paired t-test (robust < nominal) if scipy is available
    t_stat = p_value = None
    try:
        import numpy as np
        from scipy import stats
        if len(robust_vals) > 1:
            t_stat, p_value = stats.ttest_rel(robust_vals, nominal_vals, alternative="less")
    except Exception:
        pass

    if args.relative:
        valid = [(r, n, o) for r, n, o in zip(robust_vals, nominal_vals, oracle_vals) if o not in (None, 0)]
        if not valid:
            raise SystemExit("No valid oracle values to compute relative gaps.")
        rel_r = [(r - o) / o for r, _, o in valid]
        rel_n = [(n - o) / o for _, n, o in valid]
        r_mean, r_std = summarize(rel_r)
        n_mean, n_std = summarize(rel_n)
        print("\nRelative gap summary (mean ± std):")
        print(f"  Robust : {r_mean:.4f} ± {r_std:.4f}")
        print(f"  Nominal: {n_mean:.4f} ± {n_std:.4f}")
    else:
        r_mean, r_std = summarize(robust_vals)
        n_mean, n_std = summarize(nominal_vals)
        print("\nObjective summary (mean ± std):")
        print(f"  Robust : {r_mean:.4f} ± {r_std:.4f}")
        print(f"  Nominal: {n_mean:.4f} ± {n_std:.4f}")

    if t_stat is not None and p_value is not None:
        print(f"\nPaired t-test (robust < nominal): t={t_stat:.4f}, p={p_value:.4g}")
    else:
        print("\nPaired t-test unavailable (scipy missing or insufficient samples).")


if __name__ == "__main__":
    main()
