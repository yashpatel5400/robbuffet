"""
Run an example script multiple times and summarize relative suboptimality.

Relative suboptimality = (objective - true_objective) / true_objective.

Usage:
  python scripts/run_trials.py \\
      --command "PYTHONPATH=. python examples/robust_shortest_path_metrla.py" \\
      --trials 5 \\
      --robust-pattern "True path cost \\(analytic robust\\): ([\\d\\.eE+-]+)" \\
      --nominal-pattern "True path cost \\(nominal\\): ([\\d\\.eE+-]+)" \\
      --true-pattern "True path cost \\(nominal\\): ([\\d\\.eE+-]+)"

You can set robust/nominal/true patterns to match the lines printed by your example.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
from typing import Dict, List, Tuple


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


def run_experiment_module(example: str, cwd: str) -> Tuple[float, float, float]:
    """
    Import the example module and call run_experiment() to get relative gaps and oracle.
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
    results = module.run_experiment()
    oracle = results.get("avg_cost_oracle")
    robust = results.get("avg_cost_robust")
    nominal = results.get("avg_cost_nominal")
    if oracle is None or robust is None or nominal is None:
        raise ValueError("run_experiment did not return required cost fields")
    if oracle == 0:
        raise ValueError("Oracle objective is zero; relative suboptimality undefined.")
    robust_rel = (robust - oracle) / oracle
    nominal_rel = (nominal - oracle) / oracle
    return robust_rel, nominal_rel, oracle


def main():
    parser = argparse.ArgumentParser(description="Run an example multiple times and report relative suboptimality stats.")
    parser.add_argument("--example", required=True, help="Name of example script under examples/ that exposes run_experiment()")
    parser.add_argument("--trials", type=int, default=5, help="Number of runs")
    parser.add_argument("--alpha", type=float, default=0.1, help="Alpha to pass to run_experiment (if supported)")
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

    needed = max(args.trials - len(robust_vals), 0)
    if needed > 0:
        print(f"Running {needed} additional trial(s); {len(robust_vals)} cached.")
    for i in range(needed):
        try:
            r, n, _ = run_experiment_module(args.example, args.cwd)
        except Exception as e:
            raise SystemExit(f"Trial {len(robust_vals) + 1} failed: {e}")
        robust_vals.append(r)
        nominal_vals.append(n)
        cached_runs.append({"robust": r, "nominal": n})
        print(f"Trial {len(robust_vals)}: robust_rel={r:.4f}, nominal_rel={n:.4f}")

    cache[key] = cached_runs
    save_cache(args.cache, cache)

    r_mean, r_std = summarize(robust_vals)
    n_mean, n_std = summarize(nominal_vals)

    print("\nRelative suboptimality summary (mean ± std):")
    print(f"  Robust : {r_mean:.4f} ± {r_std:.4f}")
    print(f"  Nominal: {n_mean:.4f} ± {n_std:.4f}")


if __name__ == "__main__":
    main()
