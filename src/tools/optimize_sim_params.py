import argparse
import json
import numpy as np
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any


def run_simulation(config: Dict[str, Any], project_root: Path):
    """Saves a temporary config and runs the simulation via chrysalis_cli."""
    tmp_config_path = project_root / "simulations" / "tmp_opt_config.json"
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    cmd = [
        sys.executable,
        str(project_root / "chrysalis_cli.py"),
        "simulate",
        str(tmp_config_path),
    ]
    print(f"[OPTIMIZER] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    exp_name = config["experiment_name"]
    results_dir = project_root / "simulations" / "results" / exp_name

    all_runs = list(results_dir.glob("run_*/results.json"))
    if not all_runs:
        raise FileNotFoundError(f"No results found in {results_dir}")

    all_runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    with open(all_runs[0], "r", encoding="utf-8") as f:
        return json.load(f)


def optimize_critical_point(
    base_config: Dict[str, Any],
    project_root: Path,
    target_metric: str = "susceptibility",
    iterations: int = 2,
):
    """Iteratively refines parameters to find the peak of a target metric."""

    current_config = base_config.copy()
    param_to_opt = "T" if "T" in base_config["parameter_sweep"] else "p"

    peak_x = None

    for i in range(iterations):
        print(f"\n[OPTIMIZER] --- Iteration {i+1}/{iterations} ---")
        current_config["experiment_name"] = (
            f"Opt_{base_config['experiment_name']}_Iter_{i}"
        )

        # 1. Run simulation
        results = run_simulation(current_config, project_root)

        # 2. Extract metric and find peak
        main_res = results["main_results"]
        x_vals = np.array(main_res[param_to_opt])
        y_vals = np.array(main_res[target_metric])

        peak_idx = np.argmax(y_vals)
        peak_x = x_vals[peak_idx]
        peak_y = y_vals[peak_idx]

        print(
            f"[OPTIMIZER] Current Peak: {param_to_opt} = {peak_x:.4f}, {target_metric} = {peak_y:.4f}"
        )

        # 3. Refine range
        if len(x_vals) > 1:
            # Simple window-based refinement
            span = x_vals.max() - x_vals.min()

            # Narrow the window by 50% around the peak
            new_span = span * 0.5
            new_min = max(x_vals.min(), peak_x - new_span / 2)
            new_max = min(x_vals.max(), peak_x + new_span / 2)

            new_sweep = np.linspace(new_min, new_max, 10).tolist()
            current_config["parameter_sweep"][param_to_opt] = new_sweep
        else:
            break

    print(
        f"\n[OPTIMIZER] Optimization Finished! Final suggested {param_to_opt} = {peak_x:.4f}"
    )
    return peak_x


def main():
    parser = argparse.ArgumentParser(
        description="Auto-optimize simulation parameters to find criticality."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the base simulation config."
    )
    parser.add_argument(
        "--target", type=str, default="susceptibility", help="Metric to maximize."
    )
    parser.add_argument(
        "--iterations", type=int, default=2, help="Number of refinement iterations."
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    config_path = Path(args.config_file)
    if not config_path.exists():
        config_path = project_root / args.config_file
        if not config_path.exists():
            print(f"Error: Config file {args.config_file} not found.")
            exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    optimal_val = optimize_critical_point(
        base_config, project_root, args.target, args.iterations
    )

    # Save the final result
    opt_report_path = (
        project_root / "research" / f"opt_report_{base_config['experiment_name']}.json"
    )
    with open(opt_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "optimal_parameter_value": float(optimal_val),
                "target_metric": args.target,
            },
            f,
            indent=4,
        )
    print(f"[OPTIMIZER] Optimization report saved to: {opt_report_path}")


if __name__ == "__main__":
    main()
