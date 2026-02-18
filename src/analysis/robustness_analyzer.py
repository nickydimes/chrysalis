import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Chrysalis imports
from chrysalis.simulations.aggregate_results import (
    load_experiment_results,
    group_runs_by_params,
)


def evaluate_success(run_data: Dict[str, Any], threshold: float = 0.8) -> bool:
    """Determines if a simulation run reached an emergent state."""
    main_res = run_data["main_results"]
    order_param = main_res.get("magnetization", main_res.get("order_param"))
    if order_param is None:
        return False

    final_val = (
        np.abs(order_param[-1])
        if isinstance(order_param, list)
        else np.abs(order_param)
    )
    return final_val >= threshold


def calculate_robustness_metrics(base_dir: Path, threshold: float = 0.8):
    """Calculates success probability and tipping points from an experiment."""
    all_runs = load_experiment_results(base_dir)
    if not all_runs:
        return None, []

    all_keys = set()
    for r in all_runs:
        all_keys.update(r["params"].keys())

    sweep_keys = [
        k
        for k in all_keys
        if k
        not in [
            "run_seed",
            "seed",
            "output_dir",
            "no_scaling",
            "L_values",
            "n_autocorr_sweeps",
            "eq_sweeps",
            "meas_sweeps",
        ]
    ]

    grouped = group_runs_by_params(all_runs, sweep_keys)

    summary_data = []
    for params_tuple, runs in grouped.items():
        params_dict = dict(zip(sweep_keys, params_tuple))
        successes = [evaluate_success(r, threshold) for r in runs]
        success_prob = np.mean(successes)

        row = params_dict.copy()
        row["success_probability"] = success_prob
        row["num_runs"] = len(runs)
        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    return df, sweep_keys


def plot_robustness_heatmap(df: pd.DataFrame, sweep_keys: List[str], output_path: Path):
    """Generates a 2D heatmap of success probabilities."""
    if not sweep_keys:
        return

    if len(sweep_keys) < 2:
        # 1D Plot
        plt.figure(figsize=(10, 6))
        x_key = sweep_keys[0]
        df_sorted = df.sort_values(x_key)
        plt.plot(
            df_sorted[x_key],
            df_sorted["success_probability"],
            "o-",
            linewidth=2,
            color="teal",
        )
        plt.axhline(0.5, color="red", linestyle="--", label="Tipping Point (50%)")
        plt.xlabel(x_key, fontweight="bold")
        plt.ylabel("Success Probability", fontweight="bold")
        plt.title(f"Protocol Robustness: {x_key}", fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()
    else:
        # 2D Heatmap
        x_key, y_key = sweep_keys[0], sweep_keys[1]
        try:
            pivot_df = df.pivot(
                index=y_key, columns=x_key, values="success_probability"
            )

            plt.figure(figsize=(12, 8))
            im = plt.imshow(
                pivot_df,
                origin="lower",
                aspect="auto",
                extent=[
                    df[x_key].min(),
                    df[x_key].max(),
                    df[y_key].min(),
                    df[y_key].max(),
                ],
                cmap="RdYlGn",
            )
            plt.colorbar(im, label="Success Probability")
            plt.xlabel(x_key, fontweight="bold")
            plt.ylabel(y_key, fontweight="bold")
            plt.title(
                f"Phase Diagram of Protocol Success ({x_key} vs {y_key})",
                fontweight="bold",
            )
        except Exception as e:
            print(f"Warning: Could not generate heatmap: {e}")
            return

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Robustness plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the robustness of the Eight-Step Protocol."
    )
    parser.add_argument(
        "experiment_dir", type=str, help="Path to the experiment results."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Order parameter threshold for 'Success'.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the robustness report.",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    base_dir = Path(args.experiment_dir)
    if not base_dir.exists():
        base_dir = project_root / args.experiment_dir

    if not base_dir.exists():
        print(f"Error: Experiment directory {base_dir} not found.")
        exit(1)

    print(f"Analyzing robustness for: {base_dir.name}")
    df, sweep_keys = calculate_robustness_metrics(base_dir, args.threshold)

    if df is None or df.empty:
        print("No valid results found to analyze.")
        exit(0)

    # Save CSV report
    report_dir = project_root / "research" / "robustness_reports" / base_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "robustness_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Robustness data saved to {csv_path}")

    # Plot
    plot_path = report_dir / "robustness_diagram.png"
    plot_robustness_heatmap(df, sweep_keys, plot_path)

    # Summary
    print("\n--- Robustness Summary ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
