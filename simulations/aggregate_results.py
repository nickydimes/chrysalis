import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(base_dir: Path) -> List[Dict[str, Any]]:
    """
    Loads all 'results.json' files from subdirectories within a given base experiment directory.
    Assumes each subdirectory corresponds to a single simulation run.
    """
    all_runs_data = []

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return []

    for run_dir in base_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            results_json_path = run_dir / "results.json"
            if results_json_path.exists():
                try:
                    with open(results_json_path, "r", encoding="utf-8") as f:
                        run_data = json.load(f)

                    # Add run-specific metadata from directory name
                    match = re.search(r"seed_(\d+)", run_dir.name)
                    if match:
                        run_data["params"]["run_seed"] = int(match.group(1))

                    # Store the path to the results for later reference
                    run_data["run_path"] = str(run_dir)

                    all_runs_data.append(run_data)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Could not decode JSON from {results_json_path}: {e}"
                    )
                except Exception as e:
                    print(f"Warning: Error loading data from {results_json_path}: {e}")
            else:
                print(f"Warning: No results.json found in {run_dir}")
    return all_runs_data


def group_runs_by_params(
    runs_data: List[Dict[str, Any]], group_keys: List[str]
) -> Dict[Tuple, List[Dict[str, Any]]]:
    """
    Groups simulation runs by a specified set of parameter keys.
    """
    grouped_data = {}
    for run in runs_data:
        key_values = tuple(run["params"].get(k) for k in group_keys)
        if key_values not in grouped_data:
            grouped_data[key_values] = []
        grouped_data[key_values].append(run)
    return grouped_data


def find_critical_temperature(T_values, susceptibility_values) -> Tuple[float, float]:
    """
    Estimates critical temperature and peak susceptibility from susceptibility data.
    """
    if not T_values or not susceptibility_values:
        return np.nan, np.nan
    max_idx = np.argmax(susceptibility_values)
    return T_values[max_idx], susceptibility_values[max_idx]


def plot_binder_cumulant_all_L(runs_data: List[Dict[str, Any]], output_path: Path):
    """
    Generates a combined Binder Cumulant plot for different L values.
    """
    plt.figure(figsize=(10, 7))
    plt.title("Binder Cumulant U_L vs Sweep Parameter")

    # Logic to handle different x-axes
    x_label = "Parameter"

    processed_labels = []
    for run_data in runs_data:
        params = run_data["params"]
        N = params.get("N", params.get("L", "unknown"))

        main_res = run_data["main_results"]
        x_key = "T" if "T" in main_res else "p"
        x_label = "Temperature (T)" if x_key == "T" else "Occupation Probability (p)"

        x_vals = np.array(main_res[x_key])
        m2 = np.array(main_res["m2"])
        m4 = np.array(main_res["m4"])

        # Avoid division by zero
        U_L = np.where(m2 != 0, 1 - m4 / (3 * m2**2), np.nan)

        label = f"Size={N}"
        if label not in processed_labels:
            plt.plot(x_vals, U_L, marker="o", linestyle="-", markersize=4, label=label)
            processed_labels.append(label)

    plt.xlabel(x_label)
    plt.ylabel(r"$U_L = 1 - \langle m^4 \rangle / (3\langle m^2 \rangle^2)$")
    plt.legend(title="Lattice Size")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Binder Cumulant plot to {output_path}")


def plot_protocol_signature(run_data: Dict[str, Any], output_path: Path):
    """
    Plots the 'Protocol Signature' - all 8 protocol metrics over the sweep parameter.
    """
    if "protocol_metrics" not in run_data:
        print(
            "Warning: No protocol_metrics found in run data. Skipping signature plot."
        )
        return

    metrics = run_data["protocol_metrics"]
    main_res = run_data["main_results"]

    x_param = "T" if "T" in main_res else "p"
    x_values = main_res[x_param]
    x_label = "Temperature (T)" if x_param == "T" else "Occupation Probability (p)"

    plt.figure(figsize=(12, 8))
    size_val = run_data["params"].get("N", run_data["params"].get("L", "Unknown"))
    plt.title(f"Eight-Step Protocol Signature (Size={size_val})", fontweight="bold")

    for step, values in metrics.items():
        plt.plot(x_values, values, label=step, linewidth=2, alpha=0.8)

    plt.xlabel(x_label, fontweight="bold")
    plt.ylabel("Normalized Metric Value", fontweight="bold")
    plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Protocol Signature plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and report simulation results."
    )
    parser.add_argument(
        "experiment_base_dir",
        type=str,
        help="Path to the base directory of an experiment.",
    )
    parser.add_argument(
        "--output_report_dir",
        type=str,
        default="chrysalis/simulations/reports",
        help="Directory to save aggregated reports and plots.",
    )

    args = parser.parse_args()

    base_dir = Path(args.experiment_base_dir)
    output_report_dir = Path(args.output_report_dir) / base_dir.name
    output_report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {base_dir}")
    all_runs_data = load_experiment_results(base_dir)
    print(f"Loaded {len(all_runs_data)} individual run results.")

    if not all_runs_data:
        print("No simulation run data found to aggregate.")
        exit(0)

    all_param_keys = set()
    for run in all_runs_data:
        all_param_keys.update(run["params"].keys())

    grouping_keys = [
        k
        for k in all_param_keys
        if k
        not in [
            "run_seed",
            "seed",
            "L_values",
            "n_autocorr_sweeps",
            "T_values",
            "p_values",
        ]
    ]
    grouped_by_common_params = group_runs_by_params(all_runs_data, grouping_keys)

    print(f"Aggregating and generating plots in {output_report_dir}...")

    # Plot Binder Cumulant
    plot_binder_cumulant_all_L(
        all_runs_data, output_report_dir / "aggregated_binder_cumulant.png"
    )

    # Plot Protocol Signature for each group
    for params_tuple, runs in grouped_by_common_params.items():
        group_name = "_".join(f"{k}_{v}" for k, v in zip(grouping_keys, params_tuple))
        if runs:
            plot_protocol_signature(
                runs[0], output_report_dir / f"protocol_signature_{group_name}.png"
            )

    # Summary report
    report_path = output_report_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Simulation Aggregation Report for Experiment: {base_dir.name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total individual runs loaded: {len(all_runs_data)}\n\n")

        for params_tuple, runs in grouped_by_common_params.items():
            f.write(f"Group: {dict(zip(grouping_keys, params_tuple))}\n")
            f.write(f"  Runs in group: {len(runs)}\n")

            if runs and "main_results" in runs[0]:
                main_res = runs[0]["main_results"]
                x_key = "T" if "T" in main_res else "p"

                x_vals_group = []
                sus_vals_group = []

                for run in runs:
                    if (
                        "main_results" in run
                        and x_key in run["main_results"]
                        and "susceptibility" in run["main_results"]
                    ):
                        x_vals_group.extend(run["main_results"][x_key])
                        sus_vals_group.extend(run["main_results"]["susceptibility"])

                if x_vals_group:
                    crit_val, peak_sus = find_critical_temperature(
                        x_vals_group, sus_vals_group
                    )
                    label = "T_c" if x_key == "T" else "p_c"
                    f.write(
                        f"  Estimated average {label} (from max susceptibility): {crit_val:.3f}\n"
                    )
                    f.write(f"  Peak susceptibility: {peak_sus:.3f}\n")
            f.write("\n")

    print(f"Summary report generated: {report_path}")


if __name__ == "__main__":
    main()
