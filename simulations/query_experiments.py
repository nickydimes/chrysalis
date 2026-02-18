import argparse
import json
from pathlib import Path

# Define the path to the central experiment log
EXPERIMENT_LOG_PATH = Path("chrysalis/simulations/experiment_log.json")


def _load_experiment_log() -> list:
    """Loads the central experiment log."""
    if EXPERIMENT_LOG_PATH.exists():
        with open(EXPERIMENT_LOG_PATH, "r") as f:
            return json.load(f)
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Query and display simulation experiment tracking data."
    )
    parser.add_argument(
        "--name", type=str, help="Filter by experiment name (substring match)."
    )
    parser.add_argument(
        "--status",
        type=str,
        choices=["running", "completed", "failed"],
        help="Filter by experiment status.",
    )
    parser.add_argument("--id", type=str, help="Filter by exact experiment ID.")
    parser.add_argument(
        "--show_runs",
        action="store_true",
        help="Display details for individual runs within matching experiments.",
    )
    parser.add_argument(
        "--show_env",
        action="store_true",
        help="Display environment and code state metadata.",
    )

    args = parser.parse_args()

    experiment_log = _load_experiment_log()

    if not experiment_log:
        print("No experiment data found in log.")
        exit(0)

    filtered_experiments = []
    for exp in experiment_log:
        match = True
        if args.id and exp.get("experiment_id") != args.id:
            match = False
        if args.name and args.name.lower() not in exp.get("name", "").lower():
            match = False
        if args.status and exp.get("status") != args.status:
            match = False

        if match:
            filtered_experiments.append(exp)

    if not filtered_experiments:
        print("No experiments found matching the criteria.")
        exit(0)

    print(f"Found {len(filtered_experiments)} experiment(s) matching criteria:")
    print("=" * 60)

    for exp in filtered_experiments:
        print(f"Experiment ID: {exp.get('experiment_id')}")
        print(f"Name: {exp.get('name')}")
        print(f"Simulation: {exp.get('simulation_module')}")
        print(f"Status: {exp.get('status')}")
        print(f"Start Time: {exp.get('start_time')}")
        print(f"End Time: {exp.get('end_time', 'N/A')}")
        print(f"Output Base Dir: {exp.get('output_base_dir')}")
        print(f"Total Runs: {len(exp.get('runs', []))}")
        print(f"Config File: {exp.get('config_file', 'N/A')}")

        if args.show_env and exp.get("environment_metadata"):
            env = exp["environment_metadata"]
            print("  --- Environment Metadata ---")
            print(f"    Python Version: {env.get('python_version')}")
            print(f"    OS: {env.get('os')}")
            print(f"    Git Commit: {env.get('git_commit')}")
            print("    Library Versions:")
            for lib, ver in env.get("library_versions", {}).items():
                print(f"      {lib}: {ver}")
            print("  --- End Environment Metadata ---")

        if args.show_runs and exp.get("runs"):
            print("  --- Individual Runs ---")
            for run in exp["runs"]:
                print(f"    Run ID: {run.get('run_id')}")
                print(f"    Status: {run.get('status')}")
                print(f"    Seed: {run.get('seed')}")
                print(f"    Parameters: {run.get('parameters')}")
                print(f"    Output Path: {run.get('output_path')}")
                if run.get("error_message"):
                    print(f"    Error: {run.get('error_message')}")
                print("    " + "-" * 20)
            print("  --- End Individual Runs ---")
        print("=" * 60)


if __name__ == "__main__":
    main()
