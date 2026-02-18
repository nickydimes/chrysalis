import argparse
import json
import os
import importlib
import importlib.metadata
from pathlib import Path
import itertools
from datetime import datetime
import uuid
import sys
import subprocess
import platform

# Define the path to the central experiment log
EXPERIMENT_LOG_PATH = Path("simulations/experiment_log.json")


def _get_environment_metadata() -> dict:
    """Collects comprehensive environment and code state metadata."""
    metadata = {
        "python_version": sys.version,
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "library_versions": {},
    }

    # Get Git commit hash
    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent),
            )
            .decode("ascii")
            .strip()
        )
        metadata["git_commit"] = git_hash
    except Exception:
        metadata["git_commit"] = "unknown (not a git repo or git not found)"

    # Get versions of key libraries
    libraries = [
        "numpy",
        "scipy",
        "numba",
        "pandas",
        "google-generativeai",
        "requests",
        "langchain",
        "chromadb",
    ]
    for lib in libraries:
        try:
            metadata["library_versions"][lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            metadata["library_versions"][lib] = "not installed"

    return metadata


def _load_experiment_log() -> list:
    """Loads the central experiment log."""
    if EXPERIMENT_LOG_PATH.exists():
        with open(EXPERIMENT_LOG_PATH, "r") as f:
            return json.load(f)
    return []


def _save_experiment_log(log_data: list):
    """Saves the central experiment log."""
    with open(EXPERIMENT_LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Run a simulation experiment based on a configuration file."
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the JSON configuration file for the experiment.",
    )

    args = parser.parse_args()

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found.")
        exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"DEBUG: Loaded config: {config}")

    experiment_name = config.get("experiment_name", "untitled_experiment")
    simulation_module_name = config.get("simulation_module")
    output_base_dir = Path(
        config.get("output_base_dir", f"simulations/results/{experiment_name}")
    )
    global_params = config.get("global_params", {})
    parameter_sweep = config.get("parameter_sweep", {})
    num_seeds = config.get("num_seeds", 1)
    seed_offset = config.get("seed_offset", 0)

    if not simulation_module_name:
        print("Error: 'simulation_module' must be specified in the config file.")
        exit(1)

    # --- Experiment Tracking Initialization ---
    experiment_id = str(uuid.uuid4())
    start_time_iso = datetime.now().isoformat()
    env_metadata = _get_environment_metadata()

    experiment_record = {
        "experiment_id": experiment_id,
        "name": experiment_name,
        "simulation_module": simulation_module_name,
        "config_file": str(config_path),
        "output_base_dir": str(output_base_dir),
        "global_params": global_params,
        "parameter_sweep_definition": parameter_sweep,
        "num_seeds": num_seeds,
        "seed_offset": seed_offset,
        "start_time": start_time_iso,
        "end_time": None,
        "status": "running",
        "environment_metadata": env_metadata,
        "runs": [],  # List to store individual run details
    }

    experiment_log = _load_experiment_log()
    # Find and remove any existing record for this experiment_id if it was partially written
    experiment_log = [
        rec for rec in experiment_log if rec.get("experiment_id") != experiment_id
    ]
    experiment_log.append(experiment_record)
    _save_experiment_log(experiment_log)
    print(
        f"Experiment {experiment_id} '{experiment_name}' started at {start_time_iso}."
    )
    print(f"Tracking in: {EXPERIMENT_LOG_PATH}")
    # --- End Experiment Tracking Initialization ---

    print(f"Starting experiment: {experiment_name}")
    print(f"Simulation: {simulation_module_name}")
    print(f"Output will be saved in: {output_base_dir}")

    # Ensure output base directory exists
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically import the simulation module
    try:
        # Add the parent directory of 'simulations' to sys.path
        # so 'simulations.phase_transitions' can be imported
        simulations_path = os.path.abspath(os.path.dirname(__file__))
        project_root = os.path.abspath(
            os.path.join(simulations_path, "..", "..")
        )  # Go up to chrysalis root
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        module_path = f"simulations.phase_transitions.{simulation_module_name}"
        simulation_module = importlib.import_module(module_path)
        if not hasattr(simulation_module, "run") or not callable(simulation_module.run):
            print(
                f"Error: Simulation module '{simulation_module_name}' does not have a 'run' function."
            )
            # Mark experiment as failed in log
            experiment_record["status"] = "failed"
            experiment_record["error_message"] = (
                f"Simulation module '{simulation_module_name}' does not have a 'run' function."
            )
            experiment_record["end_time"] = datetime.now().isoformat()
            _save_experiment_log(experiment_log)
            exit(1)
    except ImportError as e:
        print(f"Error: Simulation module '{simulation_module_name}' not found: {e}")
        # Mark experiment as failed in log
        experiment_record["status"] = "failed"
        experiment_record["error_message"] = str(e)
        experiment_record["end_time"] = datetime.now().isoformat()
        _save_experiment_log(experiment_log)
        exit(1)

    # Prepare parameter sweep combinations
    sweep_keys = list(parameter_sweep.keys())
    sweep_values = list(parameter_sweep.values())
    param_combinations = list(itertools.product(*sweep_values))

    total_runs = len(param_combinations) * num_seeds
    current_run_idx = 0

    try:
        for combo in param_combinations:
            current_params = dict(zip(sweep_keys, combo))

            for seed_idx in range(num_seeds):
                current_run_idx += 1
                seed = seed_offset + seed_idx

                # Construct run-specific output directory
                run_output_dir_name_parts = [f"run_{current_run_idx}", f"seed_{seed}"]
                # Filter out values that are lists from current_params for directory naming if they make it too long
                for k, v in current_params.items():
                    if not isinstance(
                        v, list
                    ):  # Only include non-list params directly in name
                        run_output_dir_name_parts.append(
                            f"{k}_{str(v).replace('.', 'p').replace('[', '').replace(']', '').replace(',', '-')}"
                        )

                run_output_dir_name = "_".join(run_output_dir_name_parts)

                run_output_path = output_base_dir / run_output_dir_name
                run_output_path.mkdir(parents=True, exist_ok=True)

                print(
                    f"\n--- Running: {experiment_name} | Run {current_run_idx}/{total_runs} ---"
                )
                print(
                    f"  Params: {current_params}, Global: {global_params}, Seed: {seed}"
                )
                print(f"  Output: {run_output_path}")

                # Prepare arguments for the simulation's run() function
                sim_args_list = []
                for k, v in global_params.items():
                    sim_args_list.extend([f"--{k}", str(v)])
                for k, v in current_params.items():
                    if isinstance(v, list):  # Handle list arguments like L_values
                        sim_args_list.extend([f"--{k}"] + [str(x) for x in v])
                    else:
                        sim_args_list.extend([f"--{k}", str(v)])

                # Add seed and output_dir to arguments
                sim_args_list.extend(["--seed", str(seed)])
                sim_args_list.extend(
                    ["--output_dir", str(run_output_path)]
                )  # Pass output_dir to simulation
                sim_args_list.append("--no_scaling")

                run_start_time = datetime.now().isoformat()
                run_status = "completed"
                run_error = None
                try:
                    # Call the simulation's run function with prepared arguments
                    simulation_module.run(sim_args_list)
                    print(f"Run {current_run_idx} completed successfully.")
                except Exception as e:
                    run_status = "failed"
                    run_error = str(e)
                    print(f"Error in run {current_run_idx}: {e}")
                    # Optionally, clean up the failed run's directory
                    # shutil.rmtree(run_output_path)

                # Add individual run details to the experiment record
                run_record = {
                    "run_id": str(uuid.uuid4()),
                    "start_time": run_start_time,
                    "parameters": {**global_params, **current_params},
                    "seed": seed,
                    "output_path": str(run_output_path),
                    "status": run_status,
                    "error_message": run_error,
                    "end_time": datetime.now().isoformat(),
                }
                experiment_record["runs"].append(run_record)

                # Save log after each run to ensure progress is tracked
                # Find the record by experiment_id and update it
                # (This is inefficient for very large logs, but fine for typical experiment sizes)
                current_log = _load_experiment_log()
                for i, rec in enumerate(current_log):
                    if rec.get("experiment_id") == experiment_id:
                        current_log[i] = experiment_record
                        break
                _save_experiment_log(current_log)

        experiment_record["status"] = "completed"
        print(
            f"\nExperiment '{experiment_name}' finished. Total runs: {current_run_idx}"
        )

    except Exception as e:
        experiment_record["status"] = "failed"
        experiment_record["error_message"] = str(e)
        print(f"\nExperiment '{experiment_name}' failed unexpectedly: {e}")
    finally:
        experiment_record["end_time"] = datetime.now().isoformat()
        # Final save of the log with the overall experiment status
        current_log = _load_experiment_log()
        for i, rec in enumerate(current_log):
            if rec.get("experiment_id") == experiment_id:
                current_log[i] = experiment_record
                break
        _save_experiment_log(current_log)


if __name__ == "__main__":
    # Ensure sys.path is correct for imports if run directly
    project_root_str = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    main()
