import sqlite3
import json
import argparse
from pathlib import Path
from chrysalis.src.tools.init_db import init_db

def sync_experiments(db_path: Path, log_path: Path):
    """Syncs experiment logs from JSON to database."""
    if not log_path.exists():
        print(f"Log file {log_path} not found. Skipping.")
        return

    print(f"Syncing experiments from {log_path}...")
    with open(log_path, 'r') as f:
        log_data = json.load(f)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for exp in log_data:
        env = exp.get("environment_metadata", {})
        cursor.execute("""
        INSERT OR REPLACE INTO experiments (
            experiment_id, name, simulation_module, config_file, output_base_dir,
            start_time, end_time, status, git_commit, python_version, os_info, global_params_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp.get("experiment_id"), exp.get("name"), exp.get("simulation_module"),
            exp.get("config_file"), exp.get("output_base_dir"), exp.get("start_time"),
            exp.get("end_time"), exp.get("status"), env.get("git_commit"),
            env.get("python_version"), env.get("os"), json.dumps(exp.get("global_params", {}))
        ))

        for run in exp.get("runs", []):
            cursor.execute("""
            INSERT OR REPLACE INTO simulation_runs (
                run_id, experiment_id, seed, status, start_time, end_time,
                output_path, parameters_json, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.get("run_id"), exp.get("experiment_id"), run.get("seed"),
                run.get("status"), run.get("start_time"), run.get("end_time"),
                run.get("output_path"), json.dumps(run.get("parameters", {})),
                run.get("error_message")
            ))

    conn.commit()
    conn.close()
    print("Experiments synced successfully.")

def sync_ethnographic(db_path: Path, ethno_dir: Path):
    """Syncs ethnographic records from JSON files to database."""
    if not ethno_dir.exists():
        print(f"Ethnographic directory {ethno_dir} not found. Skipping.")
        return

    print(f"Syncing ethnographic records from {ethno_dir}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for json_file in ethno_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        cursor.execute("""
        INSERT OR REPLACE INTO ethnographic_records (
            title, source, period_or_event, geographic_location, summary,
            source_file, critical_elements_json, protocol_relevance_json, tags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("title"), data.get("source"), data.get("period_or_event"),
            data.get("geographic_location"), data.get("summary"), json_file.name,
            json.dumps(data.get("critical_elements_observed", [])),
            json.dumps(data.get("eight_step_protocol_relevance", {})),
            json.dumps(data.get("tags", []))
        ))

    conn.commit()
    conn.close()
    print("Ethnographic records synced successfully.")

def main():
    parser = argparse.ArgumentParser(description="Sync JSON logs and records to SQLite database.")
    parser.add_argument("--db_path", type=str, default="chrysalis/data/chrysalis.db")
    parser.add_argument("--log_path", type=str, default="chrysalis/simulations/experiment_log.json")
    parser.add_argument("--ethno_dir", type=str, default="chrysalis/data/ethnographic")
    
    args = parser.parse_args()
    db_path = Path(args.db_path)
    
    # Ensure tables exist
    init_db(db_path)
    
    sync_experiments(db_path, Path(args.log_path))
    sync_ethnographic(db_path, Path(args.ethno_dir))

if __name__ == "__main__":
    main()
