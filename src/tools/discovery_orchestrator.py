import argparse
import os
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import subprocess
import sys


def get_db_connection(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def update_status(
    conn: sqlite3.Connection,
    file_path: str,
    status: str,
    experiment_id: Optional[str] = None,
) -> None:
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        """
    INSERT OR REPLACE INTO discovery_state (file_path, status, last_updated, experiment_id)
    VALUES (?, ?, ?, ?)
    """,
        (file_path, status, now, experiment_id),
    )
    conn.commit()


def get_processed_files(conn: sqlite3.Connection) -> Dict[str, str]:
    cursor = conn.cursor()
    cursor.execute("SELECT file_path, status FROM discovery_state")
    return {row[0]: row[1] for row in cursor.fetchall()}


def run_cli_command(args: List[str], timeout: int = 300) -> (bool, str):
    """Runs a chrysalis_cli command via subprocess."""
    # Find the script location relative to this tool
    script_dir = Path(__file__).parent.parent.parent
    cli_path = script_dir / "chrysalis_cli.py"

    cmd = [sys.executable, str(cli_path)] + args
    print(f"\n[ORCHESTRATOR] RUNNING: {' '.join(cmd)}")

    # We set PYTHONPATH and pass through existing environment (including API keys)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{script_dir}:{env.get('PYTHONPATH', '')}"

    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )

        # Progress indicator
        spinner = "|/-\\"
        idx = 0
        while process.poll() is None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            print(f"Still running... {spinner[idx % len(spinner)]}", end="\r")
            idx += 1

        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print("\n[ORCHESTRATOR] ERROR in command execution:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False, stdout

        return True, stdout

    except subprocess.TimeoutExpired:
        print(f"\n[ORCHESTRATOR] ERROR: Command timed out after {timeout} seconds.")
        process.kill()
        stdout, stderr = process.communicate()
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False, "Timeout"
    except Exception as e:
        print(f"\n[ORCHESTRATOR] An unexpected error occurred: {e}")
        return False, str(e)


def orchestrate_discovery(
    db_path: Path, raw_dir: Path, llm_client: str, llm_model: Optional[str]
) -> None:
    if not raw_dir.exists():
        print(f"Error: Raw data directory {raw_dir} not found.")
        return

    if not db_path.exists():
        print(f"Database {db_path} not found. Initializing...")
        run_cli_command(["init-db", "--db-path", str(db_path)])

    conn = get_db_connection(db_path)
    processed = get_processed_files(conn)

    # --- STAGE 1: INGEST ---
    for raw_file in raw_dir.glob("*.txt"):
        file_key = str(raw_file)
        # Handle both relative and absolute keys if needed, but here we assume relative to project root
        if file_key not in processed or processed[file_key] == "Pending":
            print(f"\n[ORCHESTRATOR] Stage 1: Ingesting: {raw_file.name}")
            ingest_args: List[str] = [
                "ingest",
                str(raw_file),
                "--llm-client",
                llm_client,
            ]
            if llm_model:
                ingest_args.extend(["--llm-model", llm_model])

            success, output = run_cli_command(ingest_args)
            if success:
                update_status(conn, file_key, "Ingested")
            else:
                continue

    # --- STAGE 2: HYPOTHESIS & PARAMS ---
    processed = get_processed_files(conn)
    if any(s == "Ingested" for s in processed.values()):
        print("\n[ORCHESTRATOR] Stage 2: Hypothesis & Param Suggestion...")
        hypo_args: List[str] = ["generate-hypotheses", "--llm-client", llm_client]
        if llm_model:
            hypo_args.extend(["--llm-model", llm_model])
        run_cli_command(hypo_args)

        suggest_args: List[str] = ["suggest-params", "--llm-client", llm_client]
        if llm_model:
            suggest_args.extend(["--llm-model", llm_model])

        success, output = run_cli_command(suggest_args)
        if success:
            for f, s in processed.items():
                if s == "Ingested":
                    update_status(conn, f, "Hypothesized")

    # --- STAGE 3: SIMULATE ---
    processed = get_processed_files(conn)
    if any(s == "Hypothesized" for s in processed.values()):
        # Find project root
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent
        suggested_config = project_root / "simulations" / "suggested_experiment.json"

        if suggested_config.exists():
            print("\n[ORCHESTRATOR] Stage 3: Running Suggested Simulation...")
            success, output = run_cli_command(["simulate", str(suggested_config)])
            if success:
                run_cli_command(["sync-db"])

                cursor = conn.cursor()
                cursor.execute(
                    "SELECT experiment_id FROM experiments ORDER BY start_time DESC LIMIT 1"
                )
                row = cursor.fetchone()
                exp_id = row[0] if row else None

                for f, s in processed.items():
                    if s == "Hypothesized":
                        update_status(conn, f, "Simulated", experiment_id=exp_id)

    # --- STAGE 4: INTERPRET ---
    processed = get_processed_files(conn)
    for f, s in processed.items():
        if s == "Simulated":
            cursor = conn.cursor()
            cursor.execute(
                "SELECT experiment_id, output_base_dir FROM experiments WHERE experiment_id = (SELECT experiment_id FROM discovery_state WHERE file_path = ?)",
                (f,),
            )
            row = cursor.fetchone()
            if row:
                exp_id, out_dir = row
                print(f"\n[ORCHESTRATOR] Stage 4: Interpreting results for {exp_id}")
                interp_args: List[str] = [
                    "interpret-results",
                    out_dir,
                    "--llm-client",
                    llm_client,
                ]
                if llm_model:
                    interp_args.extend(["--llm-model", llm_model])

                success, output = run_cli_command(interp_args)
                if success:
                    update_status(conn, f, "Interpreted", experiment_id=exp_id)

    # --- STAGE 5: META-ANALYZE ---
    processed = get_processed_files(conn)
    if any(s == "Interpreted" for s in processed.values()):
        print("\n[ORCHESTRATOR] Stage 5: Project Meta-Analysis...")
        run_cli_command(["index-rag"])
        run_cli_command(["meta-analyze"])

        for f, s in processed.items():
            if s == "Interpreted":
                update_status(conn, f, "Meta-Analyzed")

    print("\n[ORCHESTRATOR] Discovery cycle complete.")
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous Discovery Orchestrator for the Chrysalis project."
    )
    parser.add_argument("--db_path", type=str, default="data/chrysalis.db")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--llm_client", type=str, default="ollama")
    parser.add_argument("--llm_model", type=str, default=None)

    args = parser.parse_args()

    db_path = Path(args.db_path)
    raw_dir = Path(args.raw_dir)

    if not raw_dir.exists():
        db_path = Path("chrysalis") / args.db_path
        raw_dir = Path("chrysalis") / args.raw_dir

    orchestrate_discovery(db_path, raw_dir, args.llm_client, args.llm_model)


if __name__ == "__main__":
    main()
