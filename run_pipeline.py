#!/usr/bin/env python3
"""
Chrysalis Automated Pipeline Orchestrator.
Automates the flow from database sync to meta-analysis and documentation.
"""

import subprocess
import sys
import os
from pathlib import Path
import sqlite3
from typing import List, Tuple

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


def run_command(args: List[str]) -> Tuple[bool, str]:
    """Runs a chrysalis_cli command via subprocess."""
    cli_path = PROJECT_ROOT / "chrysalis_cli.py"
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
    cmd = [str(venv_python), str(cli_path)] + args
    print(f"\n>>> RUNNING: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return False, result.stderr
        print(result.stdout)
        return True, result.stdout
    except Exception as e:
        print(f"EXCEPTION: {e}")
        return False, str(e)


def get_completed_experiments() -> List[str]:
    """Fetches completed experiment output directories from the database."""
    db_path = PROJECT_ROOT / "data" / "chrysalis.db"
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT output_base_dir FROM experiments WHERE status = 'completed'"
        )
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"Error querying database: {e}")
        return []


def main():
    print("=" * 60)
    print("🦋  CHRYSALIS AUTOMATED PIPELINE  🦋")
    print("=" * 60)

    # 1. Initialize & Sync
    print("\n[STEP 1/6] Initializing and Syncing Database...")
    run_command(["init-db"])
    run_command(["sync-db"])

    # 2. Extract Knowledge Graph
    print("\n[STEP 2/6] Extracting Knowledge Graph Triples...")
    # Using deepseek-r1:32b for high-quality extraction that fits entirely in 24GB VRAM
    run_command(
        ["extract-graph", "--llm-client", "ollama", "--llm-model", "deepseek-r1:32b"]
    )

    # 3. Analyze Robustness
    print("\n[STEP 3/6] Running Robustness Analysis on Completed Experiments...")
    exp_dirs = get_completed_experiments()
    for exp_dir in exp_dirs:
        if exp_dir:
            print(f"Analyzing robustness for: {exp_dir}")
            run_command(["analyze-robustness", exp_dir])

    # 4. RAG Indexing
    print("\n[STEP 4/6] Indexing Project for RAG...")
    run_command(["index-rag"])

    # 5. Meta-Analysis
    print("\n[STEP 5/6] Performing Meta-Analysis...")
    run_command(["meta-analyze"])

    # 6. Documentation
    print("\n[STEP 6/6] Building Documentation...")
    run_command(["docs"])

    print("\n" + "=" * 60)
    print("✅ PIPELINE EXECUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
