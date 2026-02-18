import argparse
import os
import json
import subprocess
import sys
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)


def run_cli_command(args: list, project_root: Path):
    cmd = [sys.executable, str(project_root / "chrysalis_cli.py")] + args
    print(f"Executing: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the framework's handling of a Black Swan scenario."
    )
    parser.add_argument(
        "synthetic_file", type=str, help="Path to the synthetic ethnographic account."
    )
    parser.add_argument("--llm_client", type=str, default="ollama")
    parser.add_argument("--llm_model", type=str, default="llama3.3:70b-instruct-q4_K_M")

    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    synthetic_path = Path(args.synthetic_file)

    if not synthetic_path.exists():
        # Try relative to project root
        synthetic_path = project_root / args.synthetic_file
        if not synthetic_path.exists():
            print(f"Error: File '{args.synthetic_file}' not found.")
            exit(1)

    print("--- Chrysalis Stress Test Evaluation ---")
    print(f"Target: {synthetic_path.name}")

    try:
        # 1. Ingest the synthetic data
        print("\nStep 1: Ingesting data...")
        result = run_cli_command(
            [
                "ingest",
                str(synthetic_path),
                "--llm-client",
                args.llm_client,
                "--llm-model",
                args.llm_model,
            ],
            project_root,
        )
        if result.returncode != 0:
            print(f"Ingestion failed: {result.stderr}")
            exit(1)

        # 2. Load structured output
        json_path = (
            project_root / "data" / "ethnographic" / f"{synthetic_path.stem}.json"
        )
        with open(json_path, "r", encoding="utf-8") as f:
            structured_data = json.load(f)

        # 3. Use an LLM to evaluate the extraction quality
        eval_prompt = f"""
        You are an auditor for the Chrysalis research framework.
        A "Black Swan" synthetic account was processed. Your goal is to determine if the framework's extraction was honest or if it "forced" the protocol onto non-compliant data.

        Synthetic Account Narrative:
        {synthetic_path.read_text(encoding='utf-8')}

        Framework's Structured Extraction:
        {json.dumps(structured_data, indent=2)}

        Analysis Task:
        1. Identify specific deviations in the narrative from the Eight-Step Protocol.
        2. Check if the "eight_step_protocol_relevance" fields correctly identified these deviations (e.g., used "Not explicitly observed" or mentioned the specific subversion).
        3. If the framework claimed a step happened when the narrative explicitly stated it was skipped or reversed, flag this as a "Protocol Hallucination."
        4. Rate the framework's "Integrity Score" (0-100) based on how well it handled the anomalies.

        Evaluation Report:
        """

        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            llm_client = OllamaClient(model_name=args.llm_model)

        print("\nStep 2: Evaluating extraction integrity...")
        evaluation = llm_client.generate_text(eval_prompt)

        # 4. Save and Output
        report_path = (
            project_root / "research" / f"stress_test_report_{synthetic_path.stem}.md"
        )
        report_path.write_text(evaluation, encoding="utf-8")

        print("\n--- Stress Test Report ---")
        print(evaluation)
        print(f"\nReport saved to: {report_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
