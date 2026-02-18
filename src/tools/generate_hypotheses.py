import argparse
import json
from pathlib import Path
import pandas as pd
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)
from chrysalis.src.analysis.ethnographic_data import (
    load_ethnographic_records,
    aggregate_critical_elements,
    aggregate_protocol_relevance,
)

from typing import Dict, Any, List


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate research hypotheses based on ethnographic data using an LLM."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="chrysalis/data/ethnographic",
        help="Directory containing structured ethnographic JSON files.",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="hypothesis_generation_template",
        help="Name of the prompt template to use.",
    )
    parser.add_argument(
        "--llm_client",
        type=str,
        choices=["gemini_api", "ollama"],
        default="gemini_api",
        help="Choose which LLM client to use.",
    )
    parser.add_argument(
        "--llm_model", type=str, default=None, help="Specify the LLM model name."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="chrysalis/research/hypotheses.md",
        help="File to save the generated hypotheses.",
    )

    args = parser.parse_args()

    # Find project root relative to this script
    script_dir: Path = Path(__file__).parent.absolute()
    project_root: Path = script_dir.parent.parent

    data_dir: Path = Path(args.data_dir)
    schema_path: Path = project_root / "schema" / "ethnographic_record.json"
    template_path: Path = (
        project_root / "prompts" / "templates" / f"{args.template_name}.md"
    )
    output_file_path: Path = Path(args.output_file)

    if not data_dir.exists():
        # Fallback
        data_dir = project_root / "data" / "ethnographic"
        if not data_dir.exists():
            print(f"Error: Data directory '{data_dir}' not found.")
            exit(1)
    if not schema_path.exists():
        schema_path = Path("chrysalis/schema/ethnographic_record.json")
        if not schema_path.exists():
            print(f"Error: Schema file '{schema_path}' not found.")
            exit(1)
    if not template_path.exists():
        template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
        if not template_path.exists():
            print(f"Error: Template file '{template_path}' not found.")
            exit(1)

    try:
        # 1. Load and process ethnographic data
        with open(schema_path, "r", encoding="utf-8") as f:
            schema: Dict[str, Any] = json.load(f)

        df: pd.DataFrame = load_ethnographic_records(data_dir, schema)
        if df.empty:
            print("No valid ethnographic records found to analyze.")
            exit(0)

        # Generate a summary for the prompt
        critical_elements: pd.Series = aggregate_critical_elements(df).head(10)
        protocol_relevance: pd.Series = aggregate_protocol_relevance(df)

        summary_lines: List[str] = ["Aggregated Observations:"]
        summary_lines.append(f"Total Records: {len(df)}")
        summary_lines.append("\nTop Critical Elements:")
        for elem, count in critical_elements.items():
            summary_lines.append(f"- {elem}: {count}")

        summary_lines.append("\nProtocol Phase Relevance (counts):")
        for phase, count in protocol_relevance.items():
            summary_lines.append(f"- {phase}: {count}")

        summary_lines.append("\nKey Summaries from Records:")
        for _, row in df.iterrows():
            summary_lines.append(f"- {row['title']}: {row['summary']}")

        data_summary: str = "\n".join(summary_lines)

        # 2. Prepare the LLM prompt
        template_content: str = template_path.read_text(encoding="utf-8")
        llm_prompt: str = template_content.replace("{DATA_SUMMARY}", data_summary)

        # 3. Initialize LLM Client
        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            llm_client = OllamaClient(
                model_name=(
                    args.llm_model if args.llm_model else "llama3.3:70b-instruct-q4_K_M"
                )
            )

        print(f"Using LLM client: {args.llm_client} (model: {llm_client.model_name})")
        print("Generating hypotheses...")

        hypotheses: str = llm_client.generate_text(llm_prompt)

        # 4. Save and output
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(hypotheses, encoding="utf-8")

        print("\n--- Generated Hypotheses ---")
        print(hypotheses)
        print(f"\nHypotheses saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
