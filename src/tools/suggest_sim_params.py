import argparse
import json
import re
from pathlib import Path
from datetime import datetime
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


def main():
    parser = argparse.ArgumentParser(
        description="Suggest simulation parameters based on ethnographic data using an LLM."
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
        default="sim_parameter_mapping_template",
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
        default="chrysalis/simulations/suggested_experiment.json",
        help="File to save the suggested simulation configuration.",
    )

    args = parser.parse_args()

    # Find project root relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    data_dir = Path(args.data_dir)
    schema_path = project_root / "schema" / "ethnographic_record.json"
    template_path = project_root / "prompts" / "templates" / f"{args.template_name}.md"

    # Ensure output path is relative to project root
    if args.output_file.startswith("chrysalis/"):
        output_file_path = project_root / args.output_file.replace("chrysalis/", "")
    else:
        output_file_path = project_root / args.output_file

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
            schema = json.load(f)

        df = load_ethnographic_records(data_dir, schema)
        if df.empty:
            print("No valid ethnographic records found to analyze.")
            exit(0)

        # Generate a summary for the prompt
        critical_elements = aggregate_critical_elements(df).head(15)
        protocol_relevance = aggregate_protocol_relevance(df)

        summary_lines = ["Aggregated Observations:"]
        summary_lines.append(f"Total Records: {len(df)}")
        summary_lines.append("\nTop Critical Elements Observed:")
        for elem, count in critical_elements.items():
            summary_lines.append(f"- {elem}: {count}")

        summary_lines.append("\nProtocol Phase Relevance (number of mentions):")
        for phase, count in protocol_relevance.items():
            summary_lines.append(f"- {phase}: {count}")

        aggregated_data_str = "\n".join(summary_lines)

        # 2. Prepare the LLM prompt
        template_content = template_path.read_text(encoding="utf-8")
        llm_prompt = template_content.replace("{AGGREGATED_DATA}", aggregated_data_str)

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
        print("Suggesting simulation parameters...")

        generated_response = llm_client.generate_text(llm_prompt)

        # 4. Extract JSON from response
        json_match = re.search(r"```json\n(.*?)\n```", generated_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback: assume plain JSON if no code block
            json_str = generated_response

        # Validate extracted string is JSON
        try:
            suggested_config = json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: Could not parse LLM response as JSON.")
            print(f"Raw Response:\n{generated_response}")
            exit(1)

        # Prepare the final experiment config
        final_config = {
            "experiment_name": f"Suggested_{suggested_config.get('simulation_module', 'Experiment')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "simulation_module": suggested_config.get("simulation_module"),
            "global_params": suggested_config.get("global_params", {}),
            "parameter_sweep": suggested_config.get("parameter_sweep", {}),
            "num_seeds": 1,
            "reasoning": suggested_config.get("reasoning", ""),
        }

        # 5. Save the suggested config
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=4)

        print("\n--- Suggested Simulation Mapping ---")
        print(f"Model: {final_config.get('simulation_module')}")
        print(f"Reasoning: {final_config.get('reasoning')}")
        print(f"\nFull configuration saved to: {output_file_path}")
        print("You can now use this file with `run_experiment.py`.")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
