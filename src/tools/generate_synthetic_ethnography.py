import argparse
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)

SCENARIOS = {
    "stuck_liminal": "A community enters Liminality but cannot find an Encounter. They cycle between Dissolution and Liminality indefinitely.",
    "sudden_emergence": "A system skips from Containment directly to Emergence due to a massive external shock.",
    "reversing_phase": "A system reaches Integration, but an failure in Anchoring causes regression back to Purification.",
    "overlapping_encounter": "Multiple conflicting Encounters happen during a single Liminal phase.",
    "shadow_protocol": "A transformation where steps are malicious or parasitic.",
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic 'Black Swan' ethnographic accounts."
    )
    parser.add_argument(
        "scenario",
        type=str,
        choices=list(SCENARIOS.keys()),
        help="Type of Black Swan scenario.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="File to save the generated account.",
    )
    parser.add_argument(
        "--llm_client", type=str, choices=["gemini_api", "ollama"], default="ollama"
    )
    parser.add_argument("--llm_model", type=str, default="llama3.3:70b-instruct-q4_K_M")
    parser.add_argument(
        "--template_name", type=str, default="synthetic_ethnography_template"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    template_path = project_root / "prompts" / "templates" / f"{args.template_name}.md"

    if not template_path.exists():
        print(f"Error: Template file '{template_path}' not found.")
        exit(1)

    try:
        # 1. Prepare Prompt
        template_content = template_path.read_text(encoding="utf-8")
        prompt = template_content.replace("{SCENARIO_TYPE}", args.scenario)
        prompt = prompt.replace("{FOCUS_DESCRIPTION}", SCENARIOS[args.scenario])

        # 2. Initialize LLM Client
        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            llm_client = OllamaClient(model_name=args.llm_model)

        print(f"Using LLM client: {args.llm_client} (model: {llm_client.model_name})")
        print(f"Generating synthetic ethnography for scenario: {args.scenario}...")

        narrative = llm_client.generate_text(prompt)

        # 3. Save
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            output_dir = project_root / "data" / "raw"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"synthetic_{args.scenario}.txt"

        output_path.write_text(narrative, encoding="utf-8")
        print(f"Successfully saved synthetic account to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
