import argparse
import json
import re
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate academic search queries from research hypotheses."
    )
    parser.add_argument("--hypotheses_file", type=str, default="research/hypotheses.md")
    parser.add_argument(
        "--llm_client", type=str, choices=["gemini_api", "ollama"], default="ollama"
    )
    parser.add_argument("--llm_model", type=str, default="llama3.3:70b-instruct-q4_K_M")
    parser.add_argument(
        "--template_name", type=str, default="literature_search_query_template"
    )
    parser.add_argument(
        "--output_file", type=str, default="research/search_queries.json"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    hypotheses_path = project_root / args.hypotheses_file
    template_path = project_root / "prompts" / "templates" / f"{args.template_name}.md"
    output_path = project_root / args.output_file

    if not hypotheses_path.exists():
        # Fallback
        hypotheses_path = Path(args.hypotheses_file)
        if not hypotheses_path.exists():
            print(f"Error: Hypotheses file '{hypotheses_path}' not found.")
            exit(1)

    if not template_path.exists():
        template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
        if not template_path.exists():
            print(f"Error: Template file '{template_path}' not found.")
            exit(1)

    try:
        # 1. Load Hypotheses
        hypotheses_content = hypotheses_path.read_text(encoding="utf-8")

        # 2. Prepare Prompt
        template_content = template_path.read_text(encoding="utf-8")
        prompt = template_content.replace("{HYPOTHESIS}", hypotheses_content)

        # 3. Initialize LLM Client
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
        print("Generating search queries...")

        response = llm_client.generate_text(prompt)

        # 4. Extract JSON list
        json_match = re.search(r"(\[.*\])", response, re.DOTALL)
        if json_match:
            queries = json.loads(json_match.group(1))
        else:
            # Try to split by lines if not JSON
            queries = [
                line.strip()
                for line in response.split("\n")
                if line.strip()
                and not line.startswith("[")
                and not line.startswith("]")
            ]

        # 5. Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(queries, indent=4), encoding="utf-8")
        print(f"Successfully saved {len(queries)} search queries to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
