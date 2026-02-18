import argparse
from pathlib import Path
from chrysalis.src.rag.chat_with_project import chat_with_project


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Grounding Report cross-referencing findings with academic literature."
    )
    parser.add_argument("--hypotheses_file", type=str, default="research/hypotheses.md")
    parser.add_argument("--llm_model", type=str, default="llama3.3:70b-instruct-q4_K_M")
    parser.add_argument(
        "--template_name", type=str, default="literature_grounding_template"
    )
    parser.add_argument(
        "--output_file", type=str, default="research/literature_grounding.md"
    )

    args = parser.parse_args()

    # Find project root
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    hypotheses_path = project_root / args.hypotheses_file
    output_path = project_root / args.output_file

    if not hypotheses_path.exists():
        # Fallback
        hypotheses_path = Path(args.hypotheses_file)
        if not hypotheses_path.exists():
            print(f"Error: Hypotheses file '{hypotheses_path}' not found.")
            exit(1)

    try:
        print("--- Chrysalis Literature Grounding Tool ---")

        # 1. Load Hypotheses
        hypotheses = hypotheses_path.read_text(encoding="utf-8")

        # 2. Perform Grounding Analysis using RAG
        query = f"Evaluate the following hypotheses against external academic literature: {hypotheses[:500]}..."

        print(f"Synthesizing Grounding Report using model: {args.llm_model}...")
        report = chat_with_project(
            query, llm_model_name=args.llm_model, template_name=args.template_name
        )

        # 3. Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        print("\n--- Generated Grounding Report ---")
        print(report)
        print(f"\nReport saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
