import argparse
from pathlib import Path
from chrysalis.src.rag.chat_with_project import chat_with_project


def main():
    parser = argparse.ArgumentParser(
        description="Perform a meta-analysis of project findings using RAG and an LLM."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Synthesize the primary cross-modal findings between ethnographic records and simulation results.",
        help="Specific question or focus for the meta-analysis.",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="meta_analysis_template",
        help="Name of the prompt template to use.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="llama3.3:70b-instruct-q4_K_M",
        help="Local LLM model to use via Ollama.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="chrysalis/research/meta_analysis_report.md",
        help="File to save the generated report.",
    )

    args = parser.parse_args()

    # Find project root relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    if args.output_file.startswith("chrysalis/"):
        output_file_path = project_root / args.output_file.replace("chrysalis/", "")
    else:
        output_file_path = project_root / args.output_file

    print("--- Chrysalis Meta-Analysis Tool ---")
    print(f"Goal: {args.query}")
    print(f"Using Model: {args.llm_model}")
    print(f"Template: {args.template_name}")

    try:
        # Perform the meta-analysis using the RAG system
        report = chat_with_project(
            args.query, llm_model_name=args.llm_model, template_name=args.template_name
        )

        # Save and output
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        output_file_path.write_text(report, encoding="utf-8")

        print("\n--- Generated Meta-Analysis Report ---")
        print(report)
        print(f"\nReport saved to: {output_file_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "Please ensure you have indexed the project knowledge by running `chrysalis/src/rag/index_project.py` first."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
