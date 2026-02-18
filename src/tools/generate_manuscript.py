import argparse
import re
import json
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)

from typing import List, Dict, Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize a research manuscript in LaTeX format."
    )
    parser.add_argument(
        "--llm_client", type=str, choices=["gemini_api", "ollama"], default="ollama"
    )
    parser.add_argument("--llm_model", type=str, default="llama3.3:70b-instruct-q4_K_M")
    parser.add_argument(
        "--template_name", type=str, default="manuscript_synthesis_template"
    )
    parser.add_argument("--output_file", type=str, default="research/manuscript.tex")

    args = parser.parse_args()

    # Find project root
    script_dir: Path = Path(__file__).parent.absolute()
    project_root: Path = script_dir.parent.parent

    template_path: Path = (
        project_root / "prompts" / "templates" / f"{args.template_name}.md"
    )
    output_path: Path = project_root / args.output_file

    if not template_path.exists():
        # Fallback
        template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
        if not template_path.exists():
            print(f"Error: Template file '{template_path}' not found.")
            exit(1)

    try:
        # 1. Gather Project Artifacts
        print("Gathering project artifacts for synthesis...")

        # Ethnographic Data
        ethno_dir: Path = project_root / "data" / "ethnographic"
        ethno_summary: str = ""
        if ethno_dir.exists():
            ethno_files: List[Path] = list(ethno_dir.glob("*.json"))
            for f in ethno_files[:3]:
                with open(f, "r") as jf:
                    data: Dict[str, Any] = json.load(jf)
                    ethno_summary += f"\n- {data.get('title')}: {data.get('summary')}"

        # Hypotheses
        # hypo_path was unused

        # Simulation Interpretation
        sim_summary: str = ""
        reports_dir: Path = project_root / "simulations" / "reports"
        if reports_dir.exists():
            interp_files: List[Path] = list(reports_dir.glob("**/interpretation.md"))
            for f in interp_files[:2]:
                sim_summary += f"\n---\nReport from {f.parent.name}:\n{f.read_text(encoding='utf-8')}"

        # Meta-Analysis
        meta_path: Path = project_root / "research" / "meta_analysis_report.md"
        meta_content: str = (
            meta_path.read_text(encoding="utf-8")
            if meta_path.exists()
            else "No meta-analysis available."
        )

        # Grounding
        grounding_path: Path = project_root / "research" / "literature_grounding.md"
        grounding_content: str = (
            grounding_path.read_text(encoding="utf-8")
            if grounding_path.exists()
            else "No literature grounding available."
        )

        # Figures (Paths for LLM to reference)
        figure_paths: List[str] = []
        plots_dir: Path = project_root / "data" / "analysis_plots"
        if plots_dir.exists():
            figure_paths.extend(
                [str(f.relative_to(project_root)) for f in plots_dir.glob("*.png")]
            )
        if reports_dir.exists():
            figure_paths.extend(
                [str(f.relative_to(project_root)) for f in reports_dir.glob("**/*.png")]
            )

        figures_list: str = "\n".join([f"- {path}" for path in figure_paths])

        # 2. Prepare Prompt
        template_content: str = template_path.read_text(encoding="utf-8")
        llm_prompt: str = template_content.replace("{ETHNO_DATA}", ethno_summary)
        llm_prompt = llm_prompt.replace("{SIM_DATA}", sim_summary)
        llm_prompt = llm_prompt.replace("{META_ANALYSIS}", meta_content)
        llm_prompt = llm_prompt.replace("{GROUNDING_DATA}", grounding_content)
        llm_prompt += f"\n\nAvailable Figures (reference these in LaTeX with \\includegraphics):\n{figures_list}"

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
        print("Synthesizing manuscript...")

        generated_response: str = llm_client.generate_text(llm_prompt)

        # 4. Extract LaTeX code
        latex_match = re.search(r"```latex\n(.*?)\n```", generated_response, re.DOTALL)
        if latex_match:
            manuscript_code: str = latex_match.group(1)
        else:
            # Try to find \documentclass or just use entire response
            if "\\documentclass" in generated_response:
                manuscript_code = generated_response
            else:
                print(
                    "Warning: Could not find LaTeX code block or document class. Saving entire response."
                )
                manuscript_code = generated_response

        # 5. Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(manuscript_code, encoding="utf-8")

        print(f"Successfully generated and saved manuscript to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
