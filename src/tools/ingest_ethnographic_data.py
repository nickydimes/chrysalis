import argparse
import os
import json
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)
from chrysalis.src.tools.extract_pdf_text import extract_text_from_pdf
from jsonschema import validate, ValidationError

try:
    from unstructured.partition.auto import partition
except ImportError:
    partition = None

from typing import Dict, Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLM prompt and structure ethnographic data from various file formats."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the raw ethnographic file (txt, md, pdf, wav, mp3).",
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default="ethnographic_extraction_template",
        help="Name of the LLM prompt template file.",
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
        "--output_dir",
        type=str,
        default="data/ethnographic/",
        help="Directory to save the structured JSON output.",
    )

    args = parser.parse_args()

    # Find the project root
    # Priority: 1. CHRYSALIS_PROJECT_ROOT env var, 2. Look for pyproject.toml in current or parent dirs
    env_root = os.getenv("CHRYSALIS_PROJECT_ROOT")
    if env_root:
        project_root: Path = Path(env_root)
    else:
        # Search upwards for pyproject.toml
        curr: Path = Path.cwd().absolute()
        project_root = curr
        for parent in [curr] + list(curr.parents):
            if (parent / "pyproject.toml").exists():
                project_root = parent
                break
        else:
            # Fallback to script location
            script_dir: Path = Path(__file__).parent.absolute()
            project_root = script_dir.parent.parent

    print(f"DEBUG: Using project_root: {project_root}")

    input_path: Path = Path(args.input_file)
    template_path: Path = (
        project_root / "prompts" / "templates" / f"{args.template_name}.md"
    )
    output_dir: Path = project_root / args.output_dir

    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found.")
        exit(1)
    if not template_path.exists():
        # Fallback
        template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
        if not template_path.exists():
            print(f"Error: Template file '{template_path}' not found.")
            exit(1)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    try:
        template_content: str = template_path.read_text(encoding="utf-8")

        # 1. Initialize LLM Client
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
        else:
            print(f"Error: Unknown LLM client '{args.llm_client}'.")
            exit(1)

        # 2. Handle different file types
        file_ext: str = input_path.suffix.lower()
        print(f"Processing input file: {input_path.name} (type: {file_ext})")

        structured_data: Dict[str, Any]

        if file_ext in [".pdf", ".wav", ".mp3"]:
            if args.llm_client == "gemini_api":
                print(f"Using Gemini Multimodal API for {file_ext}...")
                # We need to get the extraction instruction from the template,
                # but without the {TEXT_PLACEHOLDER}
                prompt: str = template_content.replace(
                    "{TEXT_PLACEHOLDER}", "[See attached file]"
                )
                # We need structured output, so we ask Gemini to return JSON
                # Actually, the template already asks for JSON.
                # But generate_from_file returns text. We'll need to parse it.
                raw_response: str = llm_client.generate_from_file(prompt, input_path)

                # Extract JSON from response
                import re

                json_match = re.search(r"({.*})", raw_response, re.DOTALL)
                if json_match:
                    structured_data = json.loads(json_match.group(1))
                else:
                    raise ValueError(
                        "Failed to extract JSON from Gemini multimodal response."
                    )
            else:
                # Ollama or local processing
                if file_ext == ".pdf":
                    if partition:
                        print(
                            f"Using 'unstructured' to partition PDF: {input_path.name}"
                        )
                        elements = partition(filename=str(input_path))
                        raw_text = "\n\n".join([str(el) for el in elements])
                    else:
                        print("Extracting text from PDF locally using pypdf...")
                        raw_text = extract_text_from_pdf(input_path)

                    llm_prompt: str = template_content.replace(
                        "{TEXT_PLACEHOLDER}", raw_text
                    )
                    structured_data = llm_client.generate_json(llm_prompt)
                else:
                    raise NotImplementedError(
                        f"Local client '{args.llm_client}' does not yet support audio files."
                    )
        else:
            # Assume text-based (txt, md, json)
            raw_text = input_path.read_text(encoding="utf-8")
            llm_prompt = template_content.replace("{TEXT_PLACEHOLDER}", raw_text)
            print("Sending prompt to LLM...")
            structured_data = llm_client.generate_json(llm_prompt)

        # 3. Validate the LLM's output against the schema
        schema_path: Path = project_root / "schema" / "ethnographic_record.json"
        if not schema_path.exists():
            schema_path = Path("chrysalis/schema/ethnographic_record.json")
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found at {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            ethnographic_schema: Dict[str, Any] = json.load(f)

        try:
            validate(instance=structured_data, schema=ethnographic_schema)
            print("LLM output successfully validated against schema.")
        except ValidationError as e:
            print(
                f"Error: LLM output failed schema validation for '{input_path.name}'."
            )
            print(f"Message: {e.message}")
            exit(1)

        # 4. Save
        output_file_name: str = f"{input_path.stem}.json"
        output_file_path: Path = output_dir / output_file_name

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(structured_data, f, indent=4, ensure_ascii=False)

        print(f"\nSuccessfully structured data and saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred during ingestion: {e}")
        exit(1)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
