import argparse
import re
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)


def main():
    parser = argparse.ArgumentParser(description="Refactor a Python file using an LLM.")
    parser.add_argument(
        "file_path", type=str, help="Path to the Python file to refactor."
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
        "--template_name",
        type=str,
        default="refactoring_template",
        help="Name of the prompt template to use.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original file with refactored code.",
    )

    args = parser.parse_args()

    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File '{file_path}' not found.")
        exit(1)

    template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
    if not template_path.exists():
        print(f"Error: Template file '{template_path}' not found.")
        exit(1)

    try:
        # 1. Prepare the LLM prompt
        original_code = file_path.read_text(encoding="utf-8")
        template_content = template_path.read_text(encoding="utf-8")
        llm_prompt = template_content.replace("{FILE_PATH}", str(file_path))
        llm_prompt = llm_prompt.replace("{ORIGINAL_CODE}", original_code)

        # 2. Initialize LLM Client
        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            # Using a coder model by default
            model_name = args.llm_model if args.llm_model else "qwen2.5-coder:32b"
            llm_client = OllamaClient(model_name=model_name)

        print(f"Using LLM client: {args.llm_client} (model: {llm_client.model_name})")
        print(f"Refactoring: {file_path}...")

        generated_response = llm_client.generate_text(llm_prompt)

        # 3. Extract code from response
        code_match = re.search(r"```python\n(.*?)```", generated_response, re.DOTALL)
        if code_match:
            refactored_code = code_match.group(1)
        else:
            print(
                "Warning: Could not find Python code block in LLM response. Using entire response."
            )
            refactored_code = generated_response

        # 4. Output or Save
        if args.inplace:
            file_path.write_text(refactored_code, encoding="utf-8")
            print(f"Successfully refactored and updated: {file_path}")
        else:
            print("\n--- Refactored Code ---")
            print(refactored_code)
            print("\n-----------------------")
            print("To overwrite the file, use the --inplace flag.")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
