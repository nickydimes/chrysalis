import argparse
import re
from pathlib import Path
from chrysalis.src.llm_integration.llm_clients import (
    GeminiAPIClient,
    OllamaClient,
    LLMClient,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a new Python tool using an LLM."
    )
    parser.add_argument(
        "tool_name", type=str, help="Name of the tool (e.g., data_analyzer)."
    )
    parser.add_argument(
        "description", type=str, help="Description of what the tool should do."
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        choices=["src/tools", "src/experiments", "src/analysis"],
        default="src/tools",
        help="Target directory for the new tool.",
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
        default="code_generation_template",
        help="Name of the prompt template to use.",
    )

    args = parser.parse_args()

    template_path = Path("chrysalis/prompts/templates") / f"{args.template_name}.md"
    if not template_path.exists():
        print(f"Error: Template file '{template_path}' not found.")
        exit(1)

    target_path = Path("chrysalis") / args.target_dir / f"{args.tool_name}.py"
    if target_path.exists():
        print(f"Error: Tool '{target_path}' already exists. Refusing to overwrite.")
        exit(1)

    try:
        # 1. Prepare the LLM prompt
        template_content = template_path.read_text(encoding="utf-8")
        llm_prompt = template_content.replace("{TOOL_NAME}", args.tool_name)
        llm_prompt = llm_prompt.replace("{TOOL_DESCRIPTION}", args.description)
        llm_prompt = llm_prompt.replace("{TARGET_DIR}", args.target_dir)

        # 2. Initialize LLM Client
        llm_client: LLMClient
        if args.llm_client == "gemini_api":
            llm_client = GeminiAPIClient(
                model_name=(
                    args.llm_model if args.llm_model else "models/gemini-pro-latest"
                )
            )
        elif args.llm_client == "ollama":
            # Using a coder model by default if not specified
            model_name = args.llm_model if args.llm_model else "qwen2.5-coder:32b"
            llm_client = OllamaClient(model_name=model_name)

        print(f"Using LLM client: {args.llm_client} (model: {llm_client.model_name})")
        print(f"Generating tool: {args.tool_name}...")

        generated_response = llm_client.generate_text(llm_prompt)

        # 3. Extract code from response
        code_match = re.search(r"```python\n(.*?)```", generated_response, re.DOTALL)
        if code_match:
            generated_code = code_match.group(1)
        else:
            print(
                "Warning: Could not find Python code block in LLM response. Using entire response."
            )
            generated_code = generated_response

        # 4. Save the generated code
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(generated_code, encoding="utf-8")

        print(f"Successfully generated and saved tool to: {target_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    main()
