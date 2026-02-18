import asyncio
import os
import json
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATE_DIR = BASE_DIR / "prompts" / "templates"


def load_template(name):
    """Load a prompt template by name from prompts/templates/."""
    path = TEMPLATE_DIR / name
    if not path.exists():
        print(f"Error: Template not found at '{path}'")
        sys.exit(1)
    return path.read_text(encoding="utf-8")


async def main(protocol_step: str):
    """
    Aggregates processed JSON notes for a given protocol step and uses the
    'gemini_deep_research' tool for synthesis via the step_synthesis template.
    """
    server_url = "http://localhost:3000/mcp"

    processed_notes_dir = BASE_DIR / "research" / "processed_notes"
    if not processed_notes_dir.exists():
        print(f"Error: Processed notes directory not found at '{processed_notes_dir}'")
        print("Please run supernote_parser.py first.")
        return

    json_files = sorted(
        f for f in os.listdir(processed_notes_dir) if f.endswith(".json")
    )
    if not json_files:
        print(f"No processed JSON files found in {processed_notes_dir}")
        return

    aggregated_content = []
    for filename in json_files:
        file_path = processed_notes_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            note_data = json.load(f)

        step_content = note_data.get("protocol_steps", {}).get(protocol_step)
        if step_content:
            source = note_data["metadata"].get("original_file", filename)
            aggregated_content.append(f"--- Note from {source} ---\n{step_content}\n")

    if not aggregated_content:
        print(
            f"No content found for protocol step '{protocol_step}' in processed notes."
        )
        return

    template = load_template("step_synthesis.txt")
    research_question = template.format(
        protocol_step=protocol_step, aggregated_notes="\n".join(aggregated_content)
    )

    try:
        print(f"--- Connecting to gemini-mcp server at {server_url} ---")
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(
                    f"--- Connection established. Synthesizing insights for '{protocol_step}'... ---"
                )

                synthesis_result = await session.call_tool(
                    "gemini_deep_research", {"research_question": research_question}
                )

                print(f"\n--- Gemini Deep Research Synthesis for '{protocol_step}' ---")
                if synthesis_result:
                    for content_part in synthesis_result:
                        print(content_part.get("text", ""))
                else:
                    print("No content returned from tool call.")

    except Exception as e:
        print(f"An error occurred during the workflow: {e}")
    finally:
        print("\n--- Workflow complete ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python synthesize_insights.py <protocol_step>")
        print("Example: python synthesize_insights.py Purification")
        sys.exit(1)

    protocol_step_arg = sys.argv[1]
    print(
        "Please ensure the 'gemini-mcp' server is running in a separate terminal with 'npm run start'"
    )
    asyncio.run(main(protocol_step_arg))
