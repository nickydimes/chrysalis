import asyncio
import os
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

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


def format_note_for_prompt(note_data):
    """Format a single processed note for insertion into a prompt template."""
    metadata = note_data.get('metadata', {})
    protocol_steps = note_data.get('protocol_steps', {})

    metadata_str = json.dumps(metadata, indent=2)

    steps_lines = []
    for step, content in protocol_steps.items():
        if content:
            steps_lines.append(f"  - {step}: {content}")
    steps_str = "\n".join(steps_lines) if steps_lines else "  (no protocol step content)"

    return metadata_str, steps_str


async def main():
    """
    Reads processed JSON notes and uses the 'gemini_chat' tool to analyze
    each one individually using the observation_analysis prompt template.
    """
    server_url = "http://localhost:3000/mcp"
    if urlparse(server_url).scheme not in ("http", "https"):
        print("Error: Server URL must start with http:// or https://")
        sys.exit(1)

    processed_notes_dir = BASE_DIR / "research" / "processed_notes"
    if not processed_notes_dir.exists():
        print(f"Error: Processed notes directory not found at '{processed_notes_dir}'")
        print("Please run supernote_parser.py first.")
        return

    json_files = sorted(f for f in os.listdir(processed_notes_dir) if f.endswith('.json'))
    if not json_files:
        print(f"No processed JSON files found in {processed_notes_dir}")
        return

    template = load_template("observation_analysis.txt")

    try:
        print(f"--- Connecting to gemini-mcp server at {server_url} ---")
        async with sse_client(server_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                print(f"--- Connection established. Analyzing {len(json_files)} notes... ---")

                for filename in json_files:
                    file_path = processed_notes_dir / filename
                    with open(file_path, 'r', encoding='utf-8') as f:
                        note_data = json.load(f)

                    metadata_str, steps_str = format_note_for_prompt(note_data)
                    prompt = template.format(
                        metadata=metadata_str,
                        protocol_steps=steps_str
                    )

                    source = note_data.get('metadata', {}).get('original_file', filename)
                    print(f"\n--- Analyzing: {source} ---")

                    analysis_result = await session.call_tool('gemini_chat', {'message': prompt})

                    print("\n--- Gemini Analysis ---")
                    if analysis_result:
                        for content_part in analysis_result:
                            print(content_part.get('text', ''))
                    else:
                        print("No content returned from tool call.")

    except Exception as e:
        print(f"An error occurred during the workflow: {e}")
    finally:
        print("\n--- Workflow complete ---")

if __name__ == "__main__":
    print("Please ensure the 'gemini-mcp' server is running in a separate terminal with 'npm run start'")
    asyncio.run(main())
