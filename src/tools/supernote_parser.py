import os
import re
import json
import yaml
import datetime
import typer
from pathlib import Path # Import Path

# The Eight-Step Navigation Protocol
PROTOCOL_STEPS = [
    "Purification",
    "Containment",
    "Anchoring",
    "Dissolution",
    "Liminality",
    "Encounter",
    "Integration",
    "Emergence"
]

# Create a Typer app
app = typer.Typer()

def read_markdown_file(file_path):
    """
    Reads a markdown file and separates the YAML frontmatter from the content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    frontmatter = {}
    markdown_content = content
    
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                markdown_content = parts[2]
            except yaml.YAMLError as e:
                print(f"Warning: Could not parse YAML frontmatter in {file_path}. Error: {e}")

    return frontmatter, markdown_content

def extract_protocol_content(markdown_text):
    protocol_contents = {step: "" for step in PROTOCOL_STEPS}
    # Matches headers like ## Purification or ## Stillness
    header_pattern = re.compile(r'##\s+(.*)')
    sections = header_pattern.split(markdown_text)
    
    current_section = None
    for i, section in enumerate(sections):
        if i % 2 == 0:
            if current_section and current_section in protocol_contents:
                protocol_contents[current_section] = section.strip()
        else:
            current_section = section.strip()
    return protocol_contents

@app.command() # Decorator to make this a CLI command
def process_notes(
    input_dir: Path = typer.Option(
        'data/raw_notes', # Default input directory
        "--input-dir", "-i",
        help="Directory containing raw markdown notes to process."
    ),
    output_dir: Path = typer.Option(
        'research/processed_notes', # Default output directory
        "--output-dir", "-o",
        help="Directory where structured JSON notes will be saved."
    )
):
    """
    Processes raw markdown notes, extracts metadata and protocol step content,
    and saves them as structured JSON files.
    """
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True) # Ensure input dir also exists

    files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
    if not files:
        print(f"No .md files found in {input_dir}")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        metadata, markdown_content = read_markdown_file(file_path)
        
        # Add original filename to metadata
        if metadata:
            metadata['original_file'] = filename
        else:
            metadata = {'original_file': filename}

        # Convert any date objects in metadata to string
        for key, value in metadata.items():
            if isinstance(value, datetime.date):
                metadata[key] = value.isoformat()
        
        protocol_contents = extract_protocol_content(markdown_content)
        
        # Create the structured data object
        structured_data = {
            "metadata": metadata,
            "protocol_steps": protocol_contents
        }
        
        # Write to a JSON file
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
        print(f"Processed: {filename} -> {output_filename}")

if __name__ == "__main__":
    app() # Run the Typer app
