import os
import re

# The Eight-Step Navigation Protocol
PROTOCOL_STEPS = [
    "Purification",
    "Containment",
    "Anchoring",
    "Stillness",
    "An offering",
    "Receptivity",
    "Return",
    "Integration"
]

# Directory where sorted results will be saved
OBSERVATIONS_DIR = os.path.join(os.getcwd(), 'research', 'observations')
RAW_NOTES_DIR = os.path.join(os.getcwd(), 'data', 'raw_notes')

# Ensure the directories exist
os.makedirs(OBSERVATIONS_DIR, exist_ok=True)
os.makedirs(RAW_NOTES_DIR, exist_ok=True)

def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_to_markdown_file(file_path, content):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content)

def extract_protocol_content(markdown_text):
    protocol_contents = {step: [] for step in PROTOCOL_STEPS}
    # Matches headers like ## Purification or ## Stillness
    header_pattern = re.compile(r'##\s+(.*)')
    sections = header_pattern.split(markdown_text)
    
    current_section = None
    for i, section in enumerate(sections):
        if i % 2 == 0:
            if current_section and current_section in protocol_contents:
                protocol_contents[current_section].append(section.strip())
        else:
            current_section = section.strip()
    return protocol_contents

def process_all_raw_notes():
    files = [f for f in os.listdir(RAW_NOTES_DIR) if f.endswith('.md')]
    if not files:
        print(f"No .md files found in {RAW_NOTES_DIR}")
        return

    for filename in files:
        file_path = os.path.join(RAW_NOTES_DIR, filename)
        markdown_text = read_markdown_file(file_path)
        protocol_contents = extract_protocol_content(markdown_text)
        
        for step, contents in protocol_contents.items():
            if contents:
                file_name = os.path.join(OBSERVATIONS_DIR, f"{step.lower().replace(' ', '_')}.md")
                write_to_markdown_file(file_name, f"### From {filename}\n" + "\n".join(contents) + "\n\n---\n\n")
        
        # Sync Integration to the master log
        integration_contents = protocol_contents.get("Integration", [])
        if integration_contents:
            master_log_path = os.path.join(OBSERVATIONS_DIR, "master_integration_log.md")
            write_to_markdown_file(master_log_path, f"## {filename}\n\n" + "\n".join(integration_contents) + "\n\n---\n\n")
        print(f"Processed: {filename}")

if __name__ == "__main__":
    process_all_raw_notes()