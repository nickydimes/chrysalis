import os
import json
import typer
from pathlib import Path
from jsonschema import validate, ValidationError

# Directory where processed JSON notes are stored
PROCESSED_NOTES_DIR = 'research/processed_notes'
SCHEMA_PATH = 'schema/processed_note.schema.json'

app = typer.Typer()

def load_schema(schema_path: Path):
    """Loads a JSON schema from the given path."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@app.command()
def validate_notes(
    processed_dir: Path = typer.Option(
        PROCESSED_NOTES_DIR,
        "--processed-dir", "-p",
        help="Directory containing processed JSON notes to validate."
    ),
    schema_path: Path = typer.Option(
        SCHEMA_PATH,
        "--schema", "-s",
        help="Path to the JSON schema file for validation."
    )
):
    """
    Validates processed JSON notes against a defined JSON schema.
    """
    if not processed_dir.exists():
        print(f"Error: Processed notes directory not found at '{processed_dir}'")
        raise typer.Exit(code=1)

    if not schema_path.exists():
        print(f"Error: Schema file not found at '{schema_path}'")
        raise typer.Exit(code=1)

    try:
        schema = load_schema(schema_path)
    except Exception as e:
        print(f"Error loading schema from '{schema_path}': {e}")
        raise typer.Exit(code=1)

    json_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    if not json_files:
        print(f"No processed JSON files found in '{processed_dir}'.")
        raise typer.Exit(code=0) # Exit successfully if no files to validate

    all_valid = True
    print(f"--- Validating {len(json_files)} notes in '{processed_dir}' against '{schema_path}' ---")

    for filename in json_files:
        file_path = processed_path = os.path.join(processed_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                note_data = json.load(f)
            
            validate(instance=note_data, schema=schema)
            print(f"  ✓ {filename}: Valid")
        except FileNotFoundError:
            print(f"  ✗ {filename}: File not found.")
            all_valid = False
        except json.JSONDecodeError as e:
            print(f"  ✗ {filename}: Invalid JSON format - {e}")
            all_valid = False
        except ValidationError as e:
            print(f"  ✗ {filename}: Validation Error - {e.message} (Path: {e.json_path})")
            all_valid = False
        except Exception as e:
            print(f"  ✗ {filename}: An unexpected error occurred - {e}")
            all_valid = False

    if all_valid:
        print("\nAll processed notes are valid!")
        raise typer.Exit(code=0)
    else:
        print("\nSome processed notes failed validation.")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
