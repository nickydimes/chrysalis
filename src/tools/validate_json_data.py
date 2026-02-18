import argparse
import json
from pathlib import Path
from jsonschema import validate, ValidationError

def main():
    parser = argparse.ArgumentParser(description="Validate a JSON data file against a JSON schema.")
    parser.add_argument("data_file", type=str,
                        help="Path to the JSON data file to validate.")
    parser.add_argument("schema_file", type=str,
                        help="Path to the JSON schema file.")

    args = parser.parse_args()

    data_path = Path(args.data_file)
    schema_path = Path(args.schema_file)

    if not data_path.exists():
        print(f"Error: Data file '{data_path}' not found.")
        exit(1)
    if not schema_path.exists():
        print(f"Error: Schema file '{schema_path}' not found.")
        exit(1)

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)

        validate(instance=data, schema=schema)
        print(f"Validation successful: '{data_path}' is valid against '{schema_path}'.")

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{e.doc}': {e}")
        exit(1)
    except ValidationError as e:
        print(f"Validation failed for '{data_path}' against '{schema_path}'.")
        print("--- Validation Error Details ---")
        print(f"Message: {e.message}")
        print(f"Path: {list(e.path)}")
        print(f"Validator: {e.validator} = {e.validator_value}")
        print(f"Schema Path: {list(e.schema_path)}")
        print("--- End Error Details ---")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)

if __name__ == "__main__":
    main()
