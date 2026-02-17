import json
import pytest
from pathlib import Path
from jsonschema import ValidationError

from src.tools.validate_notes import load_schema


SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema" / "processed_note.schema.json"


@pytest.fixture
def schema():
    return load_schema(SCHEMA_PATH)


def _make_valid_note(overrides=None):
    """Return a minimal valid processed note dict."""
    note = {
        "metadata": {
            "date": "2025-01-01",
            "location": "Lab",
            "tags": ["test"],
            "source": "Unit test",
            "original_file": "test.md",
        },
        "protocol_steps": {
            "Purification": "",
            "Containment": "",
            "Anchoring": "",
            "Dissolution": "",
            "Liminality": "",
            "Encounter": "",
            "Integration": "",
            "Emergence": "",
        },
    }
    if overrides:
        for key, val in overrides.items():
            keys = key.split(".")
            target = note
            for k in keys[:-1]:
                target = target[k]
            target[keys[-1]] = val
    return note


class TestLoadSchema:
    def test_loads_valid_schema(self):
        schema = load_schema(SCHEMA_PATH)
        assert schema["type"] == "object"
        assert "metadata" in schema["properties"]
        assert "protocol_steps" in schema["properties"]

    def test_missing_schema_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_schema(tmp_path / "nonexistent.json")


class TestSchemaValidation:
    def test_valid_note_passes(self, schema):
        from jsonschema import validate
        note = _make_valid_note()
        validate(instance=note, schema=schema)  # Should not raise

    def test_missing_metadata_fails(self, schema):
        from jsonschema import validate
        note = _make_valid_note()
        del note["metadata"]
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_missing_protocol_steps_fails(self, schema):
        from jsonschema import validate
        note = _make_valid_note()
        del note["protocol_steps"]
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_missing_required_metadata_field(self, schema):
        from jsonschema import validate
        note = _make_valid_note()
        del note["metadata"]["date"]
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_missing_protocol_step(self, schema):
        from jsonschema import validate
        note = _make_valid_note()
        del note["protocol_steps"]["Liminality"]
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_extra_protocol_step_fails(self, schema):
        """additionalProperties: false on protocol_steps."""
        from jsonschema import validate
        note = _make_valid_note()
        note["protocol_steps"]["ExtraStep"] = "should fail"
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_extra_metadata_field_allowed(self, schema):
        """additionalProperties: true on metadata."""
        from jsonschema import validate
        note = _make_valid_note()
        note["metadata"]["custom_field"] = "allowed"
        validate(instance=note, schema=schema)  # Should not raise

    def test_tags_must_be_array(self, schema):
        from jsonschema import validate
        note = _make_valid_note({"metadata.tags": "not-an-array"})
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_original_file_must_end_with_md(self, schema):
        from jsonschema import validate
        note = _make_valid_note({"metadata.original_file": "noext"})
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)

    def test_protocol_step_must_be_string(self, schema):
        from jsonschema import validate
        note = _make_valid_note()
        note["protocol_steps"]["Purification"] = 42
        with pytest.raises(ValidationError):
            validate(instance=note, schema=schema)


class TestValidateNotesIntegration:
    def test_valid_notes_directory(self, tmp_path):
        """Write valid JSON notes and run the validator."""
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        schema_dir = tmp_path / "schema"
        schema_dir.mkdir()

        # Copy schema
        import shutil
        shutil.copy(SCHEMA_PATH, schema_dir / "schema.json")

        # Write a valid note
        note = _make_valid_note()
        (notes_dir / "valid.json").write_text(json.dumps(note))

        import typer
        from src.tools.validate_notes import validate_notes
        with pytest.raises(typer.Exit) as exc_info:
            validate_notes(
                processed_dir=notes_dir,
                schema_path=schema_dir / "schema.json",
            )
        assert exc_info.value.exit_code == 0

    def test_invalid_note_returns_error(self, tmp_path):
        notes_dir = tmp_path / "notes"
        notes_dir.mkdir()
        schema_dir = tmp_path / "schema"
        schema_dir.mkdir()

        import shutil
        shutil.copy(SCHEMA_PATH, schema_dir / "schema.json")

        # Write an invalid note (missing protocol_steps)
        (notes_dir / "bad.json").write_text(json.dumps({"metadata": {}}))

        import typer
        from src.tools.validate_notes import validate_notes
        with pytest.raises(typer.Exit) as exc_info:
            validate_notes(
                processed_dir=notes_dir,
                schema_path=schema_dir / "schema.json",
            )
        assert exc_info.value.exit_code == 1
