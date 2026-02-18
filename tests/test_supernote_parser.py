import json
import pytest

from src.tools.supernote_parser import (
    read_markdown_file,
    extract_protocol_content,
    PROTOCOL_STEPS,
)


@pytest.fixture
def tmp_md(tmp_path):
    """Helper to write a markdown file and return its path."""

    def _write(content, name="note.md"):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    return _write


# --- read_markdown_file ---


class TestReadMarkdownFile:
    def test_with_frontmatter(self, tmp_md):
        md = tmp_md(
            "---\ndate: 2025-01-15\nlocation: Lab\ntags:\n  - physics\nsource: Observation\n---\n## Purification\nSome content\n"
        )
        meta, body = read_markdown_file(md)
        import datetime

        # YAML parses bare dates as datetime.date objects
        assert meta["date"] == datetime.date(2025, 1, 15)
        assert meta["location"] == "Lab"
        assert meta["tags"] == ["physics"]
        assert "## Purification" in body

    def test_without_frontmatter(self, tmp_md):
        md = tmp_md("## Purification\nJust content, no YAML.\n")
        meta, body = read_markdown_file(md)
        assert meta == {}
        assert "## Purification" in body

    def test_empty_frontmatter(self, tmp_md):
        md = tmp_md("---\n---\n## Anchoring\nContent\n")
        meta, body = read_markdown_file(md)
        # yaml.safe_load of empty string returns None
        assert meta is None or meta == {}
        assert "## Anchoring" in body

    def test_invalid_yaml(self, tmp_md, capsys):
        md = tmp_md("---\n: invalid: yaml: [unclosed\n---\nBody\n")
        meta, body = read_markdown_file(md)
        assert meta == {}
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_date_parsed_as_date_object(self, tmp_md):
        """YAML parses bare dates as datetime.date objects."""
        md = tmp_md("---\ndate: 2025-06-01\n---\nBody\n")
        meta, _ = read_markdown_file(md)
        import datetime

        assert isinstance(meta["date"], (str, datetime.date))


# --- extract_protocol_content ---


class TestExtractProtocolContent:
    def test_all_steps_present(self):
        text = "\n".join(f"## {step}\nContent for {step}." for step in PROTOCOL_STEPS)
        result = extract_protocol_content(text)
        for step in PROTOCOL_STEPS:
            assert result[step] == f"Content for {step}."

    def test_missing_steps_are_empty(self):
        text = "## Purification\nSome text.\n## Emergence\nFinal text."
        result = extract_protocol_content(text)
        assert result["Purification"] == "Some text."
        assert result["Emergence"] == "Final text."
        assert result["Containment"] == ""
        assert result["Liminality"] == ""

    def test_unknown_headers_ignored(self):
        text = "## RandomHeader\nIgnored.\n## Purification\nKept."
        result = extract_protocol_content(text)
        assert result["Purification"] == "Kept."
        assert "RandomHeader" not in result

    def test_empty_input(self):
        result = extract_protocol_content("")
        assert all(v == "" for v in result.values())
        assert set(result.keys()) == set(PROTOCOL_STEPS)

    def test_multiline_content(self):
        text = "## Dissolution\nLine 1.\n\nLine 2.\n\nLine 3.\n## Liminality\nNext."
        result = extract_protocol_content(text)
        assert "Line 1." in result["Dissolution"]
        assert "Line 3." in result["Dissolution"]
        assert result["Liminality"] == "Next."


# --- process_notes (integration) ---


class TestProcessNotes:
    def test_end_to_end(self, tmp_path):
        """Full pipeline: write markdown, run process_notes, check JSON output."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "processed"
        input_dir.mkdir()

        md_content = (
            "---\n"
            "date: 2025-03-01\n"
            "location: Field\n"
            "tags:\n  - test\n"
            "source: Unit test\n"
            "---\n"
            "## Purification\nPurify content.\n"
            "## Emergence\nEmerge content.\n"
        )
        (input_dir / "test_note.md").write_text(md_content)

        from src.tools.supernote_parser import process_notes

        process_notes(input_dir=input_dir, output_dir=output_dir)

        out_file = output_dir / "test_note.json"
        assert out_file.exists()

        data = json.loads(out_file.read_text())
        assert data["metadata"]["location"] == "Field"
        assert data["metadata"]["original_file"] == "test_note.md"
        assert data["protocol_steps"]["Purification"] == "Purify content."
        assert data["protocol_steps"]["Emergence"] == "Emerge content."
        assert data["protocol_steps"]["Containment"] == ""

    def test_no_md_files(self, tmp_path, capsys):
        input_dir = tmp_path / "empty"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        from src.tools.supernote_parser import process_notes

        process_notes(input_dir=input_dir, output_dir=output_dir)

        captured = capsys.readouterr()
        assert "No .md files" in captured.out

    def test_date_serialized_as_string(self, tmp_path):
        """datetime.date objects from YAML should be serialized as ISO strings."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "processed"
        input_dir.mkdir()

        (input_dir / "dates.md").write_text(
            "---\ndate: 2025-12-25\nlocation: X\ntags: [a]\nsource: S\n---\n## Purification\nP\n"
        )

        from src.tools.supernote_parser import process_notes

        process_notes(input_dir=input_dir, output_dir=output_dir)

        data = json.loads((output_dir / "dates.json").read_text())
        assert isinstance(data["metadata"]["date"], str)
        assert data["metadata"]["date"] == "2025-12-25"
