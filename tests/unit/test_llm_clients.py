from unittest.mock import patch
import pytest

# CRITICAL: Import the function you are testing
from chrysalis.src.llm_integration.llm_clients import _load_ethnographic_schema


def test_load_ethnographic_schema_file_not_found():
    """Test _load_ethnographic_schema when schema file is not found."""
    non_existent_project_root_str = "/non/existent/project/path"

    # Patch Path.exists WHERE IT IS USED in your source file
    with patch(
        "chrysalis.src.llm_integration.llm_clients.os.getenv",
        return_value=non_existent_project_root_str,
    ):
        with patch(
            "chrysalis.src.llm_integration.llm_clients.Path.exists", return_value=False
        ):
            # THESE LINES MUST BE INDENTED
            with pytest.raises(
                FileNotFoundError,
                match=f"CHRYSALIS_PROJECT_ROOT is set to an invalid path: {non_existent_project_root_str}",
            ):
                _load_ethnographic_schema()
