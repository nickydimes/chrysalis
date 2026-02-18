import abc
import json
import os
import re
from typing import Dict, Any
from pathlib import Path

# Third-party imports
from google import genai
from google.genai import types
import requests

_ETHNOGRAPHIC_SCHEMA = None


def _load_ethnographic_schema() -> Dict[str, Any]:
    global _ETHNOGRAPHIC_SCHEMA
    if _ETHNOGRAPHIC_SCHEMA is None:
        env_root = os.getenv("CHRYSALIS_PROJECT_ROOT")
        if env_root:
            project_root = Path(env_root)
            # Explicitly check if the path is invalid for testing purposes
            if not project_root.exists():
                raise FileNotFoundError(
                    f"CHRYSALIS_PROJECT_ROOT is set to an invalid path: {project_root}"
                )
        else:
            curr = Path.cwd().absolute()
            project_root = curr
            for parent in [curr] + list(curr.parents):
                if (parent / "pyproject.toml").exists():
                    project_root = parent
                    break
            else:
                script_dir = Path(__file__).parent.absolute()
                project_root = script_dir.parent.parent

        schema_path = project_root / "schema" / "ethnographic_record.json"

        if not schema_path.exists():
            raise FileNotFoundError(
                f"Ethnographic schema file not found at {schema_path}"
            )
        with open(schema_path, "r", encoding="utf-8") as f:
            _ETHNOGRAPHIC_SCHEMA = json.load(f)

        # Remove '$schema' field as genai.protos.Schema does not expect it in parameters
        if "$schema" in _ETHNOGRAPHIC_SCHEMA:
            del _ETHNOGRAPHIC_SCHEMA["$schema"]

        # Also remove 'title' and 'description' from the root schema
        # as these are not expected when defining the 'parameters' of a function call.
        _ETHNOGRAPHIC_SCHEMA.pop("title", None)
        _ETHNOGRAPHIC_SCHEMA.pop("description", None)

    return _ETHNOGRAPHIC_SCHEMA


class LLMClient(abc.ABC):
    """Abstract Base Class for LLM Clients."""

    @abc.abstractmethod
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generates a JSON response from the LLM based on the given prompt.
        The LLM is expected to return valid JSON.
        """
        pass

    @abc.abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Generates a plain text response from the LLM based on the given prompt.
        """
        pass

    @abc.abstractmethod
    def generate_from_file(self, prompt: str, file_path: Path) -> str:
        """
        Generates a response from the LLM using both a prompt and an external file (PDF, Audio, etc.).
        """
        pass


class GeminiAPIClient(LLMClient):
    """
    Client for interacting with the Google Gemini API using the new google-genai SDK.
    Requires GEMINI_API_KEY environment variable to be set.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        self.client = genai.Client(api_key=self.api_key)

        # Load schema and prepare tool for function calling
        self.ethnographic_schema = _load_ethnographic_schema()
        print(f"Initialized GeminiAPIClient with model: {self.model_name}")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        try:
            tool = types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="extract_ethnographic_data",
                        description="Extract structured ethnographic data from text.",
                        parameters=self.ethnographic_schema,
                    )
                ]
            )

            config = types.GenerateContentConfig(
                tools=[tool],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["extract_ethnographic_data"],
                    )
                ),
            )

            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, config=config
            )

            # Extract function call arguments
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    return part.function_call.args

            raise ValueError(
                "Gemini API did not return a valid function call with structured data."
            )
        except Exception as e:
            raise RuntimeError(
                f"Error communicating with Gemini API using function calling: {e}"
            )

    def generate_text(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt
            )
            return response.text
        except Exception as e:
            raise RuntimeError(
                f"Error communicating with Gemini API for text generation: {e}"
            )

    def generate_from_file(self, prompt: str, file_path: Path) -> str:
        try:
            print(f"Uploading file to Gemini File API: {file_path}")
            with open(file_path, "rb") as f:
                uploaded_file = self.client.files.upload(file=f)

            response = self.client.models.generate_content(
                model=self.model_name, contents=[prompt, uploaded_file]
            )
            return response.text
        except Exception as e:
            raise RuntimeError(
                f"Error communicating with Gemini API for multimodal generation: {e}"
            )


class OllamaClient(LLMClient):
    """
    Client for interacting with a local Ollama instance.
    Assumes Ollama server is running locally.
    """

    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        print(
            f"Initialized OllamaClient with model: {self.model_name} at {self.ollama_url}"
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,  # Ensure a single, complete response
                "format": "json",  # Request JSON format directly from Ollama
            }
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status()  # Raise an exception for HTTP errors

            response_data = response.json()
            if "response" not in response_data:
                raise ValueError("Ollama API response missing 'response' field.")

            text_response = response_data["response"]

            # Try to find JSON block or just find first { and last }
            json_match = re.search(r"({.*})", text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = text_response

            return json.loads(json_str)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"Ollama server returned 404 Not Found. This often means the model '{self.model_name}' is not available or loaded by Ollama, or the Ollama server is not running or accessible at {self.ollama_url}. Please run 'ollama list' to check available models and ensure the server is active."
                )
            else:
                raise RuntimeError(f"Ollama server returned an HTTP error: {e}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama server at {self.ollama_url}. Please ensure Ollama is running."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(
                f"Error communicating with Ollama server at {self.ollama_url}: {e}"
            )
        except json.JSONDecodeError as e:
            print(
                f"Warning: Ollama API response was not valid JSON. Attempted to parse: {text_response[:500]}..."
            )
            raise ValueError(f"Failed to decode JSON from Ollama API: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred with Ollama: {e}")

    def generate_text(self, prompt: str) -> str:
        try:
            payload = {"model": self.model_name, "prompt": prompt, "stream": False}
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("response", "")
        except Exception as e:
            raise RuntimeError(
                f"Error communicating with Ollama for text generation: {e}"
            )

    def generate_from_file(self, prompt: str, file_path: Path) -> str:
        # Fallback: If it's a text-based file, read it and append to prompt
        # For true multimodal (images), Ollama uses 'images' field in payload.
        # For PDF/Audio, Ollama doesn't natively support them in the same way.
        if file_path.suffix.lower() in [".txt", ".md", ".json"]:
            file_content = file_path.read_text(encoding="utf-8")
            full_prompt = f"{prompt}\n\nFile Content:\n{file_content}"
            return self.generate_text(full_prompt)
        else:
            raise NotImplementedError(
                f"OllamaClient does not yet support multimodal files of type {file_path.suffix}"
            )


# Add more clients (e.g., OpenAIClient) as needed
