# Chrysalis Project Setup Guide

This document outlines the steps to set up the Chrysalis research framework environment, including local LLM configuration.

## 1. Python Environment Setup

The project uses a Python virtual environment to manage dependencies.

### Prerequisites
- Python 3.10 or higher
- `pip`

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/nickydimes/chrysalis.git
    cd chrysalis
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install the project in editable mode:**
    ```bash
    pip install -e .
    ```

## 2. LLM Configuration

Chrysalis supports both cloud-based (Google Gemini) and local (Ollama) LLMs.

### Google Gemini API
1.  Obtain an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).
2.  Set the `GEMINI_API_KEY` environment variable:
    ```bash
    export GEMINI_API_KEY="your_api_key_here"
    ```

### Local LLM (Ollama)
The project is optimized for use with Ollama for local LLM inference and embeddings.

1.  **Install Ollama:** Follow instructions at [ollama.com](https://ollama.com).
2.  **Start the Ollama server:**
    ```bash
    ollama serve
    ```
3.  **Pull required models:**
    - **For Embeddings (RAG):**
      ```bash
      ollama pull nomic-embed-text
      ```
    - **For Text Generation (Hypotheses, Interpretation):**
      ```bash
      ollama pull llama3.3:70b-instruct-q4_K_M  # Or your preferred model
      ```
    - **For Code Generation (Tool Scaffolding, Refactoring):**
      ```bash
      ollama pull qwen2.5-coder:32b  # Or your preferred coder model
      ```

## 3. Verify the Installation

Run the project test suite to ensure everything is configured correctly:
```bash
python chrysalis_cli.py test
```

## 4. Usage

The project is orchestrated via a central CLI:
```bash
python chrysalis_cli.py --help
```
