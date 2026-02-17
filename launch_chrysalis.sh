#!/bin/bash
echo "Starting Chrysalis Services..."

# Start Ollama server in background
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &>/dev/null &
    sleep 3
fi

# Start Open WebUI
source .venv/bin/activate
open-webui serve