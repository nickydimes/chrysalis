# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CHRYSALIS_PROJECT_ROOT=/app

# Set work directory
WORKDIR /app

# Install system dependencies
# We need Node.js for the Gemini MCP server
RUN apt-get update && apt-get install -y
    curl
    git
    build-essential
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    && apt-get install -y nodejs
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Install Gemini MCP server dependencies
WORKDIR /app/gemini-mcp
RUN npm install && npm run build

# Switch back to root
WORKDIR /app

# Install the project in editable mode
RUN pip install -e .

# Expose ports (if needed, e.g., for Streamlit or MCP server)
EXPOSE 8501 3000

# Default command
CMD ["python", "chrysalis_cli.py", "--help"]
