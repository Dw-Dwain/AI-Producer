#!/bin/bash
# Startup script for deployment on RunPod/Linux instances

# Get the script directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Install requirements if not present (optional fallback)
# pip install -r requirements.txt

# Start the application
echo "Starting AI Drama Production Studio Gradio Server..."
python app.py --port 7860
