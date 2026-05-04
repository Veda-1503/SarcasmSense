#!/bin/bash
# Start the FastAPI backend server
# Run from project root: bash start_backend.sh

cd "$(dirname "$0")/backend"

echo "=== SarcasmSense Backend ==="
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
python -m spacy download en_core_web_sm --quiet 2>/dev/null || true

echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

# Add backend directory to PYTHONPATH so modules resolve
PYTHONPATH="$(pwd)" uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
