#!/bin/bash
# Start the React frontend dev server
# Run from project root: bash start_frontend.sh

cd "$(dirname "$0")/frontend"

echo "=== SarcasmSense Frontend ==="
echo "Installing dependencies..."
npm install

echo ""
echo "Starting Vite dev server on http://localhost:5173"
echo ""

npm run dev
