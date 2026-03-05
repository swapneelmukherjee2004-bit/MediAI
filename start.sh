#!/bin/bash
# MediAI Disease Detection System - Startup Script
# Starts both the FastAPI backend and Next.js frontend

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting MediAI Disease Detection System..."
echo ""

# Start backend in background
echo "⚕️  Starting FastAPI backend on port 8000..."
cd "$ROOT_DIR/backend"

# Activate the virtual environment first
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    source "$ROOT_DIR/.venv/bin/activate"
fi

uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

echo "  Backend PID: $BACKEND_PID"
sleep 2

# Start frontend
echo "💻 Starting Next.js frontend on port 3000..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ System started!"
echo "  Backend API: http://localhost:8000"
echo "  Frontend:    http://localhost:3000"
echo "  API Docs:    http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle shutdown gracefully
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Done.'" INT TERM

wait
