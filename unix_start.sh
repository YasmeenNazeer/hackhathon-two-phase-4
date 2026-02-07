#!/bin/bash

# Shell script to start all services for the task management system

echo "Starting Task Management System..."

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required commands
if ! command_exists python; then
    echo "Error: python is not installed or not in PATH"
    exit 1
fi

if ! command_exists npm; then
    echo "Error: npm is not installed or not in PATH"
    exit 1
fi

# Start MCP server in background
echo "Starting MCP Server on port 8001..."
cd mcp_server
python main.py &
MCP_PID=$!
cd ..

sleep 3

# Start backend server in background
echo "Starting Backend Server on port 8000..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

sleep 3

# Start frontend in background
echo "Starting Frontend on port 3000..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "All services started!"
echo "- MCP Server: http://localhost:8001"
echo "- Backend: http://localhost:8000" 
echo "- Frontend: http://localhost:3000"
echo ""
echo "Process IDs:"
echo "- MCP Server PID: $MCP_PID"
echo "- Backend PID: $BACKEND_PID"
echo "- Frontend PID: $FRONTEND_PID"
echo ""
echo "To stop all services, run: pkill -P $$"

# Wait for all background processes
wait $MCP_PID $BACKEND_PID $FRONTEND_PID