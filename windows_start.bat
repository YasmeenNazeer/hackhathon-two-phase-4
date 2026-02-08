@echo off
REM Batch script to start all services for the task management system

echo Starting Task Management System...

REM Set PYTHONPATH to include user packages
set PYTHONPATH=C:/Users/JOJIS COMPUTERS/AppData/Roaming/Python/Python311/site-packages;%PYTHONPATH%

REM Start MCP server in background
echo Starting MCP Server on port 8001...
start cmd /c "cd mcp_server && set PYTHONPATH=%PYTHONPATH% && python main.py"

timeout /t 3 /nobreak >nul

REM Start backend server in background
echo Starting Backend Server on port 8000...
start cmd /c "cd backend && set PYTHONPATH=%PYTHONPATH% && python main.py"

timeout /t 3 /nobreak >nul

REM Start frontend in background
echo Starting Frontend on port 3000...
start cmd /c "cd frontend && npm run dev"

echo.
echo All services started!
echo - MCP Server: http://localhost:8001
echo - Backend: http://localhost:8000
echo - Frontend: http://localhost:3000
echo.
echo Press Ctrl+C in any terminal window to stop a service.
pause