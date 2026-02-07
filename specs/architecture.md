# Architecture

## Monorepo Structure
- `backend/`: FastAPI application.
- `frontend/`: Next.js application.
- `mcp_server/`: FastAPI application for Model Context Protocol tools.
- `contracts/`: (Optional) Shared types or API definitions.

## System Flow
1. User interacts with the Next.js frontend, including a new chat interface.
2. Frontend manages auth via Better Auth.
3. API requests for task management are sent to FastAPI with a JWT in the `Authorization` header.
4. Chat messages are sent to the `/api/{user_id}/chat` endpoint in the FastAPI backend.
5. The FastAPI backend utilizes an AI agent (powered by Google Generative AI - Gemini) to interpret natural language commands.
6. The AI agent, for certain commands, calls the appropriate tool on the `mcp_server`.
7. The `mcp_server` executes the task management operations (e.g., list, create, delete, update tasks).
8. FastAPI validates JWTs and scopes queries to the authenticated `user_id`.
9. Data (tasks, conversations, messages) is persisted in Neon PostgreSQL.
