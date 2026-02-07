# Chatbot Feature

This document describes the chatbot feature for managing todos.

## Objective

Create an AI-powered chatbot interface for managing todos through natural language using MCP (Model Context Protocol).

## Tools

The following tools are available to the chatbot:

- `list_tasks(user_id)`: Lists all tasks for a given user.
- `create_task(user_id, title, description?)`: Creates a new task for a given user.
- `delete_task(user_id, task_id)`: Deletes a task for a given user.
- `update_task(user_id, task_id, title?, description?)`: Updates a task for a given user.

## Agent Behavior

- The agent detects the user's intent from the natural language message.
- It calls the appropriate MCP tool to perform the requested action.
- It confirms actions with the user before executing them.
- It handles errors gracefully and provides helpful feedback to the user.

## Frontend

- The frontend is a ChatKit-based UI.
- It communicates with the backend via the `/api/{user_id}/chat` endpoint.
- It displays the conversation history.
- It has a clean and minimal user interface.
- It does not access the database directly.

## Backend

- The backend provides the `/api/{user_id}/chat` endpoint.
- It contains the agent logic for processing messages.
- It communicates with the MCP server to execute tools.

## MCP Server

- The MCP server exposes the todo management tools as a service.
- It is a separate process that the backend communicates with.
