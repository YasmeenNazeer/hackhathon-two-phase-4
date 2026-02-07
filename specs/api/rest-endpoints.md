# REST API Endpoints

Base URL: `/api`

## Tasks
- `GET /tasks`: List all tasks for the authenticated user.
- `POST /tasks`: Create a new task.
- `GET /tasks/{id}`: Get details of a specific task.
- `PUT /tasks/{id}`: Update a task.
- `DELETE /tasks/{id}`: Delete a task.
- `PATCH /tasks/{id}/complete`: Toggle completion status.

## Chat
- `POST /{user_id}/chat`: Send a message to the AI chatbot and receive a response.

## Auth (Handled by Better Auth on Frontend + Middleware on Backend)
- Backend needs to trust JWTs issued by Better Auth.
