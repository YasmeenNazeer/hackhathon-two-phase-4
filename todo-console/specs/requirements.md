# Requirements: Todo Console (Phase 1)

## Functional Requirements

### 1. Task Management (CRUD)
- **Create**: Add a new task with a title and optional description.
- **Read**: List all tasks with their status (Pending/Done).
- **Update**: Edit task title/description or mark as completed.
- **Delete**: Remove a task by its ID.

### 2. User Interface
- Simple menu-driven navigation.
- Clear success and error messages.
- Input validation for IDs and menu choices.

## Non-Functional Requirements
- **Performance**: Instant execution of all operations.
- **Reliability**: No crashes on invalid numeric or string input.
- **Maintainability**: Clean project structure separating UI from logic.

## Constraints
- In-memory storage only.
- Single-user access.
- Console output only.
