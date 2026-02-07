# Task CRUD Features

## User Stories
- As a user, I want to create a task with a title and optional description.
- As a user, I want to see a list of my tasks.
- As a user, I want to mark a task as complete/incomplete.
- As a user, I want to edit a task's details.
- As a user, I want to delete a task.

## Data Model (Task)
- `id`: UUID (Primary Key)
- `user_id`: String/UUID (Foreign Key)
- `title`: String
- `description`: String (Optional)
- `is_completed`: Boolean
- `created_at`: Datetime
- `updated_at`: Datetime
