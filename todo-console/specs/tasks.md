# Tasks: Todo Console (Phase 1)

## T-001: Project Setup
- Define directory structure.
- Initialize `__init__.py` files.
- Setup `main.py` entry point.

## T-002: Domain Models
- Implement `Task` class in `app/models/task.py`.
- Define fields: ID, title, description, status, created_at.

## T-003: Task Service
- Implement `TaskService` in `app/services/task_service.py`.
- Support internal list storage.
- Implement `add_task(title, description)`.
- Implement `get_all_tasks()`.
- Implement `update_task(id, **kwargs)`.
- Implement `delete_task(id)`.

## T-004: CLI Presentation
- Implement main loop in `app/cli/menu.py`.
- Handle menu navigation:
    1. View Tasks
    2. Add Task
    3. Complete Task
    4. Delete Task
    5. Exit
- Add input validation for menu choices and numeric IDs.

## T-005: Integration & Entry Point
- Wire `main.py` to launch the CLI menu.
- Verify end-to-end CRUD flow in the console.
