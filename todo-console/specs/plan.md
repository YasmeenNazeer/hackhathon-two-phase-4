# Implementation Plan: Todo Console (Phase 1)

## 1. Scope and Dependencies
- **In Scope**: CRUD operations, in-memory storage, menu-driven CLI, basic input validation.
- **Out of Scope**: Database persistence, file exports, user accounts, search functionality.
- **External Dependencies**: Python 3.12+, `pytest` (development dependency).

## 2. Key Decisions and Rationale
- **In-Memory Storage**: Using a list of `Task` objects inside a Singleton or long-lived `TaskService`. This fulfills the requirement for Phase 1 simplicity.
- **Layered Architecture**: Decoupling the CLI logic from the Domain and Service layers ensures the core logic is testable without a terminal.
- **Error Handling Strategy**: Use custom exceptions for business rule violations (e.g., `TaskNotFoundError`) and `try-except` blocks in the CLI layer for input validation.

## 3. Interfaces and API Contracts
- `TaskService.add_task(title: str, description: str) -> Task`
- `TaskService.get_all_tasks() -> List[Task]`
- `TaskService.get_task_by_id(task_id: int) -> Optional[Task]`
- `TaskService.delete_task(task_id: int) -> bool`
- `TaskService.toggle_task_completion(task_id: int) -> Task`

## 4. Data Management
- **Source of Truth**: The `tasks` list within `TaskService`.
- **ID Generation**: simple auto-incrementing integer starting from 1.

## 5. Risk Analysis and Mitigation
1. **Invalid User Input**: CLI will wrap all inputs in validation loops.
2. **Missing IDs**: Domain services will raise specific errors when an ID does not exist.
3. **Memory Leaks**: Not a concern for this scale, but standard object references will be managed via the list.

## 6. Definition of Done
- [ ] All T-tasks in `specs/tasks.md` marked complete.
- [ ] Manual verification of all CRUD flows.
- [ ] Code passes PEP 8 linting.
- [ ] No hardcoded strings in the Service layer.
