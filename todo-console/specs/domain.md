# Domain Model: Todo Console (Phase 1)

## Entities

### Task
| Field | Type | Description |
|---|---|---|
| id | int | Unique identifier |
| title | str | Name of the task |
| description | str | Detailed info (optional) |
| status | str | 'pending' or 'completed' |
| created_at | datetime | Timestamp of creation |

## Value Objects (Potential)
- **TaskStatus**: Enum for task states.

## Domain Services
- **TaskService**: Manages the collection of tasks in memory.
