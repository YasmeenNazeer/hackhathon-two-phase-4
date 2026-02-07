# Architecture: Todo Console (Phase 1)

## Overview
The application uses a Layered Architecture to decouple user interface from business logic.

## Layers

### 1. Presentation Layer (`app/cli/`)
- Handles user interaction.
- Displays menus and output formatting.
- Collects and validates raw input strings/numbers.

### 2. Service Layer (`app/services/`)
- Orchestrates domain logic.
- Maintains the in-memory state.
- Provides high-level operations (e.g., `add_task`, `delete_task`).

### 3. Domain Layer (`app/models/`)
- Defines the `Task` entity and business rules.

## Data Persistence
- **In-Memory**: A Python list or dictionary inside the `TaskService`.
- No persistence across app restarts.

## Directory Structure
```
todo-console/
├── app/
│   ├── __init__.py
│   ├── main.py        # Entry point
│   ├── cli/           # UI components
│   ├── services/      # Business logic
│   └── models/        # Entities
├── specs/             # SDD artifacts
└── tests/             # Pytest suite
```
