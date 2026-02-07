# Todo-Console Constitution

## Core Principles

### I. Simplicity
Implementation must prioritize readability and ease of maintenance. Avoid over-engineering and focus on fulfilling the current requirements.

### II. SDD First
All development follows the Spec-Driven Development cycle. No code is written before specifications are approved and tasks are defined.

### III. In-Memory Persistence (Phase 1)
Data storage is restricted to memory structures (lists/dictionaries). No database or file I/O is permitted for task storage in this phase.

### IV. Functional Reliability
Input validation is mandatory. The application must handle invalid user inputs gracefully without crashing.

## Tech Stack
- **Language**: Python 3.12+
- **Environment**: uv / standard Python venv
- **Testing**: pytest (if applicable)

## Quality Standards
- PEP 8 compliance.
- Type hinting for all public methods and functions.
- Documentation for non-trivial logic.

**Version**: 1.0.0 | **Ratified**: 2025-12-31
