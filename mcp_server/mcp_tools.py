from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Session, select

from sdk_mock import McpBaseTool
from backend.app.models.task import Task # Import the Task model

# --- Input Schemas for Tools ---
class AddTaskInput(BaseModel):
    user_id: str = Field(description="The ID of the user for whom to add the task.")
    title: str = Field(description="The title of the task.")
    description: Optional[str] = Field(None, description="The description of the task.")
    category: Optional[str] = Field("Personal", description="The category of the task (e.g., Personal, Work, Shopping).")
    due_date: Optional[datetime] = Field(None, description="The due date for the task.")

class ListTasksInput(BaseModel):
    user_id: str = Field(description="The ID of the user for whom to list tasks.")

class CompleteTaskInput(BaseModel):
    user_id: str = Field(description="The ID of the user who owns the task.")
    task_id: UUID = Field(description="The ID of the task to complete.")

class DeleteTaskInput(BaseModel):
    user_id: str = Field(description="The ID of the user who owns the task.")
    task_id: UUID = Field(description="The ID of the task to delete.")

class UpdateTaskInput(BaseModel):
    user_id: str = Field(description="The ID of the user who owns the task.")
    task_id: UUID = Field(description="The ID of the task to update.")
    title: Optional[str] = Field(None, description="The new title for the task.")
    description: Optional[str] = Field(None, description="The new description for the task.")
    is_completed: Optional[bool] = Field(None, description="The new completion status for the task.")
    category: Optional[str] = Field(None, description="The new category for the task.")
    due_date: Optional[datetime] = Field(None, description="The new due date for the task.")


# --- Tool Implementations ---
async def add_task_impl(session: Session, user_id: str, title: str, description: Optional[str] = None, category: Optional[str] = "Personal", due_date: Optional[datetime] = None) -> str:
    new_task = Task(
        user_id=user_id,
        title=title,
        description=description,
        category=category,
        due_date=due_date
    )
    session.add(new_task)
    session.commit()
    session.refresh(new_task)
    return f"Task '{new_task.title}' (ID: {new_task.id}) for user {new_task.user_id} added successfully."

async def list_tasks_impl(session: Session, user_id: str) -> str:
    tasks = session.exec(select(Task).where(Task.user_id == user_id)).all()
    if not tasks:
        return f"No tasks found for user {user_id}."
    return "\n".join([
        f"ID: {task.id}, Title: {task.title}, Category: {task.category}, Due: {task.due_date.strftime('%Y-%m-%d') if task.due_date else 'None'}, Completed: {'Yes' if task.is_completed else 'No'}"
        for task in tasks
    ])

async def complete_task_impl(session: Session, user_id: str, task_id: UUID) -> str:
    task = session.exec(select(Task).where(Task.id == task_id, Task.user_id == user_id)).first()
    if not task:
        return f"Task with ID {task_id} not found for user {user_id}."
    task.is_completed = True
    task.updated_at = datetime.now(timezone.utc)
    session.add(task)
    session.commit()
    session.refresh(task)
    return f"Task '{task.title}' (ID: {task.id}) for user {task.user_id} marked as complete."

async def delete_task_impl(session: Session, user_id: str, task_id: UUID) -> str:
    task = session.exec(select(Task).where(Task.id == task_id, Task.user_id == user_id)).first()
    if not task:
        return f"Task with ID {task_id} not found for user {user_id}."
    session.delete(task)
    session.commit()
    return f"Task '{task.title}' (ID: {task.id}) for user {task.user_id} deleted."

async def update_task_impl(session: Session, user_id: str, task_id: UUID, title: Optional[str] = None, description: Optional[str] = None, is_completed: Optional[bool] = None, category: Optional[str] = None, due_date: Optional[datetime] = None) -> str:
    task = session.exec(select(Task).where(Task.id == task_id, Task.user_id == user_id)).first()
    if not task:
        return f"Task with ID {task_id} not found for user {user_id}."

    if title is not None:
        task.title = title
    if description is not None:
        task.description = description
    if is_completed is not None:
        task.is_completed = is_completed
    if category is not None:
        task.category = category
    if due_date is not None:
        task.due_date = due_date

    task.updated_at = datetime.now(timezone.utc)
    session.add(task)
    session.commit()
    session.refresh(task)
    return f"Task '{task.title}' (ID: {task.id}) for user {task.user_id} updated successfully."

# --- McpBaseTool Definitions ---
ADD_TASK_TOOL = McpBaseTool(
    name="add_task",
    description="Adds a new task for a user.",
    inputSchema=AddTaskInput.model_json_schema()
)

LIST_TASKS_TOOL = McpBaseTool(
    name="list_tasks",
    description="Lists all tasks for a user.",
    inputSchema=ListTasksInput.model_json_schema()
)

COMPLETE_TASK_TOOL = McpBaseTool(
    name="complete_task",
    description="Marks a user's task as complete.",
    inputSchema=CompleteTaskInput.model_json_schema()
)

DELETE_TASK_TOOL = McpBaseTool(
    name="delete_task",
    description="Deletes a user's task.",
    inputSchema=DeleteTaskInput.model_json_schema()
)

UPDATE_TASK_TOOL = McpBaseTool(
    name="update_task",
    description="Updates the title, description, category, due date, or completion status of an existing user's task.",
    inputSchema=UpdateTaskInput.model_json_schema()
)

# A dictionary to easily access implementations
TOOL_IMPLEMENTATIONS = {
    ADD_TASK_TOOL.name: add_task_impl,
    LIST_TASKS_TOOL.name: list_tasks_impl,
    COMPLETE_TASK_TOOL.name: complete_task_impl,
    DELETE_TASK_TOOL.name: delete_task_impl,
    UPDATE_TASK_TOOL.name: update_task_impl,
}

# List of all tools for easy registration
ALL_TOOLS = [
    ADD_TASK_TOOL,
    LIST_TASKS_TOOL,
    COMPLETE_TASK_TOOL,
    DELETE_TASK_TOOL,
    UPDATE_TASK_TOOL,
]