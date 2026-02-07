from typing import List, Optional
from app.models.task import Task, TaskStatus

class TaskNotFoundError(Exception):
    """Raised when a task with a given ID is not found."""
    pass

class TaskService:
    def __init__(self):
        self._tasks: List[Task] = []
        self._next_id: int = 1

    def add_task(self, title: str, description: str = "") -> Task:
        task = Task(id=self._next_id, title=title, description=description)
        self._tasks.append(task)
        self._next_id += 1
        return task

    def get_all_tasks(self) -> List[Task]:
        return self._tasks

    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def complete_task(self, task_id: int) -> Task:
        task = self.get_task_by_id(task_id)
        if not task:
            raise TaskNotFoundError(f"Task with ID {task_id} not found.")
        task.mark_completed()
        return task

    def delete_task(self, task_id: int) -> bool:
        task = self.get_task_by_id(task_id)
        if not task:
            raise TaskNotFoundError(f"Task with ID {task_id} not found.")
        self._tasks.remove(task)
        return True

    def update_task(self, task_id: int, **kwargs) -> Task:
        task = self.get_task_by_id(task_id)
        if not task:
            raise TaskNotFoundError(f"Task with ID {task_id} not found.")

        if "title" in kwargs:
            task.title = kwargs["title"]
        if "description" in kwargs:
            task.description = kwargs["description"]
        if "status" in kwargs:
            # Assuming status is passed as a string and needs conversion to TaskStatus Enum
            if isinstance(kwargs["status"], str):
                task.status = TaskStatus[kwargs["status"].upper()]
            else:
                task.status = kwargs["status"]
        return task

