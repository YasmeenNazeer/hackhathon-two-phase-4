from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class TaskStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"

@dataclass
class Task:
    id: int
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)

    def mark_completed(self) -> None:
        self.status = TaskStatus.COMPLETED

    def __str__(self) -> str:
        status_icon = "X" if self.status == TaskStatus.COMPLETED else " "
        return f"[{self.id}] [{status_icon}] {self.title} - {self.description}"
