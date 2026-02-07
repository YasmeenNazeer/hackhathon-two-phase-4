from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel

class Task(SQLModel, table=True):
    id: Optional[UUID] = Field(default_factory=uuid4, primary_key=True)
    user_id: str = Field(index=True)
    title: str
    description: Optional[str] = None
    is_completed: bool = Field(default=False)
    category: Optional[str] = Field(default="Personal")
    due_date: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)

class TaskCreate(SQLModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = "Personal"
    due_date: Optional[datetime] = None

class TaskUpdate(SQLModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_completed: Optional[bool] = None
    category: Optional[str] = None
    due_date: Optional[datetime] = None
