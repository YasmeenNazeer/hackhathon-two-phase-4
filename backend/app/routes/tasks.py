from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from uuid import UUID
from datetime import datetime, timezone

from ..database import get_session
from ..models import Task, TaskCreate, TaskUpdate
from ..auth import get_current_user_id

router = APIRouter(prefix="/tasks", tags=["tasks"])

@router.post("/", response_model=Task)
def create_task(
    task_in: TaskCreate,
    session: Session = Depends(get_session),
    user_id: str = Depends(get_current_user_id)
):
    # Validate inputs
    if not task_in.title.strip():
        raise HTTPException(status_code=400, detail="Task title cannot be empty")

    if len(task_in.title) > 200:
        raise HTTPException(status_code=400, detail="Task title too long, max 200 characters")

    if task_in.description and len(task_in.description) > 1000:
        raise HTTPException(status_code=400, detail="Task description too long, max 1000 characters")

    try:
        db_task = Task.model_validate(task_in, update={"user_id": user_id})
        session.add(db_task)
        session.commit()
        session.refresh(db_task)
        return db_task
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to create task")

@router.get("/", response_model=List[Task])
def read_tasks(
    session: Session = Depends(get_session),
    user_id: str = Depends(get_current_user_id)
):
    statement = select(Task).where(Task.user_id == user_id)
    return session.exec(statement).all()

@router.get("/{id}", response_model=Task)
def read_task(
    id: UUID,
    session: Session = Depends(get_session),
    user_id: str = Depends(get_current_user_id)
):
    task = session.get(Task, id)
    if not task or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.put("/{id}", response_model=Task)
def update_task(
    id: UUID,
    task_update: TaskUpdate,
    session: Session = Depends(get_session),
    user_id: str = Depends(get_current_user_id)
):
    db_task = session.get(Task, id)
    if not db_task or db_task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    # Validate inputs
    if task_update.title and not task_update.title.strip():
        raise HTTPException(status_code=400, detail="Task title cannot be empty")

    if task_update.title and len(task_update.title) > 200:
        raise HTTPException(status_code=400, detail="Task title too long, max 200 characters")

    if task_update.description and len(task_update.description) > 1000:
        raise HTTPException(status_code=400, detail="Task description too long, max 1000 characters")

    try:
        task_data = task_update.model_dump(exclude_unset=True)
        for key, value in task_data.items():
            setattr(db_task, key, value)

        db_task.updated_at = datetime.now(timezone.utc)
        session.add(db_task)
        session.commit()
        session.refresh(db_task)
        return db_task
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail="Failed to update task")

@router.delete("/{id}")
def delete_task(
    id: UUID,
    session: Session = Depends(get_session),
    user_id: str = Depends(get_current_user_id)
):
    task = session.get(Task, id)
    if not task or task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    session.delete(task)
    session.commit()
    return {"ok": True}

@router.patch("/{id}/complete", response_model=Task)
def toggle_task_complete(
    id: UUID,
    session: Session = Depends(get_session),
    user_id: str = Depends(get_current_user_id)
):
    db_task = session.get(Task, id)
    if not db_task or db_task.user_id != user_id:
        raise HTTPException(status_code=404, detail="Task not found")

    db_task.is_completed = not db_task.is_completed
    db_task.updated_at = datetime.now(timezone.utc)
    session.add(db_task)
    session.commit()
    session.refresh(db_task)
    return db_task
