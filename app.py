import os
import hashlib
import secrets
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timedelta
import uuid
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize the main app
app = FastAPI(title="Elevate Task Management Backend", docs_url="/docs", redoc_url="/redoc")

# Add CORS middleware to allow requests from your Vercel frontend
origins = [
    "https://*.vercel.app",  # Allow all Vercel domains
    "http://localhost:3000",  # For local development
    "http://localhost:3001",  # Alternative local dev port
    "https://yasmeennazeer-taskmanagement.hf.space",  # Your backend domain
    "https://*.hf.space",  # Allow other Hugging Face spaces if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage (replace with database in production)
users_db = {}
sessions_db = {}
tasks_db = []

# Pydantic models
class UserRegistration(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class Task(BaseModel):
    id: Optional[str] = None
    user_id: str
    title: str
    description: Optional[str] = None
    category: Optional[str] = "Personal"
    due_date: Optional[datetime] = None
    is_completed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    due_date: Optional[datetime] = None
    is_completed: Optional[bool] = None

# Utility functions
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = secrets.token_urlsafe(32)  # Simplified token for demo
    return encoded_jwt

# Routes
@app.get("/")
def root():
    return {"message": "Elevate Task Management Backend on Hugging Face Spaces", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "platform": "huggingface-spaces"}

# Add an endpoint to check environment variables
@app.get("/config")
def config_check():
    return {
        "database_url_set": bool(os.getenv("DATABASE_URL")),
        "port": os.getenv("PORT", "7860"),
        "environment": "huggingface-spaces"
    }

# Authentication endpoints
@app.post("/api/register")
def register_user(user_data: UserRegistration):
    # Check if user already exists
    for user in users_db.values():
        if user["email"] == user_data.email:
            raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    user_id = str(uuid.uuid4())
    hashed_pw = hash_password(user_data.password)
    
    new_user = {
        "id": user_id,
        "name": user_data.name,
        "email": user_data.email,
        "hashed_password": hashed_pw,
        "created_at": datetime.now()
    }
    
    users_db[user_id] = new_user
    
    # Create access token
    access_token = create_access_token(data={"sub": user_id})
    
    return {
        "message": "User registered successfully",
        "user_id": user_id,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/api/login")
def login_user(login_data: UserLogin):
    # Find user by email
    user = None
    for u in users_db.values():
        if u["email"] == login_data.email:
            user = u
            break
    
    if not user or not verify_password(login_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    # Create access token
    access_token = create_access_token(data={"sub": user["id"]})
    
    return {
        "message": "Login successful",
        "user_id": user["id"],
        "access_token": access_token,
        "token_type": "bearer"
    }

# Task endpoints
@app.get("/api/tasks/{user_id}", response_model=List[Task])
def get_tasks(user_id: str):
    user_tasks = [task for task in tasks_db if task.user_id == user_id]
    return user_tasks

@app.post("/api/tasks", response_model=Task)
def create_task(task: Task):
    task.id = str(uuid.uuid4())
    task.created_at = datetime.now()
    task.updated_at = datetime.now()
    tasks_db.append(task)
    return task

@app.put("/api/tasks/{task_id}", response_model=Task)
def update_task(task_id: str, task_update: TaskUpdate):
    for i, task in enumerate(tasks_db):
        if task.id == task_id:
            update_data = task_update.dict(exclude_unset=True)
            updated_task = task.copy(update=update_data)
            updated_task.updated_at = datetime.now()
            tasks_db[i] = updated_task
            return updated_task
    raise HTTPException(status_code=404, detail="Task not found")

@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str):
    global tasks_db
    tasks_db = [task for task in tasks_db if task.id != task_id]
    return {"message": "Task deleted successfully"}

@app.patch("/api/tasks/{task_id}/complete")
def complete_task(task_id: str):
    for i, task in enumerate(tasks_db):
        if task.id == task_id:
            task.is_completed = True
            task.updated_at = datetime.now()
            tasks_db[i] = task
            return task
    raise HTTPException(status_code=404, detail="Task not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))