import os
# Set dummy environment variables before importing app components
os.environ["DATABASE_URL"] = "sqlite:///"
os.environ["JWT_SECRET"] = "testclient_secret"

from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine
from sqlmodel.pool import StaticPool
import pytest
import jwt
from datetime import datetime, timedelta

from app.main import app
from app.database import get_session
from app.config import settings

# Setup in-memory SQLite for testing
engine = create_engine(
    "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
)

@pytest.fixture(name="session")
def session_fixture():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="client")
def client_fixture(session: Session):
    def get_session_override():
        return session
    app.dependency_overrides[get_session] = get_session_override
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

@pytest.fixture(name="auth_headers")
def auth_headers_fixture():
    payload = {
        "sub": "test_user_123",
        "exp": datetime.utcnow() + timedelta(days=1)
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.ALGORITHM)
    return {"Authorization": f"Bearer {token}"}

def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Todo API"}

def test_create_task(client: TestClient, auth_headers: dict):
    response = client.post(
        "/api/tasks/",
        json={"title": "Test Task"},
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Test Task"
    assert data["user_id"] == "test_user_123"
    assert data["is_completed"] is False

def test_read_tasks(client: TestClient, auth_headers: dict):
    # Create a task first
    client.post("/api/tasks/", json={"title": "Task 1"}, headers=auth_headers)

    response = client.get("/api/tasks/", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["title"] == "Task 1"

def test_unauthorized(client: TestClient):
    response = client.get("/api/tasks/")
    assert response.status_code == 401 # Bearer token missing
