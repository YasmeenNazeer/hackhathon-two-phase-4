from sqlmodel import create_engine, Session, SQLModel
from .config import settings
# Import all models to ensure they're registered with SQLModel before creating tables
from .models import Task, Conversation, Message

engine = create_engine(settings.DATABASE_URL)

def init_db():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
