from typing import Optional, List
from sqlmodel import Field, Relationship, SQLModel
from datetime import datetime, timezone

class Conversation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    metadata_info: Optional[str] = Field(default=None)  # For storing conversation metadata

    messages: List["Message"] = Relationship(back_populates="conversation")

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: Optional[int] = Field(default=None, foreign_key="conversation.id")
    sender: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    message_type: str = Field(default="standard")  # Type of message: standard, memory, reminder, etc.
    tags: Optional[str] = Field(default=None)  # For categorizing important messages

    conversation: Optional[Conversation] = Relationship(back_populates="messages")
