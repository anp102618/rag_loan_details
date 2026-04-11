from typing import Optional, Any, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field as SQLField
from sqlalchemy import Column, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel, Field as PydanticField
import uuid


# =========================
# Conversation Model
# =========================
class Conversation(SQLModel, table=True):
    __tablename__ = "conversation"

    id: str = SQLField(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        index=True
    )

    user_id: int = SQLField(
        foreign_key="users.user_id",
        nullable=False,
        index=True
    )

    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))


# =========================
# Query State Model
# =========================
class QueryState(SQLModel, table=True):
    __tablename__ = "query_states"

    __table_args__ = (
        Index("idx_conversation_sequence", "conversation_id", "sequence_id"),
    )

    trace_id: str = SQLField(
        default_factory=lambda: str(uuid.uuid4()),
        primary_key=True,
        index=True
    )

    conversation_id: str = SQLField(
        foreign_key="conversation.id",   # ✅ FIXED
        nullable=False,
        index=True
    )

    user_id: int = SQLField(nullable=False)

    sequence_id: int = SQLField(nullable=False)

    query: str

    answer: Optional[str] = None

    retrieved_chunks: Optional[str] = SQLField(
        default=None,
        sa_column=Column(Text)
    )

    memory: Optional[str] = SQLField(
        default=None,
        sa_column=Column(Text)
    )
    
    logs: List[Any] = SQLField(
        default_factory=list,
        sa_column=Column(JSONB)
    )

    created_at: datetime = SQLField(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

# =========================
# API Schemas
# =========================
class ChatRequest(BaseModel):
    conversation_id: str
    query: str = PydanticField(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    status: str = "success"