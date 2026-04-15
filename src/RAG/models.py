from typing import Dict, Optional, Any, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field as SQLField
from sqlalchemy import Column, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel, Field as PydanticField
from pgvector.sqlalchemy import Vector

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




class ChunkModel(SQLModel, table=True):
    __tablename__ = "document_chunks"

    # 🔹 Primary key - Now Integer
    id: Optional[int] = SQLField(default=None, primary_key=True)

    # 🔹 Core content
    chunk_text: str = SQLField(nullable=False)

    # 🔹 Embedding (768 for nomic)
    embedding: Optional[List[float]] = SQLField(
        default=None,
        sa_column=Column(Vector(768))
    )
    
    confidence_score: float = SQLField(default=0.0)

    # 🔹 Metadata JSON
    chunk_metadata: Dict[str, Any] = SQLField(
        default_factory=dict,
        sa_column=Column(JSONB)
    )

    # 🔹 Chunk linking - FIXED: Changed from uuid.UUID to int
    prev_chunk_id: Optional[int] = SQLField(default=None, foreign_key="document_chunks.id")
    next_chunk_id: Optional[int] = SQLField(default=None, foreign_key="document_chunks.id")

    created_at: datetime = SQLField(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None)
    )

    context_chunks: List[Dict[str, Any]] = SQLField(
        default_factory=list,
        sa_column=Column(JSONB)
    )



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