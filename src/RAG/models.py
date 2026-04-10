from typing import Optional, Any, List
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field as SQLField
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from pydantic import BaseModel, Field

# 1. Database Model (SQLModel)
class QueryState(SQLModel, table=True):
    __tablename__ = "query_states"

    trace_id: str = SQLField(primary_key=True, index=True)
    query: str
    answer: Optional[str] = None

    logs: Any = SQLField(
        default_factory=list,
        sa_column=Column(JSONB, nullable=False)
    )

    created_at: datetime = SQLField(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        nullable=False
    )

# 2. API Request Model (Pydantic)
class ChatRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="The user's question or search query.",
        # Use json_schema_extra for Pydantic v2 compatibility
        json_schema_extra={"example": "What is the total revenue mentioned in the PDF?"}
    )

# 3. API Response Model (Pydantic)
class ChatResponse(BaseModel):
    trace_id: str
    query: str
    answer: str
    status: str = "success"