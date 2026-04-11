from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.RAG.models import ChatRequest, ChatResponse, QueryState, Conversation
from src.db.main import get_session
from src.RAG.app_state import get_pipeline
from src.RAG.Service.pipeline import RAGPipeline
from src.Users.auth import get_current_user
from src.Users.models import User

router = APIRouter()

# =========================
# START CONVERSATION
# =========================
@router.post("/start", status_code=status.HTTP_201_CREATED)
async def start_chat(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    """Starts a new conversation session for the authenticated user."""

    new_conv = Conversation(user_id=current_user.user_id)
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)  

    return {
        "status": "success",
        "conversation_id": new_conv.id,
        "message": "Conversation started successfully"
    }


# =========================
# CHAT QUERY (MAIN PIPELINE)
# =========================
@router.post("/", response_model=ChatResponse)
async def chat_query(
    request: ChatRequest,
    db: AsyncSession = Depends(get_session),
    pipeline: RAGPipeline = Depends(get_pipeline),
    current_user: User = Depends(get_current_user),
):
    """
    Chat pipeline:
    - API → validation + security
    - Pipeline → memory + RAG + persistence
    """

    # Input validation
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # 1. Ownership validation
    res = await db.execute(
        select(Conversation).where(Conversation.id == request.conversation_id)
    )
    conv = res.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conv.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized access")

    # 2. Execute RAG pipeline
    state: QueryState = await pipeline.run(
        request, db, user_id=current_user.user_id
    )

    # 3. Response
    return ChatResponse(
        trace_id=state.trace_id,
        query=state.query,
        answer=state.answer or "No response generated.",
        status="success"
    )


# =========================
# TRACE INSPECTION
# =========================
@router.get("/trace/{trace_id}")
async def get_trace_logs(
    trace_id: str,
    db: AsyncSession = Depends(get_session),
    current_user: User = Depends(get_current_user)
):
    """Fetch full trace logs with ownership validation."""

    result = await db.execute(
        select(QueryState).where(QueryState.trace_id == trace_id)
    )
    state = result.scalar_one_or_none()

    if not state:
        raise HTTPException(status_code=404, detail="Trace ID not found")

    if state.user_id != current_user.user_id:
        raise HTTPException(status_code=403, detail="Unauthorized access")

    return state