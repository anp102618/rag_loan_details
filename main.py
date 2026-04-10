import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from src.RAG.models import QueryState
from sqlalchemy import select

# Service Imports
from src.RAG.Service.extractor import PDFIngestor
from src.RAG.Service.chunker import Chunker
from src.RAG.Service.embedder import OllamaEmbeddings
from src.RAG.Service.retriever import FaissVectorStore
from src.RAG.Service.generator import Generator
from src.RAG.Service.pipeline import RAGPipeline

# Model and DB Imports
from src.RAG.models import ChatRequest, ChatResponse
from src.db.main import get_session 
from src.Utils.logger_setup import get_log

# Global container for singleton RAG components
rag_container = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles server startup:
    1. Initializes all RAG services.
    2. Ingests, chunks, and embeds PDFs from the data directory.
    3. Builds the Vector Store index in memory.
    """
    logger = get_log() 
    logger.info("[STARTUP] Initializing RAG Components...")

    try:
        # 1. Initialize Worker Instances
        ingestor = PDFIngestor()
        chunker = Chunker(chunk_size=800, chunk_overlap=150)
        embedder = OllamaEmbeddings(model="nomic-embed-text")
        generator = Generator(model="qwen2.5:7b")
        
        # 2. Ingest initial documents
        raw_docs = await ingestor.load_pdfs("./Data")
        text_chunks = chunker.chunk(raw_docs)
        
        # 3. Setup Vector Store (nomic-embed-text dimensionality is 768)
        vector_store = FaissVectorStore(dim=768)
        
        if text_chunks:
            logger.info(f"[STARTUP] Embedding {len(text_chunks)} chunks...")
            embeddings = await embedder.embed(text_chunks)
            await vector_store.add(embeddings, text_chunks)
        
        # 4. Initialize the Orchestrator
        rag_container["pipeline"] = RAGPipeline(
            embedder=embedder,
            vectorstore=vector_store,
            generator=generator
        )
        
        logger.info("[STARTUP] RAG Engine Online and Ready.")
        yield
    except Exception as e:
        logger.error(f"[STARTUP-FAILED] {str(e)}")
        raise e
    finally:
        logger.info("[SHUTDOWN] Releasing RAG resources.")
        rag_container.clear()

app = FastAPI(
    title="Local RAG API",
    description="Asynchronous RAG Pipeline using Ollama and FAISS",
    lifespan=lifespan
)

# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    db: AsyncSession = Depends(get_session)
):
    """
    Primary chat endpoint:
    - Validates input via ChatRequest model.
    - Runs the full RAG pipeline (Normalize -> Embed -> Retrieve -> Generate).
    - Persists query state and logs to Postgres via SQLAlchemy.
    - Returns structured ChatResponse.
    """
    pipeline: RAGPipeline = rag_container.get("pipeline")
    
    if not pipeline:
        raise HTTPException(
            status_code=503, 
            detail="Pipeline not initialized. Check server startup logs."
        )

    try:
        # Pass the Pydantic request object to the pipeline
        query_state = await pipeline.run(request, db)
        
        # Explicitly map the SQLModel 'QueryState' result to 'ChatResponse'
        return ChatResponse(
            trace_id=query_state.trace_id,
            query=query_state.query,
            answer=query_state.answer or "No response generated.",
            status="success"
        )

    except Exception as e:
        # Pipeline errors are caught and logged; this re-raises for the API client
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Pipeline Error: {str(e)}"
        )
    


@app.get("/trace/{trace_id}")
async def get_trace_logs(trace_id: str, db: AsyncSession = Depends(get_session)):
    """
    Retrieves the full record, including JSONB logs, for a specific query.
    """
    result = await db.execute(
        select(QueryState).where(QueryState.trace_id == trace_id)
    )
    state = result.scalar_one_or_none()
    
    if not state:
        raise HTTPException(status_code=404, detail="Trace ID not found")
        
    return {
        "trace_id": state.trace_id,
        "query": state.query,
        "answer": state.answer,
        "created_at": state.created_at,
        "logs": state.logs  # This returns your captured internal steps
    }