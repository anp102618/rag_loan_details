import uuid
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.models import QueryState, ChatRequest
from .normalizer import Normalizer


class RAGPipeline:
    """
    The central orchestrator that connects all RAG components into a 
    unified execution flow with logging, tracing, and persistence.
    """

    def __init__(self, embedder, vectorstore, generator) -> None:
        """
        Initializes the pipeline with pre-configured worker components.
        """
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.generator = generator
        self.normalizer = Normalizer()

    @track_performance
    async def run(self, request: ChatRequest, db: AsyncSession) -> QueryState:
        """
        Executes the full RAG cycle: Normalize -> Embed -> Retrieve -> Generate.
        
        Args:
            request (ChatRequest): Validated Pydantic request object.
            db (AsyncSession): SQLModel/SQLAlchemy async session.
        """
        # 1. TRACING SETUP
        # Generating the primary key for our QueryState table
        trace_id = f"CHAT-{uuid.uuid4().hex[:8].upper()}"
        logger = setup_logger(trace_id)
        token = current_logger.set(logger)

        # Initialize our SQLModel instance
        state = QueryState(trace_id=trace_id, query=request.query)

        try:
            logger.info(f"[START] Pipeline execution | trace_id={trace_id}")

            # 2. PREPARATION: Normalization
            logger.info("[STEP 1] Normalizing query")
            normalized_query = self.normalizer.normalize(request.query)
            
            # 3. VECTORIZATION: Embedding
            logger.info("[STEP 2] Generating query embedding")
            q_emb_list: List[List[float]] = await self.embedder.embed([normalized_query])
            
            if not q_emb_list:
                raise ValueError("The embedding service returned an empty result.")
            
            q_emb = q_emb_list[0]

            # 4. STORAGE/RETRIEVAL: Fetch Context
            logger.info("[STEP 3] Retrieving relevant context from FAISS")
            docs = await self.vectorstore.search(q_emb, k=5)
            
            # Construct context block (clamped to 4000 chars for LLM safety)
            context = "\n".join(docs)[:4000]
            logger.info(f"[STEP 3] Context built | chunks={len(docs)} | chars={len(context)}")

            # 5. GENERATION: Produce Answer
            logger.info("[STEP 4] Querying LLM for final answer")
            answer = await self.generator.generate(normalized_query, context)
            state.answer = answer

            # 6. LOGGING: Capture internal traces for JSONB storage
            if hasattr(logger, "memory_handler"):
                state.logs = logger.memory_handler.logs

            # 7. PERSISTENCE: Save to Postgres
            db.add(state)
            await db.commit()
            await db.refresh(state)

            logger.info(f"[SUCCESS] Pipeline complete | trace_id={trace_id}")
            return state

        except Exception as e:
            logger.error(f"[ERROR] Pipeline failed: {str(e)}")
            
            # Recovery: Persist the failure state so we don't lose the trace
            state.answer = "I'm sorry, I encountered an error while processing your request."
            if hasattr(logger, "memory_handler"):
                state.logs = logger.memory_handler.logs
            
            db.add(state)
            await db.commit()
            
            raise CustomException(e, logger=logger)

        finally:
            # Clean up the ContextVar to prevent log bleeding
            current_logger.reset(token)