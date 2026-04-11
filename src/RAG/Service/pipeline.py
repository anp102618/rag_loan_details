import uuid
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.RAG.models import QueryState, ChatRequest
from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException


class RAGPipeline:
    def __init__(self, embedder, vectorstore, generator, normalizer, memory_manager) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.generator = generator
        self.normalizer = normalizer
        self.memory_manager = memory_manager  

    @track_performance
    async def run(self, request: ChatRequest, db: AsyncSession, user_id: int) -> QueryState:

        # 1. Setup Tracing
        trace_id = f"CHAT-{uuid.uuid4().hex[:8].upper()}"
        logger = setup_logger(trace_id)
        token = current_logger.set(logger)

        # 2. Sequence Management
        res = await db.execute(
            select(func.count(QueryState.trace_id))
            .where(QueryState.conversation_id == request.conversation_id)
        )
        sequence_id = (res.scalar() or 0) + 1

        state = QueryState(
            trace_id=trace_id,
            conversation_id=request.conversation_id,
            user_id=user_id,
            sequence_id=sequence_id,
            query=request.query
        )

        try:
            logger.info(f"Starting RAG flow for {trace_id}")

            #3. MEMORY FETCH (NEW)
            stm = await self.memory_manager.get_stm(db, request.conversation_id)
            ltm = await self.memory_manager.get_ltm(db, request.conversation_id)

            logger.info("Memory fetched successfully")

            # 4. Normalize Query
            norm_query = self.normalizer.normalize(request.query)

            # 5. CONTEXT-AWARE QUERY (NEW)
            enriched_query = f"""
            Long-term memory:
            {ltm}

            Recent conversation:
            {stm}

            User query:
            {norm_query}
            """

            # 6. Embedding
            q_emb = await self.embedder.embed([enriched_query])

            # 7. Retrieval
            docs = await self.vectorstore.search(q_emb[0], k=5)

            state.retrieved_chunks = "\n".join(docs)[:4000]

            # 8. GENERATION WITH MEMORY (NEW)
            generation_input = f"""
            Context:
            {state.retrieved_chunks}

            Conversation Memory:
            {ltm}

            Recent Turns:
            {stm}

            Query:
            {norm_query}
            """

            state.answer = await self.generator.generate(norm_query, generation_input)

            # 9. UPDATE MEMORY (NEW)
            interaction = f"User: {request.query}\nAssistant: {state.answer}"
            state.memory = await self.memory_manager.update_ltm(ltm, interaction)

            # 10. Save Logs
            if hasattr(logger, "memory_handler"):
                state.logs = logger.memory_handler.logs

            # 11. Persist
            db.add(state)
            await db.commit()
            await db.refresh(state)

            logger.info("RAG flow completed successfully")

            return state

        except Exception as e:
            state.answer = "An error occurred during generation."
            db.add(state)
            await db.commit()
            raise CustomException(e, logger=logger)

        finally:
            current_logger.reset(token)