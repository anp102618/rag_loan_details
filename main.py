from fastapi import FastAPI
from contextlib import asynccontextmanager

# Services
from src.RAG.Service.extractor import PDFIngestor
from src.RAG.Service.chunker import Chunker
from src.RAG.Service.embedder import OllamaEmbeddings
from src.RAG.Service.retriever import FaissVectorStore
from src.RAG.Service.generator import Generator
from src.RAG.Service.pipeline import RAGPipeline
from src.RAG.Service.normalizer import Normalizer
from src.RAG.Utils.memory import MemoryManager

# Utils
from src.Utils.logger_setup import get_log
from src.Utils.exception_handler import CustomException

from src.RAG.app_state import rag_container

# Routers
from src.RAG.routes import router as chat_router
from src.Users.routes import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger = get_log()
    logger.info("[STARTUP] Initializing RAG Components...")

    try:
        # =========================
        # 1. Initialize Services
        # =========================
        try:
            ingestor = PDFIngestor()
            chunker = Chunker(chunk_size=500, chunk_overlap=100)
            embedder = OllamaEmbeddings(model="nomic-embed-text")
            generator = Generator(model="qwen2.5:7b")
            normalizer = Normalizer()
            memory_manager = MemoryManager(stm_k=3, max_memory_chars=1000)

            logger.info("Services initialized")

        except Exception as e:
            raise CustomException(e, logger=logger)

        # =========================
        # 2. Load Documents
        # =========================
        try:
            raw_docs = await ingestor.load_pdfs("./Data")

            if not raw_docs:
                logger.warning("No PDFs found in ./Data")

        except Exception as e:
            raise CustomException(e, logger=logger)

        # =========================
        # 3. Chunking
        # =========================
        try:
            text_chunks = chunker.chunk(raw_docs) if raw_docs else []

        except Exception as e:
            raise CustomException(e, logger=logger)

        # =========================
        # 4. Vector Store Setup
        # =========================
        try:
            vector_store = FaissVectorStore(dim=768)

            if text_chunks:
                logger.info(f"Embedding {len(text_chunks)} chunks...")

                embeddings = await embedder.embed(text_chunks)
                await vector_store.add(embeddings, text_chunks)

            else:
                logger.warning("Vector store initialized empty")

        except Exception as e:
            raise CustomException(e, logger=logger)

        # =========================
        # 5. Pipeline Assembly
        # =========================
        try:
            rag_container["pipeline"] = RAGPipeline(
                embedder=embedder,
                vectorstore=vector_store,
                generator=generator,
                normalizer=normalizer,
                memory_manager=memory_manager
            )

            logger.info("RAG Engine Ready ")

        except Exception as e:
            raise CustomException(e, logger=logger)

        yield

    except CustomException:
        # Already logged → just propagate
        raise

    except Exception as e:
        # Catch any unknown failure
        raise CustomException(e, logger=logger)

    finally:
        logger.info("Shutting down cleanup...")
        rag_container["pipeline"] = None


app = FastAPI(
    title="Local RAG API",
    description="Async RAG Pipeline using Ollama + FAISS",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(chat_router, prefix="/chat", tags=["Chat"])
app.include_router(auth_router, prefix="/auth", tags=["Auth"])