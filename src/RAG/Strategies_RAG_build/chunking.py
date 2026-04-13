import sys
import json
import uuid
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# Third Party
from sqlalchemy.dialects.postgresql import insert
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Internal Imports
from src.RAG.models import ChunkModel
from src.db.main import get_session
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

# Local Import from embedder.py
from embedder import DocumentEmbedder

# Initialize Logger
logger = get_log("DocumentChunker")

class DocumentChunker:
    """
    Refactored Chunker: Splits text, computes confidence-based dynamic context, 
    and performs bulk upserts to PostgreSQL.
    """
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "", ". ", " "]
        )
        logger.info(f"Chunker Configured | Size: {chunk_size}, Overlap: {chunk_overlap}")

    def _compute_confidence(self, text: str, keywords: List[str]) -> float:
        """Heuristic scoring based on length, keyword density, and structure."""
        if not text: return 0.0
        text_len = len(text)
        
        # Length scoring logic
        if text_len < 40: length_score = 0.3
        elif text_len < 80: length_score = 0.6
        elif text_len <= 200: length_score = 1.0
        else: length_score = 0.8

        keyword_hits = sum(1 for kw in keywords if kw.lower() in text.lower())
        keyword_score = (keyword_hits / len(keywords)) if keywords else 0.5
        structure_score = 1.0 if text.strip().endswith((".", "!", "?")) else 0.7

        score = 0.5 * length_score + 0.3 * keyword_score + 0.2 * structure_score
        return round(min(score, 1.0), 3)

    def _get_dynamic_window(self, confidence: float) -> int:
        """Determines context expansion range based on chunk quality."""
        if confidence >= 0.85: return 1  # High quality: minimal expansion
        if confidence >= 0.6:  return 2  # Medium quality
        return 3                        # Low quality: needs more context

    @track_performance
    async def process_and_upload(self, data: List[Dict[str, Any]], embedder: DocumentEmbedder):
        try:
            if not data:
                logger.warning("No data found for ingestion.")
                return

            # 1. Split Text and Prepare Embeddings
            all_chunks: List[ChunkModel] = []
            raw_texts = []
            meta_map = []
            counter = 1

            for item in data:
                splits = self.splitter.split_text(item["text"])
                for idx, txt in enumerate(splits):
                    raw_texts.append(txt)
                    meta_map.append({"item": item, "idx_in_para": idx + 1})

            # 2. Batch Embedding
            embeddings = await embedder.embed_batch(raw_texts)

            # 3. Initial Chunk Object Creation
            for i, (txt, emb) in enumerate(zip(raw_texts, embeddings)):
                item = meta_map[i]["item"]
                keywords = [str(k).strip() for k in item.get("keywords", []) if k]
                
                chunk = ChunkModel(
                    chunk_id=counter,
                    chunk_text=txt,
                    embedding=emb,
                    confidence_score=self._compute_confidence(txt, keywords),
                    chunk_metadata={
                        "paragraph_id": item.get("paragraph_id"),
                        "section_id": item.get("section_id"),
                        "section_name": item.get("section_name"),
                        "topic_id": item.get("topic_id"),
                        "topic_name": item.get("topic_name"),
                        "keywords": keywords,
                        "index_in_para": meta_map[i]["idx_in_para"]
                    }
                )
                
                # Linking logic
                if all_chunks and all_chunks[-1].chunk_metadata["section_id"] == item.get("section_id"):
                    chunk.prev_chunk_id = all_chunks[-1].chunk_id
                    all_chunks[-1].next_chunk_id = chunk.chunk_id
                
                all_chunks.append(chunk)
                counter += 1

            # 4. Dynamic Context Expansion
            chunk_lookup = {c.chunk_id: c for c in all_chunks}
            for chunk in all_chunks:
                window = self._get_dynamic_window(chunk.confidence_score)
                context_ids = {chunk.chunk_id}

                # Expand Left/Right
                for direction in ["prev_chunk_id", "next_chunk_id"]:
                    curr = chunk
                    for _ in range(window):
                        cid = getattr(curr, direction)
                        if cid is None or cid not in chunk_lookup: break
                        context_ids.add(cid)
                        curr = chunk_lookup[cid]

                sorted_context = sorted([chunk_lookup[cid] for cid in context_ids], key=lambda x: x.chunk_id)
                chunk.context_chunks = [
                    {"chunk_id": c.chunk_id, "chunk_text": c.chunk_text} for c in sorted_context
                ]

            # 5. DB Upsert
            async for session in get_session():
                try:
                    logger.info(f"Syncing {len(all_chunks)} chunks to Postgres...")
                    for chunk in all_chunks:
                        # Convert SQLModel to dict for SQLAlchemy core insert
                        row = chunk.model_dump()
                        stmt = insert(ChunkModel).values(**row)
                        
                        upsert_stmt = stmt.on_conflict_do_update(
                            index_elements=['id'],
                            set_={
                                "chunk_text": stmt.excluded.chunk_text,
                                "embedding": stmt.excluded.embedding,
                                "confidence_score": stmt.excluded.confidence_score,
                                "chunk_metadata": stmt.excluded.chunk_metadata,
                                "prev_chunk_id": stmt.excluded.prev_chunk_id,
                                "next_chunk_id": stmt.excluded.next_chunk_id,
                                "context_chunks": stmt.excluded.context_chunks
                            }
                        )
                        await session.execute(upsert_stmt)
                    
                    await session.commit()
                    logger.info("Database Synchronization Successful.")
                    break 

                except Exception as db_err:
                    await session.rollback()
                    logger.error(f"Database Error: {db_err}")
                    raise db_err

        except Exception as e:
            logger.error("Chunking Pipeline failed.")
            raise CustomException(e, sys)

# --- Orchestrator ---

async def run_pipeline(json_path: str):
    try:
        embedder = DocumentEmbedder()
        chunker = DocumentChunker()
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        await chunker.process_and_upload(data, embedder)
        logger.info("Pipeline Complete.")
    except Exception as e:
        logger.critical(f"Crash: {e}")
        raise CustomException(e, sys)