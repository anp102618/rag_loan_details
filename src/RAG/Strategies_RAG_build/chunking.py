import sys
import json
import nest_asyncio
from typing import List, Dict, Any, Optional

# Third Party
from sqlalchemy.dialects.postgresql import insert
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.RAG.Strategies_RAG_build.embeddings import DocumentEmbedder

# Internal Imports
from src.RAG.models import ChunkModel
from src.db.main import get_session
from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("chunking")
current_logger.set(logger)
nest_asyncio.apply()

class DocumentChunker:
    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "•", ". ", " "]
        )
        logger.info(f"[INIT] Chunker Configured | Size: {chunk_size}, Overlap: {chunk_overlap}")

    def _compute_confidence(self, text: str, keywords: List[str]) -> float:
        if not text: return 0.0
        text_len = len(text)
        length_score = 1.0 if 80 <= text_len <= 200 else 0.6
        keyword_hits = sum(1 for kw in keywords if kw and str(kw).lower() in text.lower())
        keyword_score = (keyword_hits / len(keywords)) if keywords else 0.5
        structure_score = 1.0 if text.strip().endswith((".", "!", "?")) else 0.7
        return round(float(0.5 * length_score + 0.3 * keyword_score + 0.2 * structure_score), 3)

    def _get_dynamic_window(self, confidence: float) -> int:
        if confidence >= 0.85: return 1
        if confidence >= 0.6:  return 2
        return 3

    @track_performance
    async def process_and_upload(self, data: List[Dict[str, Any]], embedder: Any):
        try:
            if not data:
                logger.warning("[STEP 0] Ingestion skipped: Input data is empty.")
                return

            # --- STEP 1: TEXT SPLITTING ---
            logger.info(f"[STEP 1] Starting text splitting for {len(data)} source items...")
            raw_texts, meta_map = [], []
            for item_idx, item in enumerate(data):
                splits = self.splitter.split_text(item["text"])
                for idx, txt in enumerate(splits):
                    raw_texts.append(txt)
                    meta_map.append({"item": item, "idx_in_para": idx + 1})
            logger.info(f"[STEP 1] Completed. Created {len(raw_texts)} total chunks.")

            # --- STEP 2: EMBEDDING ---
            logger.info(f"[STEP 2] Sending {len(raw_texts)} chunks to the embedding model...")
            embeddings = await embedder.embed_batch(raw_texts)
            logger.info(f"[STEP 2] Received {len(embeddings)} embedding vectors successfully.")

            # --- STEP 3: OBJECT CREATION & LINKING ---
            logger.info("[STEP 3] Initializing ChunkModel objects and sequence linking...")
            all_chunks: List[ChunkModel] = []
            for i, (txt, emb) in enumerate(zip(raw_texts, embeddings)):
                item = meta_map[i]["item"]
                chunk = ChunkModel(
                    id=i + 1,  # Sequential Integer PK
                    chunk_text=txt,
                    embedding=emb,
                    confidence_score=self._compute_confidence(txt, item.get("keywords", [])),
                    chunk_metadata={**item, "index_in_para": meta_map[i]["idx_in_para"]}
                )
                
                # Sequential Linking (Bi-directional)
                if all_chunks and all_chunks[-1].chunk_metadata.get("section_id") == item.get("section_id"):
                    chunk.prev_chunk_id = all_chunks[-1].id
                    all_chunks[-1].next_chunk_id = chunk.id
                
                all_chunks.append(chunk)
            logger.info(f"[STEP 3] Created {len(all_chunks)} objects with integer IDs 1 to {len(all_chunks)}.")

            # --- STEP 4: DYNAMIC CONTEXT EXPANSION ---
            logger.info("[STEP 4] Computing dynamic context windows based on confidence scores...")
            chunk_lookup = {c.id: c for c in all_chunks}
            for chunk in all_chunks:
                window = self._get_dynamic_window(chunk.confidence_score)
                context_ids = {chunk.id}

                for attr in ["prev_chunk_id", "next_chunk_id"]:
                    curr = chunk
                    for _ in range(window):
                        cid = getattr(curr, attr, None)
                        if cid and cid in chunk_lookup:
                            context_ids.add(cid)
                            curr = chunk_lookup[cid]
                        else: break

                sorted_context = sorted([chunk_lookup[cid] for cid in context_ids], key=lambda x: x.id)
                chunk.context_chunks = [{"chunk_id": c.id, "chunk_text": c.chunk_text} for c in sorted_context]
            logger.info("[STEP 4] Context expansion complete for all chunks.")

            # --- STEP 5: DATABASE SYNCHRONIZATION ---
            logger.info("[STEP 5] Preparing bulk upsert for PostgreSQL...")
            async for session in get_session():
                try:
                    # Convert models to dicts for insert statement
                    rows = [c.model_dump(exclude={"created_at"}) for c in all_chunks]
                    
                    stmt = insert(ChunkModel).values(rows)
                    # Dynamically set columns to update on conflict
                    update_cols = {k: getattr(stmt.excluded, k) for k in rows[0].keys() if k != 'id'}
                    
                    upsert_stmt = stmt.on_conflict_do_update(
                        index_elements=['id'], 
                        set_=update_cols
                    )
                    
                    logger.debug(f"[STEP 5] Executing upsert for {len(rows)} rows...")
                    await session.execute(upsert_stmt)
                    await session.commit()
                    logger.info(f"[STEP 5] Database Sync Successful. {len(all_chunks)} chunks upserted.")
                    break 
                except Exception as db_err:
                    await session.rollback()
                    logger.error(f"[STEP 5] Database Sync Failed: {str(db_err)}")
                    raise db_err

        except Exception as e:
            logger.exception("[CRITICAL] Pipeline failed during processing.")
            raise CustomException(e, sys)

# --- Orchestrator ---

async def chunking_pipeline(json_path: str):
    logger.info(f"--- Starting Pipeline for: {json_path} ---")
    try:
        
        embedder = DocumentEmbedder()
        chunker = DocumentChunker()
        
        logger.info("[PIPE] Loading JSON source data...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        await chunker.process_and_upload(data, embedder)
        logger.info("--- Pipeline Completed Successfully ---")
        
    except Exception as e:
        logger.critical(f"[PIPE] Fatal Crash: {str(e)}")
        raise CustomException(e, sys)