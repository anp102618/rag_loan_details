import faiss
import numpy as np
import asyncio
import pickle
import sys
from typing import List, Dict, Any, Optional, Union

from sqlalchemy import select
from src.RAG.models import ChunkModel
from src.db.main import get_session
from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

class FaissVectorStore:
    def __init__(
        self,
        dim: int = 768,
        M: int = 32,
        ef_construction: int = 200,
    ) -> None:
        """
        Initializes the FAISS HNSW index and local ID mappings.
        """
        try:
            self.logger = setup_logger("FaissVectorStore")
            current_logger.set(self.logger)

            self.dim = dim
            self._lock = asyncio.Lock()

            self.logger.info(f"[INIT] Initializing FAISS HNSW | dim={dim}, M={M}")

            # 🔹 FAISS HNSW Index (Efficient for large-scale similarity search)
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = 50

            # 🔹 Light-weight mappings to link FAISS results back to Postgres data
            self.faiss_id_to_db_id: Dict[int, int] = {}
            self.db_id_to_metadata: Dict[int, Dict[str, Any]] = {}

            self._last_synced_id = 0

            self.logger.info("[INIT COMPLETE] FAISS ready")

        except Exception as e:
            # Removed 'sys' argument to match your CustomException signature
            raise CustomException(e)

    # =========================================================
    #  SYNC FROM DB (INCREMENTAL)
    # =========================================================
    @track_performance
    async def sync_from_db(self, batch_size: int = 100):
        self.logger.info("[SYNC] Starting incremental DB sync")
        total_added = 0

        try:
            async for session in get_session():
                while True:
                    stmt = (
                        select(ChunkModel)
                        .where(ChunkModel.id > self._last_synced_id)
                        .order_by(ChunkModel.id.asc())
                        .limit(batch_size)
                    )

                    result = await session.execute(stmt)
                    chunks = result.scalars().all()

                    if not chunks:
                        self.logger.info("[SYNC] No new chunks found")
                        break

                    self.logger.info(f"[SYNC] Processing batch: size={len(chunks)}")
                    await self.add(chunks)

                    self._last_synced_id = chunks[-1].id
                    total_added += len(chunks)

                break # Exit the session generator

            self.logger.info(f"[SYNC COMPLETE] Total synced: {total_added} | Index Total: {self.index.ntotal}")

        except Exception as e:
            self.logger.error(f"[SYNC ERROR] {str(e)}")
            raise CustomException(e)

    # =========================================================
    #  ADD CHUNKS (TYPE-SAFE)
    # =========================================================
    @track_performance
    async def add(self, chunks: List[ChunkModel]):
        if not chunks:
            return

        try:
            # 1. Prepare valid embeddings (Ensure they aren't None)
            valid_chunks = [c for c in chunks if c.embedding is not None]
            if not valid_chunks:
                self.logger.warning("[ADD] Batch contained no valid embeddings")
                return

            embeddings = np.array(
                [c.embedding for c in valid_chunks],
                dtype="float32"
            )

            async with self._lock:
                start_idx = self.index.ntotal
                # FAISS addition is CPU bound; offload to a thread to keep event loop free
                await asyncio.to_thread(self.index.add, embeddings)

                for i, chunk in enumerate(valid_chunks):
                    fid = start_idx + i
                    
                    # Robust ID extraction (Handles objects and ensures 'id' exists)
                    db_id = getattr(chunk, 'id', None)
                    if db_id is None:
                        continue

                    self.faiss_id_to_db_id[fid] = db_id

                    # Safe Context Parsing (Handles dicts or objects in context_chunks)
                    context_ids = []
                    raw_context = getattr(chunk, 'context_chunks', [])
                    if isinstance(raw_context, list):
                        for c in raw_context:
                            cid = c.get("id") if isinstance(c, dict) else getattr(c, 'id', None)
                            if cid:
                                context_ids.append(cid)

                    # Populate Metadata mapping
                    self.db_id_to_metadata[db_id] = {
                        "text": getattr(chunk, 'chunk_text', ""),
                        "metadata": getattr(chunk, 'chunk_metadata', {}),
                        "confidence": getattr(chunk, 'confidence_score', 0.0),
                        "context_ids": context_ids
                    }

        except Exception as e:
            self.logger.error(f"[ADD ERROR] {str(e)}")
            raise CustomException(e)

    # =========================================================
    #  SEARCH
    # =========================================================
    @track_performance
    async def search(
        self,
        query_embedding: Union[List[float], np.ndarray],
        top_k: int = 5,
        expand_context: bool = True
    ):
        try:
            if self.index.ntotal == 0:
                return []

            q = np.array(query_embedding, dtype="float32").reshape(1, -1)
            distances, indices = await asyncio.to_thread(self.index.search, q, top_k)

            results = []
            for fid, dist in zip(indices[0], distances[0]):
                if fid == -1: continue

                db_id = self.faiss_id_to_db_id.get(fid)
                meta = self.db_id_to_metadata.get(db_id)

                if meta:
                    results.append({
                        "id": db_id,
                        "chunk_text": meta["text"],
                        "metadata": meta["metadata"],
                        "confidence": meta["confidence"],
                        "context_ids": meta["context_ids"],
                        "distance": float(dist),
                        "is_context": False
                    })

            return self._expand_context(results) if expand_context else results

        except Exception as e:
            raise CustomException(e)

    def _expand_context(self, results):
        expanded = []
        seen = {r["id"] for r in results}

        for r in results:
            expanded.append(r)
            # Retrieve neighbor chunks for context (limit to first 3)
            for cid in r.get("context_ids", [])[:3]:
                if cid in seen: continue

                meta = self.db_id_to_metadata.get(cid)
                if meta:
                    expanded.append({
                        "id": cid,
                        "chunk_text": meta["text"],
                        "metadata": meta["metadata"],
                        "confidence": meta["confidence"],
                        "is_context": True,
                        "distance": None
                    })
                    seen.add(cid)
        return expanded

    # =========================================================
    #  PERSISTENCE
    # =========================================================
    async def save(self, path: str):
        try:
            self.logger.info(f"[SAVE] Persisting index to {path}")
            await asyncio.to_thread(faiss.write_index, self.index, f"{path}.index")

            payload = {
                "faiss_to_db": self.faiss_id_to_db_id,
                "metadata": self.db_id_to_metadata,
                "last_id": self._last_synced_id
            }

            with open(f"{path}.meta", "wb") as f:
                pickle.dump(payload, f)
            self.logger.info("[SAVE COMPLETE]")
        except Exception as e:
            raise CustomException(e)

    async def load(self, path: str):
        try:
            self.logger.info(f"[LOAD] Loading index from {path}")
            self.index = await asyncio.to_thread(faiss.read_index, f"{path}.index")

            with open(f"{path}.meta", "rb") as f:
                payload = pickle.load(f)

            self.faiss_id_to_db_id = payload["faiss_to_db"]
            self.db_id_to_metadata = payload["metadata"]
            self._last_synced_id = payload["last_id"]

            self.logger.info(f"[LOAD COMPLETE] Total vectors: {self.index.ntotal}")
        except Exception as e:
            raise CustomException(e)