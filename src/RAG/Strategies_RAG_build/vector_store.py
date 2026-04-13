import faiss
import numpy as np
import asyncio
import pickle
from typing import List, Dict, Any, Optional
from sqlalchemy import select

from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.models import ChunkModel
from src.db.main import get_session

class FaissVectorStore:
    def __init__(
        self,
        dim: int = 768,
        index_type: str = "hnsw",
        M: int = 32,
        ef_construction: int = 200,
        nlist: int = 100
    ) -> None:
        self.logger = get_log("FaissVectorStore")
        self.dim = dim
        self.index_type = index_type
        self._lock = asyncio.Lock()

        # 🔹 Core storage
        self.index = None
        self.id_to_chunk: Dict[int, ChunkModel] = {}

        try:
            if index_type == "hnsw":
                self.index = faiss.IndexHNSWFlat(dim, M)
                self.index.hnsw.efConstruction = ef_construction
            elif index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dim)
                self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
            else:
                raise ValueError("index_type must be 'hnsw' or 'ivf'")

            self.logger.info(f"[INIT] FAISS | type={index_type}, dim={dim}")
        except Exception as e:
            raise CustomException(e, logger=self.logger)

    # ----------------------------------
    # 🔹 DB SYNC (New Addition)
    # ----------------------------------
    @track_performance
    async def sync_from_db(self):
        """Rebuilds the FAISS index by pulling all chunks from Postgres."""
        self.logger.info("Synchronizing FAISS index with PostgreSQL...")
        try:
            async for session in get_session():
                stmt = select(ChunkModel).order_by(ChunkModel.chunk_id.asc())
                result = await session.execute(stmt)
                chunks = result.scalars().all()
                
                if chunks:
                    # Clear current state before rebuild
                    async with self._lock:
                        self.index.reset()
                        self.id_to_chunk.clear()
                        await self.add(chunks)
                break
            self.logger.info(f"Sync Complete. Total vectors: {self.index.ntotal}")
        except Exception as e:
            self.logger.error(f"Sync failed: {e}")
            raise CustomException(e, logger=self.logger)

    # ----------------------------------
    # 🔹 ADD (Ingestion)
    # ----------------------------------
    @track_performance
    async def add(self, chunks: List[ChunkModel]) -> None:
        if not chunks:
            self.logger.warning("[ADD] No chunks")
            return

        try:
            embeddings = np.array(
                [np.array(c.embedding, dtype="float32") for c in chunks]
            )

            async with self._lock:
                if self.index_type == "ivf" and not self.index.is_trained:
                    self.logger.info("[ADD] Training IVF")
                    await asyncio.to_thread(self.index.train, embeddings)

                start_pos = len(self.id_to_chunk)
                await asyncio.to_thread(self.index.add, embeddings)

                # Map the FAISS internal index (0, 1, 2...) to the ChunkModel
                for i, chunk in enumerate(chunks):
                    self.id_to_chunk[start_pos + i] = chunk

            self.logger.info(f"[ADD] Done | total={self.index.ntotal}")
        except Exception as e:
            self.logger.error("[ADD] Failed")
            raise CustomException(e, logger=self.logger)

    # ----------------------------------
    # 🔹 SEARCH
    # ----------------------------------
    @track_performance
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        expand: bool = True
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        try:
            q = np.array([query_embedding]).astype("float32")
            distances, indices = await asyncio.to_thread(self.index.search, q, top_k)

            base_results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx == -1: continue
                
                chunk = self.id_to_chunk.get(idx)
                if not chunk: continue

                base_results.append({
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.chunk_text,
                    "metadata": chunk.chunk_metadata,
                    "confidence": chunk.confidence_score,
                    # We use the pre-computed context IDs from your refactored chunker
                    "context_ids": [c['chunk_id'] for c in chunk.context_chunks] if chunk.context_chunks else [],
                    "distance": float(dist)
                })

            if not expand:
                return base_results

            return self._expand_context(base_results)

        except Exception as e:
            self.logger.error(f"[SEARCH] Failed: {e}")
            raise CustomException(e, logger=self.logger)

    # ----------------------------------
    # 🔹 OPTIMIZED CONTEXT EXPANSION
    # ----------------------------------
    def _expand_context(self, base_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_chunk_ids = set()
        expanded = []
        
        # Create a fast lookup for chunk_id -> chunk mapping
        # Since id_to_chunk keys are FAISS indices, we map by the property
        chunk_id_map = {c.chunk_id: c for c in self.id_to_chunk.values()}

        for res in base_results:
            # Add the base chunk first if not seen
            if res["chunk_id"] not in seen_chunk_ids:
                expanded.append(res)
                seen_chunk_ids.add(res["chunk_id"])

            # Add neighbors from context_chunks
            for cid in res.get("context_ids", []):
                if cid not in seen_chunk_ids:
                    neighbor = chunk_id_map.get(cid)
                    if neighbor:
                        expanded.append({
                            "chunk_id": neighbor.chunk_id,
                            "chunk_text": neighbor.chunk_text,
                            "metadata": neighbor.chunk_metadata,
                            "confidence": neighbor.confidence_score,
                            "is_context": True # Flag to distinguish from primary hit
                        })
                        seen_chunk_ids.add(cid)

        return expanded

    # ----------------------------------
    # 🔹 PERSISTENCE
    # ----------------------------------
    async def save(self, index_path: str, metadata_path: str):
        try:
            await asyncio.to_thread(faiss.write_index, self.index, index_path)
            with open(metadata_path, "wb") as f:
                pickle.dump(self.id_to_chunk, f)
            self.logger.info("[SAVE] Index and Metadata saved successfully.")
        except Exception as e:
            raise CustomException(e, logger=self.logger)

    async def load(self, index_path: str, metadata_path: str):
        try:
            self.index = await asyncio.to_thread(faiss.read_index, index_path)
            with open(metadata_path, "rb") as f:
                self.id_to_chunk = pickle.load(f)
            self.logger.info(f"[LOAD] Index loaded with {self.index.ntotal} vectors.")
        except Exception as e:
            raise CustomException(e, logger=self.logger)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dim,
            "index_type": self.index_type
        }