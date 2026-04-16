import faiss
import numpy as np
import asyncio
import pickle
import os
from typing import List, Union, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select

# Project-specific imports
from src.RAG.models import ChunkModel
from src.db.main import get_session
from src.Utils.logger_setup import setup_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("VectorStore")

class VectorStore:
    def __init__(self, dimension: int = 768, index_type: str = "IVF_HNSW", nlist: int = 100):
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Internal state
        self.metadata: Dict[int, Dict[str, Any]] = {}
        self._last_synced_id = 0
        
        try:
            self.index = self._initialize_index()
            logger.info(f"VectorStore initialized with {index_type} (Dimension: {dimension})")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            raise CustomException("Index initialization failed", status_code=500)

    def _initialize_index(self):
        """Builds the internal FAISS index structure."""
        if self.index_type == "HNSW":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        
        elif self.index_type == "IVF_HNSW":
            # HNSW quantizer + IVF partitions
            quantizer = faiss.IndexHNSWFlat(self.dimension, 32)
            return faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            
        elif self.index_type == "Flat":
            return faiss.IndexFlatL2(self.dimension)
            
        raise CustomException(f"Unsupported index type: {self.index_type}", status_code=400)

    def _format_vectors(self, vectors: Union[List, np.ndarray]) -> np.ndarray:
        """Ensures vectors are float32 and 2D."""
        v = np.array(vectors).astype('float32')
        return v if v.ndim > 1 else v.reshape(1, -1)

    @track_performance
    async def add(self, chunks: List[ChunkModel]) -> int:
        """Asynchronous wrapper for adding chunks."""
        if not chunks:
            return 0
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._process_add, chunks)

    def _process_add(self, chunks: List[ChunkModel]) -> int:
        """Core logic to process embeddings and update metadata."""
        try:
            embeddings = [c.embedding for c in chunks if c.embedding is not None]
            if not embeddings:
                logger.warning("No embeddings found in the provided chunks.")
                return 0
                
            vectors = self._format_vectors(embeddings)
            
            # Handle IVF Training
            if "IVF" in self.index_type and not self.index.is_trained:
                if len(vectors) < self.nlist:
                    logger.warning(f"Training skipped: Need {self.nlist} vectors, got {len(vectors)}.")
                    return 0
                logger.info(f"Training IVF index with {len(vectors)} vectors...")
                self.index.train(vectors)
            
            start_id = self.index.ntotal
            self.index.add(vectors)
            
            # Map FAISS ID to the exact ChunkModel schema
            for i, chunk in enumerate(chunks):
                faiss_id = start_id + i
                self.metadata[faiss_id] = {
                    "id": chunk.id,
                    "chunk_text": chunk.chunk_text,
                    "chunk_metadata": chunk.chunk_metadata,
                    "confidence_score": chunk.confidence_score,
                    "prev_chunk_id": chunk.prev_chunk_id,
                    "next_chunk_id": chunk.next_chunk_id,
                    "context_chunks": chunk.context_chunks
                }
            
            logger.info(f"Successfully added {len(embeddings)} vectors. Index total: {self.index.ntotal}")
            return len(embeddings)
            
        except Exception as e:
            logger.error(f"Error during vector addition: {str(e)}")
            return 0

    async def sync_from_db(self, batch_size: int = 100):
        """Syncs the vector store with the PostgreSQL DB incrementally."""
        logger.info(f"Starting sync from DB. Last synced ID: {self._last_synced_id}")
        total_synced = 0

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
                    logger.info("Database sync complete. No new chunks found.")
                    break

                count = await self.add(chunks)
                if count > 0:
                    self._last_synced_id = chunks[-1].id
                    total_synced += count
                    logger.info(f"Synced batch of {count}. New last_id: {self._last_synced_id}")
                else:
                    # If addition failed (e.g., training requirement not met), stop sync
                    break
            break 
            
        return total_synced

    async def search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Asynchronous search interface."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._process_search, query_vector, k)

    def _process_search(self, query_vector: List[float], k: int) -> List[Dict[str, Any]]:
        """Performs search and returns dictionary items matching ChunkModel."""
        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty index.")
            return []

        try:
            vector = self._format_vectors(query_vector)
            distances, indices = self.index.search(vector, k)
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx in self.metadata:
                    chunk_data = self.metadata[idx].copy()
                    chunk_data['search_distance'] = float(dist)
                    results.append(chunk_data)
            
            logger.info(f"Search completed. Found {len(results)} matches.")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []

    def save(self, folder: str = "vector_store"):
        """Saves both the FAISS index and the metadata dictionary."""
        try:
            os.makedirs(folder, exist_ok=True)
            faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
            with open(os.path.join(folder, "metadata.pkl"), "wb") as f:
                pickle.dump({"metadata": self.metadata, "last_id": self._last_synced_id}, f)
            logger.info(f"VectorStore successfully saved to {folder}")
        except Exception as e:
            logger.error(f"Failed to save VectorStore: {str(e)}")

    def load(self, folder: str = "vector_store") -> bool:
        """Loads the store from disk."""
        if not os.path.exists(folder):
            logger.warning(f"Load failed: Folder {folder} does not exist.")
            return False
        try:
            self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
            with open(os.path.join(folder, "metadata.pkl"), "rb") as f:
                data = pickle.load(f)
                self.metadata = data["metadata"]
                self._last_synced_id = data["last_id"]
            logger.info(f"VectorStore loaded from {folder}. Total vectors: {self.index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"Error loading VectorStore: {str(e)}")
            return False