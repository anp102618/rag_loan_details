import faiss
import numpy as np
import asyncio
from typing import List, Dict, Any
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException


class FaissVectorStore:
    """
    Handles local vector storage and similarity search using the FAISS library.
    """

    def __init__(self, dim: int) -> None:
        """
        Initializes the FAISS index and internal text storage.

        Args:
            dim (int): The dimensionality of the embeddings (e.g., 768 for nomic-embed-text).
        """
        self.logger = get_log("VectorStore")
        self.dim = dim
        
        # IndexFlatL2 measures Euclidean distance (smaller is better)
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []
        self._lock = asyncio.Lock()

        self.logger.info(f"[INIT] FaissVectorStore initialized | dimension={dim}")

    @track_performance
    async def add(self, embeddings: List[List[float]], texts: List[str]) -> None:
        """
        Adds vectors and their corresponding text chunks to the store.

        Args:
            embeddings (List[List[float]]): The vector representations.
            texts (List[str]): The raw text chunks associated with the vectors.
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have the same length.")

        self.logger.info(f"[FAISS-ADD] Attempting to add count={len(texts)}")

        async with self._lock:
            try:
                # Convert to numpy float32 as required by FAISS
                vecs = np.array(embeddings).astype("float32")
                
                # Execute blocking FAISS operation in a thread
                await asyncio.to_thread(self.index.add, vecs)
                self.texts.extend(texts)
                
                self.logger.info(f"[FAISS-ADD] Success | Total index size={len(self.texts)}")
            except Exception as e:
                self.logger.error("[FAISS-ADD] Failed to update index")
                raise CustomException(e, logger=self.logger)

    @track_performance
    async def search(self, query_embedding: List[float], k: int = 5) -> List[str]:
        """
        Searches the index for the most similar text chunks.

        Args:
            query_embedding (List[float]): The vectorized user query.
            k (int): Number of top results to retrieve.

        Returns:
            List[str]: The top k matching text chunks.
        """
        self.logger.info(f"[FAISS-SEARCH] Searching for k={k} nearest neighbors")

        if not self.texts:
            self.logger.warning("[FAISS-SEARCH] Search called on empty index")
            return []

        try:
            # Prepare query vector
            q = np.array([query_embedding]).astype("float32")

            # Perform search
            distances, indices = await asyncio.to_thread(self.index.search, q, k)

            results: List[str] = []
            for idx in indices[0]:
                if idx != -1 and idx < len(self.texts):
                    results.append(self.texts[idx])

            self.logger.info(f"[FAISS-SEARCH] Completed | found={len(results)} matches")
            return results

        except Exception as e:
            self.logger.error("[FAISS-SEARCH] Search operation failed")
            raise CustomException(e, logger=self.logger)

    def get_stats(self) -> Dict[str, Any]:
        """Returns basic statistics about the current index."""
        return {
            "total_elements": self.index.ntotal,
            "dimension": self.dim,
            "is_trained": self.index.is_trained
        }