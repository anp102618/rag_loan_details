import sys
import uuid
import asyncio
from typing import List, Dict, Any, Set, Optional
from rank_bm25 import BM25Okapi

# Internal Imports
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.Strategies_RAG_build.embeddings import DocumentEmbedder
from src.RAG.Strategies_RAG_build.vector_store import FaissVectorStore

logger = get_log("HybridRetriever")

class HybridRetriever:
    """
    Retriever that maintains the integrity of ChunkModel objects.
    Each returned 'chunk' is a ChunkModel instance with an additional 
    dynamic 'retrieval_source' attribute.
    """
    def __init__(self, vector_store: Optional[FaissVectorStore] = None, 
                 embedder: Optional[DocumentEmbedder] = None):
        try:
            self.vs = vector_store or FaissVectorStore(dim=768)
            self.embedder = embedder or DocumentEmbedder()
            self._bm25_indices: Dict[str, BM25Okapi] = {}
            self._section_corpora: Dict[str, List[Any]] = {} # List[ChunkModel]
            logger.info("HybridRetriever initialized (Object-Preservation Mode).")
        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    async def main_retrieval_flow(self, query_section_map: Dict[str, List[str]]) -> List[Any]:
        """
        End-to-end flow returning a list of ChunkModel objects.
        """
        try:
            if not query_section_map:
                return []

            # 1. Prepare data state
            await self.vs.sync_from_db()
            self._build_filtered_indices()

            # 2. Execute retrieval
            return await self._get_hybrid_context(query_section_map)

        except Exception as e:
            logger.error(f"Retrieval flow failed: {str(e)}")
            raise CustomException(e, sys)

    def _build_filtered_indices(self):
        """Organizes ChunkModel instances into section-based BM25 indices."""
        try:
            all_chunks = list(self.vs.id_to_chunk.values())
            temp_corpora = {}
            for chunk in all_chunks:
                section = chunk.chunk_metadata.get("section_name", "General")
                temp_corpora.setdefault(section, []).append(chunk)

            self._section_corpora = temp_corpora
            for section, chunks in self._section_corpora.items():
                tokenized = [c.chunk_text.lower().split() for c in chunks]
                self._bm25_indices[section] = BM25Okapi(tokenized)
        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    async def _get_hybrid_context(
        self, 
        query_section_map: Dict[str, List[str]], 
        vector_k: int = 4, 
        bm25_k: int = 3
    ) -> List[Any]:
        """
        Filters by section first, then returns ChunkModel objects with source IDs.
        """
        try:
            final_chunks = []
            seen_ids: Set[uuid.UUID] = set()

            for query, sections in query_section_map.items():
                # --- STEP 1: PRE-FILTER ---
                allowed_ids = [
                    chunk.id for chunk in self.vs.id_to_chunk.values()
                    if chunk.chunk_metadata.get("section_name") in sections
                ]
                if not allowed_ids: continue

                # --- STEP 2: DUAL SEARCH ---
                v_hits = await self._vector_search_filtered(query, allowed_ids, vector_k)
                b_hits = self._bm25_search_filtered(query, sections, bm25_k)

                # --- STEP 3: CONTEXT INJECTION ---
                for hit in (v_hits + b_hits):
                    # 'hit' is the full ChunkModel object
                    source_label = getattr(hit, "retrieval_source", "unknown")
                    
                    # A. Add Primary Hit
                    if hit.id not in seen_ids:
                        final_chunks.append(hit)
                        seen_ids.add(hit.id)

                    # B. Add Neighbors from context_chunks field
                    if hit.context_chunks:
                        for ctx_data in hit.context_chunks:
                            # Context data is usually a dict in the DB; we fetch the actual model
                            ctx_id = uuid.UUID(str(ctx_data["id"]))
                            if ctx_id not in seen_ids:
                                neighbor_obj = self.vs.id_to_chunk.get(ctx_id)
                                if neighbor_obj:
                                    # Tag the neighbor with context provenance
                                    setattr(neighbor_obj, "retrieval_source", f"{source_label}_context")
                                    final_chunks.append(neighbor_obj)
                                    seen_ids.add(ctx_id)

            return final_chunks
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise CustomException(e, sys)

    async def _vector_search_filtered(self, query: str, allowed_ids: List[uuid.UUID], top_k: int) -> List[Any]:
        """Returns ChunkModel objects from vector search."""
        try:
            query_emb = await self.embedder.embed_query(query)
            # The vector store search should return ChunkModel instances
            results = await self.vs.search(query_emb, top_k=top_k, filter_ids=allowed_ids)
            for chunk in results:
                setattr(chunk, "retrieval_source", "vector")
            return results
        except Exception as e:
            logger.warning(f"Vector search error: {e}")
            return []

    def _bm25_search_filtered(self, query: str, sections: List[str], top_k: int) -> List[Any]:
        """Returns ChunkModel objects from BM25 search."""
        hits = []
        tokenized_query = query.lower().split()
        for section in sections:
            index = self._bm25_indices.get(section)
            corpus = self._section_corpora.get(section)
            if not index or not corpus: continue

            scores = index.get_scores(tokenized_query)
            top_n = scores.argsort()[-top_k:][::-1]
            for idx in top_n:
                if scores[idx] <= 0: continue
                chunk = corpus[idx]
                setattr(chunk, "retrieval_source", "bm25")
                hits.append(chunk)
        return hits