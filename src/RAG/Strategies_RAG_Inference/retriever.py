import sys
import asyncio
from typing import List, Dict, Any, Set, Optional
from rank_bm25 import BM25Okapi

# Internal Imports
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.Strategies_RAG_build.embeddings import DocumentEmbedder
from src.RAG.Strategies_RAG_build.vector_store import FaissVectorStore

# Initialize Logger
logger = get_log("HybridRetriever")

class HybridRetriever:
    """
    Orchestrates Hybrid Retrieval (Semantic + Keyword) with Metadata Filtering
    and Context Expansion, fully wrapped with logging and performance tracking.
    """
    def __init__(self, vector_store: Optional[FaissVectorStore] = None, 
                 embedder: Optional[DocumentEmbedder] = None):
        try:
            self.vs = vector_store or FaissVectorStore(dim=768)
            self.embedder = embedder or DocumentEmbedder()
            
            self._bm25_indices: Dict[str, BM25Okapi] = {}
            self._section_corpora: Dict[str, List[Any]] = {}
            
            logger.info("HybridRetriever initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    async def main_retrieval_flow(self, query_section_map: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        The primary entry point that runs the end-to-end retrieval logic.
        """
        try:
            if not query_section_map:
                logger.warning("Main flow received an empty query-section map.")
                return []

            # 1. Sync FAISS with DB state
            logger.info("Step 1: Synchronizing VectorStore with database...")
            await self.vs.sync_from_db()
            
            # 2. Build section-specific BM25 indices
            logger.info("Step 2: Building partitioned BM25 indices...")
            self._build_filtered_indices()

            # 3. Perform hybrid search and expansion
            logger.info(f"Step 3: Executing hybrid search for {len(query_section_map)} variations...")
            context_chunks = await self._get_hybrid_context(query_section_map)
            
            logger.info(f"Retrieval flow complete. Total unique context blocks: {len(context_chunks)}")
            return context_chunks

        except Exception as e:
            logger.error(f"Main retrieval flow failed: {str(e)}")
            raise CustomException(e, sys)

    @track_performance
    def _build_filtered_indices(self):
        """Organizes data into section-specific BM25 indices for precision."""
        try:
            all_chunks = list(self.vs.id_to_chunk.values())
            if not all_chunks:
                logger.warning("No chunks available in VectorStore. BM25 build skipped.")
                return

            temp_corpora = {}
            for chunk in all_chunks:
                # Accessing metadata from the ChunkModel object
                section = chunk.chunk_metadata.get("section_name", "General")
                if section not in temp_corpora:
                    temp_corpora[section] = []
                temp_corpora[section].append(chunk)

            self._section_corpora = temp_corpora

            for section, chunks in self._section_corpora.items():
                tokenized = [c.chunk_text.lower().split() for c in chunks]
                self._bm25_indices[section] = BM25Okapi(tokenized)
            
            logger.info(f"Built BM25 indices for sections: {list(self._bm25_indices.keys())}")
        except Exception as e:
            logger.error("Failed to build BM25 indices.")
            raise CustomException(e, sys)

    @track_performance
    async def _get_hybrid_context(
        self, 
        query_section_map: Dict[str, List[str]], 
        vector_k: int = 4, 
        bm25_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Internal logic to merge vector and keyword hits."""
        try:
            seen_ids: Set[int] = set()
            base_results = []
            chunk_id_map = {c.chunk_id: c for c in self.vs.id_to_chunk.values()}

            for query, sections in query_section_map.items():
                # Parallel Vector Search
                v_hits = await self._vector_search_filtered(query, sections, vector_k)
                # Sync BM25 Search
                b_hits = self._bm25_search_filtered(query, sections, bm25_k)

                for hit in (v_hits + b_hits):
                    if hit["chunk_id"] not in seen_ids:
                        base_results.append(hit)
                        seen_ids.add(hit["chunk_id"])

            return self._finalize_expansion(base_results, chunk_id_map)
        except Exception as e:
            logger.error("Hybrid context merging failed.")
            raise CustomException(e, sys)

    async def _vector_search_filtered(self, query: str, sections: List[str], top_k: int):
        """Semantic search with section-based metadata filtering."""
        try:
            query_emb = await self.embedder.embed_query(query)
            # Expand=False as we handle global expansion later
            raw_hits = await self.vs.search(query_emb, top_k=top_k * 4, expand=False)
            
            return [h for h in raw_hits if h.get("metadata", {}).get("section_name") in sections][:top_k]
        except Exception as e:
            logger.warning(f"Vector search failed for '{query[:30]}...': {e}")
            return []

    def _bm25_search_filtered(self, query: str, sections: List[str], top_k: int):
        """Keyword search within the specific indices for identified sections."""
        try:
            all_hits = []
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
                    all_hits.append({
                        "chunk_id": chunk.chunk_id, 
                        "chunk_text": chunk.chunk_text,
                        "metadata": chunk.chunk_metadata, 
                        "search_type": "bm25",
                        "context_ids": [c['chunk_id'] for c in chunk.context_chunks] if chunk.context_chunks else []
                    })
            return all_hits
        except Exception as e:
            logger.warning(f"BM25 search failed for '{query[:30]}...': {e}")
            return []

    def _finalize_expansion(self, hits: List[Dict[str, Any]], chunk_map: Dict[int, Any]):
        """Deduplicates hits and injects neighboring context for better LLM reasoning."""
        try:
            final_list, final_seen = [], set()
            for hit in hits:
                # 1. Primary Result
                if hit["chunk_id"] not in final_seen:
                    final_list.append({
                        "chunk_id": hit["chunk_id"], 
                        "chunk_text": hit["chunk_text"],
                        "metadata": hit.get("metadata", {}), 
                        "source": hit.get("search_type", "vector")
                    })
                    final_seen.add(hit["chunk_id"])

                # 2. Neighbors (Context)
                for cid in hit.get("context_ids", []):
                    if cid not in final_seen:
                        neighbor = chunk_map.get(cid)
                        if neighbor:
                            final_list.append({
                                "chunk_id": neighbor.chunk_id, 
                                "chunk_text": neighbor.chunk_text,
                                "metadata": neighbor.chunk_metadata, 
                                "source": "context_expansion"
                            })
                            final_seen.add(cid)
            return final_list
        except Exception as e:
            logger.error("Context expansion finalization failed.")
            raise CustomException(e, sys)