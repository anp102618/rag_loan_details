import sys
import numpy as np
import asyncio
from typing import List, Dict, Any, Set
from rank_bm25 import BM25Okapi
from src.Utils.logger_setup import setup_logger, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.Strategies_RAG_build.embeddings import DocumentEmbedder
from src.RAG.Strategies_RAG_build.vector_store import FaissVectorStore

logger = setup_logger("HybridRetriever")

class HybridRetriever:
    def __init__(self, vector_store = FaissVectorStore(), embedder=DocumentEmbedder()):
        self.vs = vector_store
        self.embedder = embedder

    @track_performance
    async def retrieve_batch(
        self, 
        query_section_map: Dict[str, List[str]], 
        vector_k: int = 5, 
        bm25_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Processes multiple queries. 
        Flow: Classify by Section -> Deduplicate IDs -> Local Hybrid Search.
        """
        try:
            # 1. Sync FAISS with DB
            await self.vs.sync_from_db()

            # 2. Global Classification & Deduplication
            # Pre-group all chunks into their respective sections to avoid redundant filtering
            all_required_sections = set().union(*query_section_map.values())
            section_pools: Dict[str, List[Dict[str, Any]]] = {s: [] for s in all_required_sections}
            section_seen_ids: Dict[str, Set[int]] = {s: set() for s in all_required_sections}

            for db_id, meta in self.vs.db_id_to_metadata.items():
                section_name = meta.get("metadata", {}).get("section_name")
                if section_name in all_required_sections:
                    if db_id not in section_seen_ids[section_name]:
                        section_pools[section_name].append({
                            "id": db_id,
                            "text": meta.get("text", ""),
                            "metadata": meta.get("metadata", {}),
                            "confidence": meta.get("confidence", 0.0)
                        })
                        section_seen_ids[section_name].add(db_id)

            # 3. Process Each Query
            batch_results = {}
            for query, target_sections in query_section_map.items():
                # Combine pools for the specific sections requested for this query
                query_pool = []
                query_seen_ids = set()
                for s in target_sections:
                    for chunk in section_pools.get(s, []):
                        if chunk["id"] not in query_seen_ids:
                            query_pool.append(chunk)
                            query_seen_ids.add(chunk["id"])

                if not query_pool:
                    batch_results[query] = []
                    continue

                # --- Execute Hybrid Search on the Filtered Query Pool ---
                
                # A. BM25
                tokenized_corpus = [doc["text"].lower().split() for doc in query_pool]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)
                
                b_hits = []
                for idx in np.argsort(bm25_scores)[-bm25_k:][::-1]:
                    if bm25_scores[idx] > 0:
                        hit = query_pool[idx].copy()
                        hit.update({"retrieval_source": "bm25", "score": float(bm25_scores[idx])})
                        b_hits.append(hit)

                # B. Vector
                query_emb = await self.embedder.embed_query(query)
                v_hits_raw = await self.vs.search(query_emb, top_k=vector_k * 10)
                
                v_hits = []
                for v_hit in v_hits_raw:
                    if v_hit["id"] in query_seen_ids:
                        v_hit["retrieval_source"] = "vector"
                        v_hits.append(v_hit)
                    if len(v_hits) >= vector_k:
                        break

                # 4. Final Merge & Deduplication for this query
                final_results = []
                final_seen = set()
                for hit in v_hits + b_hits:
                    if hit["id"] not in final_seen:
                        final_results.append(hit)
                        final_seen.add(hit["id"])
                
                batch_results[query] = final_results

            return batch_results

        except Exception as e:
            logger.error(f"Batch retrieval failed: {str(e)}")
            raise CustomException(e, sys)