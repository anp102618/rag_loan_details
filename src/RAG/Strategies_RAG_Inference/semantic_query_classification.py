import sys
import json
import asyncio
import nest_asyncio
import ollama  # Raw Ollama library
import numpy as np
from typing import List, Dict, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity
from src.RAG.Strategies_RAG_Inference.metadata_extractor import MetadataExtractor

from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("query_classifier")
current_logger.set(logger)
nest_asyncio.apply()

class QueryClassifier:
    """
    Classifies expanded queries into specific banking sections using 
    Nomic Embeddings and Qwen 2.5 via the raw Ollama API.
    """
    def __init__(self, embed_model: str = "nomic-embed-text", llm_model: str = "phi3.5"):
        try:
            self.embed_model = embed_model
            self.llm_model = llm_model
            self.metadata_tool = MetadataExtractor()
            
            self.section_list: List[str] = []
            self.section_embeddings: Optional[np.ndarray] = None
            
            logger.info(f"Classifier initialized | Embedding Model: {embed_model} | LLM: {llm_model}")
        except Exception as e:
            logger.error("Failed to initialize SectionClassifier")
            raise CustomException(e, sys)

    async def _get_embedding(self, text: str) -> List[float]:
        """Utility to fetch embeddings from Nomic via raw Ollama."""
        try:
            response = ollama.embeddings(model=self.embed_model, prompt=text)
            return response['embedding']
        except Exception as e:
            logger.error(f"Ollama Embedding Error for text '{text[:20]}...': {e}")
            raise CustomException(e, sys)

    @track_performance
    async def prepare_context(self):
        """
        Fetches section names from DB and pre-calculates Nomic embeddings.
        This should be called once before classification starts.
        """
        try:
            logger.info("Syncing unique section names from database...")
            sections_set = await self.metadata_tool.get_unique_sections()
            self.section_list = list(sections_set)
            
            if not self.section_list:
                logger.warning("Section list is empty. Ensure ChunkModel table is populated.")
                return

            # Batch process embeddings for sections
            embeddings = []
            for section in self.section_list:
                emb = await self._get_embedding(section)
                embeddings.append(emb)
            
            self.section_embeddings = np.array(embeddings)
            logger.info(f"Successfully prepared {len(self.section_list)} section vectors.")
        except Exception as e:
            logger.error("Context preparation failed.")
            raise CustomException(e, sys)

    def _get_top_k_candidates(self, query_vector: List[float], k: int = 5) -> List[str]:
        """Performs cosine similarity to find candidate sections."""
        if self.section_embeddings is None:
            raise ValueError("Section embeddings not initialized. Call prepare_context() first.")
            
        q_vec = np.array(query_vector).reshape(1, -1)
        sims = cosine_similarity(q_vec, self.section_embeddings)[0]
        
        # Get indices of top k scores
        top_indices = np.argsort(sims)[-k:][::-1]
        return [self.section_list[i] for i in top_indices]

    async def _rerank_with_qwen(self, query: str, candidates: List[str]) -> List[str]:
        """Uses Qwen 2.5 to pick the best 2 sections from candidates."""
        try:
            prompt = (
                f"User Query: \"{query}\"\n"
                f"Candidate Banking Sections: {candidates}\n\n"
                "Task: Select the 2 most relevant section names for the query. "
                "Return ONLY a Python-style list of strings. No preamble or chat."
            )
            
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0} # Deterministic output
            )
            
            content = response['message']['content'].strip()
            
            # Robust JSON/List parsing
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end != -1:
                selected = json.loads(content[start:end])
                return selected[:2]
            
            return candidates[:2]
        except Exception as e:
            logger.warning(f"Qwen reranking failed for query: {query}. Falling back to top-2 similarity.")
            return candidates[:2]

    @track_performance
    async def classify_expanded_queries(self, expanded_queries: List[str]) -> Dict[str, List[str]]:
        """
        Main pipeline: Maps expanded queries to a list of relevant sections.
        Returns: { "query_text": ["section_name_1", "section_name_2"] }
        """
        try:
            if not self.section_list:
                await self.prepare_context()

            results = {}
            logger.info(f"Classifying {len(expanded_queries)} query variations...")

            for query in expanded_queries:
                # 1. Get query embedding
                q_emb = await self._get_embedding(query)
                
                # 2. Get 5 similarity-based candidates
                candidates = self._get_top_k_candidates(q_emb, k=5)
                
                # 3. Refine to 2 via Qwen 2.5
                final_sections = await self._rerank_with_qwen(query, candidates)
                
                results[query] = final_sections

            logger.info("Query-to-Section mapping completed successfully.")
            return results

        except Exception as e:
            logger.error("Bulk query classification failed.")
            raise CustomException(e, sys)

