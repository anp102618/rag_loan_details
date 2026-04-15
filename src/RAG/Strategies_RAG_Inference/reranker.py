import sys
import asyncio
import nest_asyncio
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# Internal Imports
from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("reranker")
current_logger.set(logger)
nest_asyncio.apply()

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initializes the Cross-Encoder. 
        MS-MARCO models are industry standard for RAG re-ranking.
        """
        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"Cross-Encoder loaded: {model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Re-scores chunks against the original query and sorts them.
        
        Args:
            query: The original user query (best for intent matching).
            chunks: The deduplicated list from HybridRetriever.
            top_n: How many chunks to pass to the LLM.
        """
        if not chunks:
            return []

        try:
            # 1. Prepare pairs for the model: [(query, text1), (query, text2), ...]
            pairs = [[query, chunk["chunk_text"]] for chunk in chunks]

            # 2. Predict Relevancy Scores
            scores = self.model.predict(pairs)

            # 3. Attach scores to chunks
            for i, score in enumerate(scores):
                chunks[i]["rerank_score"] = float(score)

            # 4. Sort by score descending
            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

            # 5. Select top N
            final_selection = reranked_chunks[:top_n]
            
            logger.info(f"Reranking complete. Best score: {final_selection[0]['rerank_score']:.4f}")
            return final_selection

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            raise CustomException(e, sys)

    def format_for_llm(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Converts the list of dictionaries into a single clean string 
        for the LLM context window.
        """
        context_parts = []
        for i, chunk in enumerate(chunks):
            # We include metadata for better LLM grounding
            header = f"--- Context Block {i+1} (Source: {chunk.get('metadata', {}).get('section_name', 'General')}) ---"
            body = chunk["chunk_text"]
            context_parts.append(f"{header}\n{body}")
        
        return "\n\n".join(context_parts)