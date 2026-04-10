import ollama
import asyncio
from typing import List
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException


class OllamaEmbeddings:
    """
    Handles the generation of high-dimensional vector embeddings 
    using local models via Ollama.
    """

    def __init__(self, model: str = "nomic-embed-text") -> None:
        """
        Initializes the embedder with a specific local model.

        Args:
            model (str): The name of the Ollama model to use for embeddings.
        """
        self.model = model
        self.logger = get_log("Embedder")
        self.logger.info(f"[INIT] OllamaEmbeddings using model='{self.model}'")

    @track_performance
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Converts a list of text strings into a list of vector embeddings.

        Args:
            texts (List[str]): The text chunks to be vectorized.

        Returns:
            List[List[float]]: A list of embeddings (arrays of floats).
        """
        self.logger.info(f"[EMBED] Starting vectorization | count={len(texts)}")

        if not texts:
            self.logger.warning("[EMBED] No text provided for embedding")
            return []

        try:
            results: List[List[float]] = []

            for i, t in enumerate(texts):
                # Using to_thread because the ollama library is currently synchronous
                res = await asyncio.to_thread(
                    ollama.embeddings,
                    model=self.model,
                    prompt=t
                )
                
                if "embedding" in res:
                    results.append(res["embedding"])
                else:
                    self.logger.error(f"[EMBED] Malformed response at index {i}")

            self.logger.info(f"[EMBED] Success | total_embeddings={len(results)}")
            return results

        except Exception as e:
            self.logger.error("[EMBED] Critical failure during Ollama API call")
            raise CustomException(e, logger=self.logger)