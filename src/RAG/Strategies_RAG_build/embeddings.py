import ollama
import asyncio
from typing import List, Optional
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

class DocumentEmbedder:
    """
    Optimized embedding generator using Ollama.
    Supports parallel execution with controlled concurrency via Semaphore.
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        max_concurrent: int = 10
    ) -> None:
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.logger = get_log("DocumentEmbedder")
        self._semaphore = asyncio.Semaphore(max_concurrent)

        self.logger.info(
            f"[INIT] Embedder | model={model_name}, concurrency={max_concurrent}"
        )

    async def _embed_single(self, text: str, index: int) -> Optional[List[float]]:
        """Embed a single text with concurrency control."""
        async with self._semaphore:
            try:
                # to_thread is used because the ollama library is synchronous
                response = await asyncio.to_thread(
                    ollama.embeddings,
                    model=self.model_name,
                    prompt=text
                )

                embedding = response.get("embedding")
                if not embedding:
                    self.logger.error(f"[EMBED] Empty embedding at index={index}")
                    return None

                return embedding

            except Exception as e:
                self.logger.error(f"[EMBED] Failed at index={index}: {str(e)}")
                return None

    @track_performance
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Parallel embedding with controlled concurrency."""
        if not texts:
            self.logger.warning("[EMBED] Empty input list")
            return []

        self.logger.info(f"[EMBED] Processing batch size={len(texts)}")

        try:
            tasks = [
                self._embed_single(text, idx)
                for idx, text in enumerate(texts)
            ]

            results = await asyncio.gather(*tasks)
            success_count = sum(1 for r in results if r is not None)

            self.logger.info(
                f"[EMBED] Completed | success={success_count}/{len(texts)}"
            )
            return results

        except Exception as e:
            self.logger.error("[EMBED] Critical batch failure")
            raise CustomException(e, sys)