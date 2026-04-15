import ollama
import asyncio
from typing import List, Optional
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

class DocumentEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text", max_concurrent: int = 10):
        self.model_name = model_name
        self.logger = get_log("DocumentEmbedder")
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def _embed_single(self, text: str, index: int = 0) -> Optional[List[float]]:
        """Embed with concurrency control. index is now optional to prevent TypeErrors."""
        async with self._semaphore:
            try:
                response = await asyncio.to_thread(
                    ollama.embeddings, model=self.model_name, prompt=text
                )
                return response.get("embedding")
            except Exception as e:
                self.logger.error(f"[EMBED] Failed at index={index}: {str(e)}")
                return None

    async def embed_query(self, query: str) -> Optional[List[float]]:
        """Specific method for queries (adds prefixing for nomic models)."""
        prefix = "search_query: " if "nomic" in self.model_name.lower() else ""
        return await self._embed_single(f"{prefix}{query}", index=-1)

    @track_performance
    async def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        prefix = "search_document: " if "nomic" in self.model_name.lower() else ""
        tasks = [self._embed_single(f"{prefix}{t}", i) for i, t in enumerate(texts)]
        return await asyncio.gather(*tasks)