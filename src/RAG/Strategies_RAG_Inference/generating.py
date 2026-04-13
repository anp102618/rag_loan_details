import ollama
import asyncio
from typing import Dict, Any, List, Optional

from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException


class LLMGenerator:
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: Optional[int] = 512,
        max_context_chars: int = 4000  # 🔥 context control
    ) -> None:

        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_context_chars = max_context_chars

        self.logger = get_log("LLMGenerator")

        self.logger.info(
            f"[INIT] Generator | model={model}, max_ctx={max_context_chars}"
        )

    # ----------------------------------
    # 🔹 CONTEXT BUILDER (CORE FIX)
    # ----------------------------------
    def _build_context(
        self,
        chunks: List[Dict[str, Any]]
    ) -> str:

        if not chunks:
            return ""

        # -------------------------
        # 1. Deduplicate
        # -------------------------
        seen = set()
        unique_chunks = []

        for c in chunks:
            text = c.get("chunk_text", "").strip()
            if text and text not in seen:
                seen.add(text)
                unique_chunks.append(c)

        # -------------------------
        # 2. Sort by relevance
        # -------------------------
        unique_chunks = sorted(
            unique_chunks,
            key=lambda x: x.get("final_score", 0),
            reverse=True
        )

        # -------------------------
        # 3. Group by section (coherence)
        # -------------------------
        grouped = {}
        for c in unique_chunks:
            section = c.get("chunk_metadata", {}).get("section_name", "General")
            grouped.setdefault(section, []).append(c)

        # -------------------------
        # 4. Build context with limit
        # -------------------------
        context_parts = []
        total_len = 0

        for section, items in grouped.items():

            section_header = f"\n[{section}]\n"
            section_text = ""

            for c in items:
                txt = c.get("chunk_text", "").strip()

                if total_len + len(txt) > self.max_context_chars:
                    break

                section_text += txt + "\n"
                total_len += len(txt)

            if section_text:
                context_parts.append(section_header + section_text)

            if total_len >= self.max_context_chars:
                break

        return "\n".join(context_parts).strip()

    # ----------------------------------
    # 🔹 PROMPT
    # ----------------------------------
    def _build_prompt(self, query: str, context: str):

        system_prompt = (
            "You are a professional AI assistant.\n\n"
            "Strict Rules:\n"
            "1. Answer ONLY using the provided context.\n"
            "2. If the answer is not present, say:\n"
            "'I do not have enough information in the provided context.'\n"
            "3. Do NOT use prior knowledge.\n"
            "4. Be concise and structured.\n"
        )

        user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    # ----------------------------------
    # 🔹 GENERATE
    # ----------------------------------
    @track_performance
    async def generate(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:

        if not query.strip():
            return "Invalid query."

        try:
            context = self._build_context(context_chunks)

            if not context:
                self.logger.warning("[GEN] Empty context")

            messages = self._build_prompt(query, context)

            response: Dict[str, Any] = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=messages,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "num_predict": self.max_tokens
                }
            )

            answer = response.get("message", {}).get("content", "").strip()

            if not answer:
                return "No response generated."

            return answer

        except Exception as e:
            raise CustomException(e, logger=self.logger)

    # ----------------------------------
    # 🔹 GENERATE WITH SOURCES
    # ----------------------------------
    @track_performance
    async def generate_with_sources(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        try:
            answer = await self.generate(query, context_chunks)

            sources = [
                {
                    "id": c.get("id"),
                    "section": c.get("chunk_metadata", {}).get("section_name"),
                    "topic": c.get("chunk_metadata", {}).get("topic_name")
                }
                for c in context_chunks
            ]

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception as e:
            raise CustomException(e, logger=self.logger)

    def get_config(self):
        return {
            "model": self.model,
            "max_context_chars": self.max_context_chars
        }