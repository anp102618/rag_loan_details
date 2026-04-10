import ollama
import asyncio
from typing import Dict, Any
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException


class Generator:
    """
    Handles the final generation phase of the RAG pipeline by prompting
    a Local LLM with retrieved context and user queries.
    """

    def __init__(self, model: str = "qwen2.5:7b") -> None:
        """
        Initializes the generator with a specific LLM.

        Args:
            model (str): The name of the Ollama model to use for generation.
        """
        self.model = model
        self.logger = get_log("Generator")
        self.logger.info(f"[INIT] Generator initialized with model='{self.model}'")

    @track_performance
    async def generate(self, query: str, context: str) -> str:
        """
        Generates a response based strictly on the provided context.

        Args:
            query (str): The normalized user question.
            context (str): The concatenated string of retrieved document chunks.

        Returns:
            str: The generated response from the LLM.
        """
        self.logger.info(f"[GEN] Generating response | query_len={len(query)}")

        if not context.strip():
            self.logger.warning("[GEN] Empty context provided; results may be generic.")

        # Constructing a strict system prompt to control model behavior
        system_content = (
            "You are a professional assistant. Use the provided context to answer the question. "
            "If the answer is not in the context, say that you do not have enough information. "
            "Do not use outside knowledge. Answer concisely and accurately."
        )
        
        user_prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

        try:
            # Execute the synchronous Ollama chat call in a separate thread
            response: Dict[str, Any] = await asyncio.to_thread(
                ollama.chat,
                model=self.model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": 0.1,  # Keep responses deterministic and focused
                    "top_p": 0.9
                }
            )

            answer = response.get("message", {}).get("content", "")
            
            if not answer:
                self.logger.error("[GEN] Received an empty response from the model")
                return "I'm sorry, I was unable to generate a response."

            self.logger.info(f"[GEN] Success | Response length={len(answer)} chars")
            return answer

        except Exception as e:
            self.logger.error(f"[GEN] Critical failure during LLM generation: {str(e)}")
            raise CustomException(e, logger=self.logger)