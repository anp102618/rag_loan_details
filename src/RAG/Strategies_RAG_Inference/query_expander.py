import asyncio
import nest_asyncio
from typing import List
import ollama

# Internal Imports
from src.RAG.Strategies_RAG_build.text_guardrails import TextGuardrails
from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("query_expander")
current_logger.set(logger)
nest_asyncio.apply()


class QueryExpander:
    """
    Expands banking queries using Ollama (Qwen 2.5) with safety validation.
    """

    def __init__(self, model_name: str = "qwen-2.5:7b"):
        try:
            self.model_name = model_name
            self.guardrails = TextGuardrails()

            logger.info(f"QueryExpander active with {model_name} (Ollama native).")

        except Exception as e:
            raise CustomException(e, logger)

    @track_performance
    async def expand_query(self, query: str) -> List[str]:
        try:
            # Input Guardrail
            if not self.guardrails.apply(query):
                logger.warning(f"Blocked unsafe input query: {query}")
                return [query]

            #Prompt
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a professional banking query optimizer. "
                        "Generate exactly two alternative versions of the user's question. "
                        "Focus on technical synonyms and banking terminology. "
                        "Output only the expansions, separated by a newline. No preamble."
                    )
                },
                {
                    "role": "user",
                    "content": f"Expand: {query}"
                }
            ]

            # Async call to Ollama
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.3,
                    "num_predict": 150
                }
            )

            # Safe extraction
            content = response.get("message", {}).get("content", "").strip()

            if not content:
                logger.warning("Empty response from Ollama.")
                return [query]
            

            #Parse response
            raw_lines = content.split("\n")

            candidates = [
                line.strip("- ").strip()
                for line in raw_lines
                if line.strip()
            ][:2]

            #  Output Guardrails
            validated_expansions = []

            for candidate in candidates:
                if self.guardrails.apply(candidate):
                    validated_expansions.append(candidate)
                else:
                    logger.warning(f"Guardrail rejected: {candidate}")

            # Final fallback
            if not validated_expansions:
                logger.info("No valid expansions. Returning original query.")
                return [query]

            logger.info(f"Expanded into {len(validated_expansions)} queries.")
            return validated_expansions

        except Exception as e:
            logger.error("Error during query expansion phase.")
            raise CustomException(e, logger)