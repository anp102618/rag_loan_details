import sys
import asyncio
from typing import List

# Third Party
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Internal Imports
from src.RAG.Strategies_RAG_build.text_guardrails import TextGuardrails
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

# Initialize Logger
logger = get_log("QueryExpander")

class QueryExpander:
    """
    Expands banking queries using Qwen 2.5 with safety validation via TextGuardrails.
    """
    def __init__(self, model_name: str = "qwen2.5"):
        try:
            self.llm = ChatOllama(
                model=model_name,
                temperature=0.3,  # Lower temperature for precision
                num_predict=150
            )
            # 🔹 Initialize Guardrails
            self.guardrails = TextGuardrails()
            logger.info(f"QueryExpander active with {model_name} and TextGuardrails.")
        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    async def expand_query(self, query: str) -> List[str]:
        """
        Expands input into 2 variations. Applies guardrails to input and outputs.
        """
        try:
            # 1️⃣ Input Guardrail: Stop unsafe queries before they hit the LLM
            if not self.guardrails.apply(query):
                logger.warning(f"Blocked unsafe input query: {query}")
                return [query]  # Fallback to original (or empty list per policy)

            # 2️⃣ Prompt Construction
            system_msg = SystemMessage(
                content=(
                    "You are a professional banking query optimizer. "
                    "Generate exactly two alternative versions of the user's question. "
                    "Focus on technical synonyms and banking terminology. "
                    "Output only the expansions, separated by a newline. No preamble."
                )
            )
            user_msg = HumanMessage(content=f"Expand: {query}")

            # 3️⃣ Async LLM Invocation
            response = await self.llm.ainvoke([system_msg, user_msg])
            
            # Clean response into a list
            raw_lines = response.content.strip().split("\n")
            candidates = [line.strip("- ").strip() for line in raw_lines if line.strip()][:2]

            # 4️⃣ Output Guardrails: Validate each expansion
            validated_expansions = []
            for candidate in candidates:
                if self.guardrails.apply(candidate):
                    validated_expansions.append(candidate)
                else:
                    logger.warning(f"Guardrail rejected expansion: {candidate}")

            # 5️⃣ Final Logic: Ensure we return at least the original if all fails
            if not validated_expansions:
                logger.info("No safe expansions generated. Returning original query.")
                return [query]

            logger.info(f"Expanded into {len(validated_expansions)} safe variations.")
            return validated_expansions

        except Exception as e:
            logger.error("Error during query expansion phase.")
            raise CustomException(e, sys)



