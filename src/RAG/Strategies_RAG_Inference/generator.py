import sys
import ollama
import asyncio
import nest_asyncio
from typing import List, Any

from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("generator")
current_logger.set(logger)
nest_asyncio.apply()

class LLMGenerator:
    """
    Refactored Generator using Ollama (qwen2.5:7b).
    Handles ChunkModel object extraction and prompt synthesis internally.
    """
    
    def __init__(self, model_name: str = "phi3.5"):
        self.model_name = model_name

    @track_performance
    async def generate_response(
        self, 
        queries: List[str], 
        retrieved_chunks: List[Any], 
        ltm: str, 
        stm: str
    ) -> str:
        """
        Args:
            queries: List of user queries (first two will be merged).
            retrieved_chunks: List of ChunkModel objects with 'retrieval_source' attributes.
            ltm: Long-term memory string.
            stm: Short-term memory string.
        """
        try:
            # 1. Merge Queries
            merged_query = f"{queries[0]}. {queries[1]}" if len(queries) > 1 else queries[0]

            # 2. Extract and Categorize Text from ChunkModels
            primary_list = []
            context_list = []

            for chunk in retrieved_chunks:
                # Use getattr to safely grab the source tag we injected in the retriever
                source = getattr(chunk, "retrieval_source", "unknown").lower()
                text_block = f"[{source.upper()}] {chunk.chunk_text}"

                if "context" in source:
                    context_list.append(text_block)
                else:
                    primary_list.append(text_block)

            primary_text = "\n".join(primary_list) if primary_list else "No direct data."
            context_text = "\n".join(context_list) if context_list else "No extra context."

            # 3. Construct the RAG Context Prompt
            full_prompt = f"""
            SYSTEM ROLE: You are a precise technical assistant. 
            
            ### PRIMARY REFERENCE DATA:
            {primary_text}

            ### SUPPORTING BACKGROUND INFORMATION:
            {context_text}

            ### CONVERSATION MEMORY:
            - LTM: {ltm}
            - STM: {stm}

            ### USER QUESTION:
            {merged_query}

            ### INSTRUCTIONS:
            1. Priority: Answer using PRIMARY REFERENCE DATA.
            2. Supplemental: Use SUPPORTING BACKGROUND only for broader context.
            3. Context: Maintain continuity with CONVERSATION MEMORY.
            4. Honesty: If information is missing from the data, state you don't know.
            """

            logger.info(f"Invoking Ollama ({self.model_name}) for merged query.")

            # 4. Ollama Execution
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': full_prompt}],
                options={'temperature': 0.1}
            )

            return response.get('message', {}).get('content', "No response generated.")

        except Exception as e:
            logger.error(f"LLM Generation Error: {str(e)}")
            raise CustomException(e, sys)