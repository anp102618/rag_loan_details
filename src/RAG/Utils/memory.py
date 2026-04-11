# src/RAG/Service/memory_manager.py

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

import ollama

from src.RAG.models import QueryState
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException


class MemoryManager:
    def __init__(self, stm_k: int = 3, max_memory_chars: int = 1000):
        self.stm_k = stm_k
        self.max_memory_chars = max_memory_chars
        self.logger = get_log("MemoryManager")
        
    
    # 1. Short-Term Memory (STM)
    @track_performance
    async def get_stm(self, db: AsyncSession, conversation_id: str) -> str:
        try:
            self.logger.info(f"[STM START] conversation_id={conversation_id}")

            result = await db.execute(
                select(QueryState)
                .where(QueryState.conversation_id == conversation_id)
                .order_by(QueryState.sequence_id.desc())
                .limit(self.stm_k)
            )

            states: List[QueryState] = result.scalars().all()

            # Reverse → chronological order
            states = list(reversed(states))

            stm = []
            for s in states:
                stm.append(f"User: {s.query}")
                stm.append(f"Assistant: {s.answer}")

            stm_text = "\n".join(stm)

            self.logger.info(f"[STM SUCCESS] fetched {len(states)} interactions")

            return stm_text

        except Exception as e:
            self.logger.exception("[STM ERROR]")
            raise CustomException(e)

    
    # 2. Long-Term Memory (LTM)
    @track_performance
    async def get_ltm(self, db: AsyncSession, conversation_id: str) -> str:
        try:
            self.logger.info(f"[LTM FETCH START] conversation_id={conversation_id}")

            result = await db.execute(
                select(QueryState)
                .where(QueryState.conversation_id == conversation_id)
                .order_by(QueryState.sequence_id.desc())
                .limit(1)
            )

            last_state = result.scalars().first()

            if last_state and last_state.memory:
                self.logger.info("[LTM FETCH SUCCESS]")
                return last_state.memory

            self.logger.info("[LTM EMPTY]")
            return ""

        except Exception as e:
            self.logger.exception("[LTM ERROR]")
            raise CustomException(e)

    
    #3. Memory Compression using Ollama
    @track_performance
    async def update_ltm(self, previous_memory: str, interaction: str) -> str:
        try:
            self.logger.info("[LTM UPDATE START]")

            prompt = f"""
You are a memory compression system.

Rules:
- Keep important facts only
- Remove redundancy
- Maintain conversation intent
- Keep it concise

Previous Memory:
{previous_memory}

New Interaction:
{interaction}

Updated Memory:
"""

            response = ollama.chat(
                model="qwen2.5:7b",
                messages=[
                    {"role": "system", "content": "You compress conversation memory."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
            )

            updated_memory = response["message"]["content"].strip()

            # Safety cap
            if len(updated_memory) > self.max_memory_chars:
                updated_memory = updated_memory[-self.max_memory_chars:]

            self.logger.info("[LTM UPDATE SUCCESS] memory compressed")

            return updated_memory

        except Exception as e:
            self.logger.exception("[LTM UPDATE ERROR]")
            raise CustomException(e)