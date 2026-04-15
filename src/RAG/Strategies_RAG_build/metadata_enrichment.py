import json
import asyncio
import sys
import time
import ollama
from typing import List, Dict, Any, Optional

from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

logger = setup_logger("metadata_enrichment")
current_logger.set(logger)

class BatchPDFKeywordPipeline:
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        batch_size: int = 3,
        max_retries: int = 2,
        timeout: int = 540,
        concurrency_limit: int = 1
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(concurrency_limit)

        logger.info(
            f"[INIT] Model={model_name} | Batch={batch_size} | "
            f"Retries={max_retries} | Timeout={timeout}s | Concurrency={concurrency_limit}"
        )

    # ---------------- PROMPT CONSTRUCTION ---------------- #

    def _build_prompt(self, batch: List[Dict[str, Any]]) -> str:
        """Constructs a high-density SME prompt."""
        return (
            "You are an expert Retail Banking Domain Loan Advisor. Return valid JSON only.\n"
            "TASK: Assign EXACTLY 5 banking keywords per topic_id.\n"
                    " - in case relevant keywords not found , then strictly assign keywords as per your domain knowledge.\n"
           
            '{ "topic_id": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"] }\n\n'
            f"DATA:\n{json.dumps(batch, indent=2)}"
        )

    # ---------------- UTILITIES & PARSING ---------------- #

    def _clean_json_string(self, content: str) -> str:
        """Strips markdown artifacts and whitespace."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return content.strip()

    def _safe_json_parse(self, content: str, batch_num: int) -> Dict:
        """Attempts robust JSON parsing with fallback cleanup."""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                logger.warning(f"[BATCH {batch_num}] Attempting Markdown cleanup.")
                cleaned = self._clean_json_string(content)
                return json.loads(cleaned)
            except Exception as e:
                logger.error(f"[BATCH {batch_num}] JSON recovery failed: {e}")
                return {}

    # ---------------- LLM INTERACTION ---------------- #

    async def _call_llm(self, prompt: str, batch_num: int) -> Dict:
        """Executes LLM call safely with timeout and format enforcement."""
        try:
            # Note: format="json" in Ollama forces the model to output a JSON object
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=self.model_name,
                    format="json",
                    messages=[{"role": "user", "content": prompt}]
                ),
                timeout=self.timeout
            )
            content = response.get("message", {}).get("content", "{}")
            return self._safe_json_parse(content, batch_num)
        except asyncio.TimeoutError:
            logger.error(f"[BATCH {batch_num}] Ollama request timed out.")
            raise

    # ---------------- CORE BATCH PROCESS ---------------- #

    @track_performance
    async def process_batch(
        self, 
        batch: List[Dict[str, Any]], 
        batch_num: int
    ) -> Dict[str, Any]:
        """Manages lifecycle of a single batch: Semaphore, Retries, and Backoff."""
        async with self.semaphore:
            start_time = time.perf_counter()
            prompt = self._build_prompt(batch)
            
            logger.info(f"[BATCH {batch_num}] START | Size={len(batch)}")

            for attempt in range(1, self.max_retries + 2):
                try:
                    result = await self._call_llm(prompt, batch_num)
                    
                    if not result:
                        raise ValueError("LLM returned empty JSON")

                    elapsed = round(time.perf_counter() - start_time, 2)
                    logger.info(f"[BATCH {batch_num}] SUCCESS | Time={elapsed}s")
                    return result

                except Exception as e:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"[BATCH {batch_num}] Attempt {attempt} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    if attempt > self.max_retries:
                        break
                    await asyncio.sleep(wait_time)

            logger.error(f"[BATCH {batch_num}] TERMINATED | Exhausted all retries.")
            return {}

    # ---------------- PIPELINE EXECUTION ---------------- #

    async def run_pipeline(self, all_data: List[Dict]) -> List[Dict]:
        """Orchestrates the division of data into batches and executes concurrently."""
        try:
            total_items = len(all_data)
            # Efficient batch slicing
            batches = [
                all_data[i : i + self.batch_size] 
                for i in range(0, total_items, self.batch_size)
            ]
            
            logger.info(f"[PIPELINE] Processing {total_items} items in {len(batches)} batches.")

            tasks = [
                self.process_batch(batch, idx + 1) 
                for idx, batch in enumerate(batches)
            ]

            # Gather results concurrently
            results = await asyncio.gather(*tasks)
            
            # Post-run metrics
            flat_results = {}
            for r in results:
                flat_results.update(r)

            logger.info(f"[PIPELINE COMPLETE] Generated keys for {len(flat_results)} topics.")
            return results

        except Exception as e:
            logger.critical("[PIPELINE CRASH]", exc_info=True)
            raise CustomException(e, sys)