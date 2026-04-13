import json
import asyncio
import sys
import ollama
from typing import List, Dict, Any
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

# Initialize logger
logger = get_log("metadata_enrcichment")

class BatchPDFKeywordPipeline:
    def __init__(self, model_name: str = "qwen2.5:7b", batch_size: int = 6):
        self.model_name = model_name
        self.batch_size = batch_size
        # Limits concurrent calls to the Ollama server to prevent OOM/hangs
        self.semaphore = asyncio.Semaphore(2)
        logger.info(f"Pipeline initialized | Model: {self.model_name} | Batch Size: {self.batch_size}")

    @track_performance
    async def process_batch(self, batch: List[Dict[str, Any]], batch_num: int) -> Dict[str, Any]:
        """
        Sends a batch of text data to the LLM for keyword extraction.
        """
        async with self.semaphore:
            logger.info(f"==> Batch {batch_num}: Processing {len(batch)} items.")
            
            try:
                prompt = (
                    "You are a Retail Banking Domain Expert. Return valid JSON only.\n"
                    "TASK: Assign EXACTLY 5 banking keywords per topic_id.\n"
                    f"DATA: {json.dumps(batch)}"
                )
                
                # Use to_thread to prevent blocking the event loop during the synchronous ollama call
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        ollama.chat,
                        model=self.model_name,
                        format="json",
                        messages=[{"role": "user", "content": prompt}]
                    ), 
                    timeout=90.0
                )
                
                content = response.get("message", {}).get("content", "{}")
                parsed_data = json.loads(content)
                
                logger.info(f"==> Batch {batch_num}: Successfully processed.")
                return parsed_data

            except asyncio.TimeoutError:
                logger.error(f"!!! Batch {batch_num} FAILED: Request timed out after 90s.")
                return {}

            except json.JSONDecodeError as je:
                logger.error(f"!!! Batch {batch_num} FAILED: Invalid JSON returned from LLM. Error: {je}")
                return {}

            except Exception as e:
                # Catch-all for connection issues or Ollama internal errors
                logger.error(f"!!! Batch {batch_num} CRITICAL FAILURE: {str(e)}")
                raise CustomException(f"Pipeline batch {batch_num} failed: {e}", sys)

    async def run_pipeline(self, all_data: List[Dict]):
        """
        Helper to split data and run batches concurrently.
        """
        tasks = []
        for i in range(0, len(all_data), self.batch_size):
            batch = all_data[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            tasks.append(self.process_batch(batch, batch_num))
        
        return await asyncio.gather(*tasks)