import fitz
import json
import asyncio
import sys
import nest_asyncio
from collections import Counter
from typing import List, Dict, Any

# Internal Project Imports
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.Strategies_RAG_build.text_guardrails import TextGuardrails
from metadata_enrichment import BatchPDFKeywordPipeline

# Initialize structured logger
logger = get_log("DeepPipeline")

# Allow nested event loops for notebook/environment compatibility
nest_asyncio.apply()

class PDFProcessor:
    def __init__(self):
        self.guardrails = TextGuardrails()

    @track_performance
    def extract_hierarchy(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extracts text hierarchy based on font size analysis."""
        try:
            logger.info(f"Step 1: Opening PDF at {pdf_path}")
            doc = fitz.open(pdf_path)
            spans = []
            
            for page in doc:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            for s in l["spans"]:
                                if s["text"].strip():
                                    spans.append({
                                        "text": s["text"].strip(), 
                                        "size": round(s["size"], 1)
                                    })
            
            if not spans:
                logger.warning("No text spans extracted from PDF.")
                return []

            # Analyze font sizes to determine structure
            size_counts = Counter([s["size"] for s in spans])
            unique_sizes = sorted(size_counts.keys(), reverse=True)
            
            section_f = unique_sizes[0]
            topic_f = unique_sizes[1] if len(unique_sizes) > 1 else unique_sizes[0]
            logger.info(f"Hierarchy Rule -> Section Size: {section_f}, Topic Size: {topic_f}")

            results, cur_sec_name, cur_sec_id = [], "Header", "S001"
            s_idx, t_idx = 1, 0

            for span in spans:
                # Section Header Detection
                if span["size"] == section_f and len(span["text"]) < 80:
                    s_idx += 1
                    cur_sec_name, cur_sec_id = span["text"], f"S{s_idx:03}"
                    t_idx = 0
                # Topic Header Detection
                elif span["size"] == topic_f and len(span["text"]) < 80:
                    t_idx += 1
                    results.append({
                        "section_id": cur_sec_id,
                        "section_name": cur_sec_name,
                        "topic_id": f"{cur_sec_id}_T{t_idx:03}",
                        "topic_name": span["text"],
                        "raw_lines": []
                    })
                # Content Collection
                elif results:
                    results[-1]["raw_lines"].append(span["text"])
            
            doc.close()
            logger.info(f"Extraction complete. {len(results)} topics identified.")
            return results

        except Exception as e:
            raise CustomException(f"PDF Extraction Failed: {e}", sys)

async def run_master_pipeline(pdf_path: str , output_path: str = "structured_loan_data.json"):
    """Orchestrates the full extraction, cleaning, and enrichment process."""
    try:
        logger.info("--- PIPELINE STARTING ---")
        
        processor = PDFProcessor()
        pipeline = BatchPDFKeywordPipeline(batch_size=5)
        guardrails = TextGuardrails()

        # 1. EXTRACTION
        raw_data = processor.extract_hierarchy(pdf_path)
        if not raw_data:
            logger.error("No data extracted. Aborting.")
            return

        # 2. CLEANING & DEDUPLICATION
        logger.info("Step 2: Cleaning and Deduplicating content...")
        processed_items = []
        for item in raw_data:
            raw_string = " ".join(item.pop("raw_lines"))
            clean_text = guardrails.apply(raw_string)
            
            if clean_text:
                item["text"] = clean_text
                processed_items.append(item)

        logger.info(f"Step 2 Complete: {len(processed_items)} items ready for LLM.")

        # 3. LLM BATCH KEYWORD GENERATION
        logger.info(f"Step 3: Triggering Concurrent LLM Batches...")
        
        # Prepare batches for concurrent execution
        batch_tasks = []
        for i in range(0, len(processed_items), pipeline.batch_size):
            batch = processed_items[i : i + pipeline.batch_size]
            batch_num = (i // pipeline.batch_size) + 1
            
            # Slim down input for the LLM
            llm_input = [{"topic_id": x["topic_id"], "text": x["text"][:600]} for x in batch]
            batch_tasks.append(pipeline.process_batch(llm_input, batch_num))

        # Run batches concurrently (respecting the internal semaphore)
        all_keyword_maps = await asyncio.gather(*batch_tasks)

        # Merge results back into the final list
        final_output = []
        # Combine all dictionary results from batches into one lookup
        master_keywords = {k: v for d in all_keyword_maps for k, v in d.items()}

        for item in processed_items:
            tid = item["topic_id"]
            item["keywords"] = master_keywords.get(tid, ["Banking", "Retail", "General"])
            final_output.append(item)

        # 4. EXPORT
        output_path = "structured_loan_data.json"
        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=4)
        
        logger.info(f"PIPELINE SUCCESS | Results saved to: {output_path}")

    except Exception as e:
        logger.critical(f"Master Pipeline crashed: {e}")
        raise CustomException(e, sys)

def start_processing(file_path: str):
    """
    Entry point to trigger the asynchronous master pipeline 
    from a synchronous context.
    """
    try:
        logger.info(f"Starting process for file: {file_path}")
        # asyncio.run handles the event loop lifecycle
        asyncio.run(run_master_pipeline(file_path))
        logger.info("Process completed successfully.")
        
    except Exception as e:
        # Catching at the entry point to ensure we log the final crash state
        logger.critical(f"Entry point 'start_processing' failed: {e}")
        raise CustomException(e, sys)

# Example usage:
# start_processing(r"F:\Loan_Details_RAG_System1\Data\loan_products_final.pdf")