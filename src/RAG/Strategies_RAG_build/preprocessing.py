import fitz
import json
import asyncio
import sys
import nest_asyncio
import time
import re
from collections import Counter
from typing import List, Dict, Any

from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException
from src.RAG.Strategies_RAG_build.text_guardrails import TextGuardrails
from src.RAG.Strategies_RAG_build.metadata_enrichment import BatchPDFKeywordPipeline

# ---------------- GLOBAL SETUP ---------------- #
logger = setup_logger("preprocessing")
current_logger.set(logger)
nest_asyncio.apply()

# ---------------- FALLBACK KEYWORD GENERATOR ---------------- #
DEFAULT_KEYWORDS = ["Banking", "Loan", "Finance", "Customer", "Policy"]

def generate_fallback_keywords(text: str) -> List[str]:
    """
    Deterministic keyword extraction.
    Used when LLM output is missing/invalid.
    Guarantees 5 keywords always.
    """
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    stopwords = {"this", "that", "with", "from", "have", "will", "your"}
    words = [w for w in words if w not in stopwords]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq, key=freq.get, reverse=True)

    if len(sorted_words) >= 5:
        return sorted_words[:5]

    return (sorted_words + DEFAULT_KEYWORDS)[:5]


# ---------------- PDF PROCESSOR ---------------- #
class PDFProcessor:
    def __init__(self):
        self.guardrails = TextGuardrails()

    @track_performance
    def extract_hierarchy(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extracts structured sections/topics using font hierarchy.
        """
        try:
            logger.info(f"[STEP 1] Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)

            spans = []

            # -------- Extract spans -------- #
            for page in doc:
                blocks = page.get_text("dict")["blocks"]

                for b in blocks:
                    if "lines" not in b:
                        continue

                    for l in b["lines"]:
                        for s in l["spans"]:
                            text = s["text"].strip()
                            if text:
                                spans.append({
                                    "text": text,
                                    "size": round(s["size"], 1)
                                })

            doc.close()

            if not spans:
                logger.warning("[STEP 1] No text found in PDF.")
                return []

            # -------- Detect hierarchy -------- #
            size_counts = Counter([s["size"] for s in spans])
            unique_sizes = sorted(size_counts.keys(), reverse=True)

            section_f = unique_sizes[0]
            topic_f = unique_sizes[1] if len(unique_sizes) > 1 else unique_sizes[0]

            logger.info(f"[STEP 1] Font Rules → Section={section_f}, Topic={topic_f}")

            results = []
            cur_sec_name, cur_sec_id = "Header", "S001"
            s_idx, t_idx = 1, 0

            # -------- Build structure -------- #
            for span in spans:
                text = span["text"]

                if span["size"] == section_f and len(text) < 100:
                    s_idx += 1
                    cur_sec_name = text
                    cur_sec_id = f"S{s_idx:03}"
                    t_idx = 0

                elif span["size"] == topic_f and len(text) < 100:
                    t_idx += 1

                    results.append({
                        "section_id": cur_sec_id,
                        "section_name": cur_sec_name,
                        "topic_id": f"{cur_sec_id}_T{t_idx:03}",
                        "topic_name": text,
                        "raw_lines": []
                    })

                elif results:
                    results[-1]["raw_lines"].append(text)

            logger.info(f"[STEP 1 COMPLETE] Topics extracted: {len(results)}")
            return results

        except Exception as e:
            logger.exception("[ERROR] Extraction failed")
            raise CustomException(f"PDF Extraction Failed: {e}", sys)


# ---------------- MASTER PIPELINE ---------------- #
async def preprocessing_pipeline(pdf_path: str, output_path: str = "structured_loan_data.json"):
    try:
        start_time = time.perf_counter()

        processor = PDFProcessor()
        guardrails = TextGuardrails()

        pipeline = BatchPDFKeywordPipeline(
            batch_size=3,
            max_retries=2,
            timeout=540,
            concurrency_limit=1
        )

        # ---------------- STEP 1: EXTRACTION ----------------
        raw_data = processor.extract_hierarchy(pdf_path)
        logger.info(f"[STEP 1] Extracted topics: {len(raw_data)}")

        # ---------------- STEP 2: CLEANING ----------------
        processed_items = []

        for item in raw_data:
            text = guardrails.apply(" ".join(item.pop("raw_lines")))
            if text:
                item["text"] = text
                processed_items.append(item)

        logger.info(f"[STEP 2] Cleaned items: {len(processed_items)}")

        # ---------------- STEP 3: LLM ----------------
        batch_tasks = []

        for i in range(0, len(processed_items), pipeline.batch_size):
            batch = processed_items[i:i + pipeline.batch_size]
            batch_num = (i // pipeline.batch_size) + 1

            llm_payload = [
                {"topic_id": x["topic_id"], "text": x["text"][:500]}
                for x in batch
            ]

            batch_tasks.append(pipeline.process_batch(llm_payload, batch_num))

        keyword_results = await asyncio.gather(*batch_tasks)

        logger.info(f"[STEP 3] LLM completed: {len(processed_items)} items")

        # ---------------- STEP 4: ENRICHMENT ----------------
        master_map = {}
        for d in keyword_results:
            if d:
                master_map.update(d)

        final_output = []
        llm_used = 0
        fallback_used = 0

        for item in processed_items:
            tid = item["topic_id"]
            keywords = master_map.get(tid)

            if isinstance(keywords, list) and len(keywords) >= 3:
                item["keywords"] = keywords[:5]
                llm_used += 1
            else:
                item["keywords"] = generate_fallback_keywords(item["text"])
                fallback_used += 1

            final_output.append(item)

        logger.info(
            f"[STEP 4] Enriched: {len(final_output)} "
            f"(LLM={llm_used} | Fallback={fallback_used})"
        )

        # ---------------- STEP 5: EXPORT ----------------
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        total_time = round(time.perf_counter() - start_time, 2)

        logger.info(
            f"[PIPELINE DONE] Total={len(final_output)} | Time={total_time}s"
        )

    except Exception as e:
        logger.error(f"[PIPELINE ERROR] {str(e)}")
        raise CustomException(e, sys)