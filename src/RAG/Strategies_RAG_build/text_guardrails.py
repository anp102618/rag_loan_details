import re
import binascii
import unicodedata
import sys
from hashlib import blake2b
from typing import List, Set, Optional, Dict

# Importing your specific utility modules
from src.Utils.logger_setup import setup_logger, current_logger, track_performance
from src.Utils.exception_handler import CustomException

# Initialize logger
logger = setup_logger("text_guardrails")
current_logger.set(logger)

class TextGuardrails:
    def __init__(self, semantic_threshold: float = 0.85, num_perm: int = 64):
        try:
            self.threshold = semantic_threshold
            self.num_perm = num_perm
            self.exact_hashes: Set[str] = set()
            self.minhash_registry: List[List[int]] = []
            
            # Compiled Regex Patterns for PII
            self.pii_patterns = {
                "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
                "PHONE": r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
                "ADHAAR": r"\b\d{4}\s\d{4}\s\d{4}\b",
                "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b"
            }
            logger.info("TextGuardrails initialized successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def _detect_safety_risks(self, text: str) -> bool:
        """Heuristic detection for Prompt Injection and Harmful Intent."""
        injection_signals = [
            "ignore all previous instructions", "system prompt",
            "forget everything", "bypass safety", "dan mode"
        ]
        harmful_keywords = ["bomb", "exploit", "hack", "kill", "suicide"]
        
        normalized_text = text.lower()
        
        if any(signal in normalized_text for signal in injection_signals):
            logger.warning("Potential Prompt Injection detected.")
            return True
            
        if any(word in normalized_text.split() for word in harmful_keywords):
            logger.warning("Harmful intent keywords detected.")
            return True

        return False

    def _mask_pii(self, text: str) -> str:
        """Identifies and replaces sensitive data with placeholders."""
        for label, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                logger.info(f"Masking {len(matches)} instances of {label}.")
                text = re.sub(pattern, f"[{label}]", text)
        return text

    def is_duplicate(self, text: str) -> bool:
        """Checks for exact and near-duplicate text using MinHash."""
        try:
            h = blake2b(text.encode(), digest_size=16).hexdigest()
            if h in self.exact_hashes:
                return True
            
            if len(text) < 20: 
                return False 
            
            clean = "".join(text.lower().split())
            shingles = {clean[i:i + 5] for i in range(len(clean) - 4)}
            if not shingles: 
                return False
            
            current_sig = [
                min(binascii.crc32(f"{s}{j}".encode()) & 0xffffffff for s in shingles) 
                for j in range(self.num_perm)
            ]

            for existing_sig in self.minhash_registry:
                match_count = sum(a == b for a, b in zip(current_sig, existing_sig))
                if (match_count / self.num_perm) >= self.threshold:
                    logger.info("Semantic duplicate detected via MinHash.")
                    return True

            self.exact_hashes.add(h)
            self.minhash_registry.append(current_sig)
            return False
        except Exception as e:
            raise CustomException(e, sys)

    @track_performance
    def apply(self, text: str) -> Optional[str]:
        """
        Runs the full guardrail pipeline.
        Returns processed text if valid and length > 5, otherwise None.
        """
        try:
            if not text or len(text.strip()) == 0:
                return None

            # 1. Normalization
            text = unicodedata.normalize("NFKC", text)
            
            # 2. Safety Checks
            if self._detect_safety_risks(text):
                return None

            # 3. PII Masking
            text = self._mask_pii(text)
            
            # 4. Deduplication Logic
            text = " ".join(text.split())
            sentences = re.split(r'(?<=[.!?]) +', text)
            
            unique_sentences = []
            for s in sentences:
                clean_s = s.strip()
                if clean_s and not self.is_duplicate(clean_s):
                    unique_sentences.append(clean_s)
            
            result = " ".join(unique_sentences)
            
            # Final return check
            return result if len(result) > 5 else None

        except Exception as e:
            logger.error("Error occurred in TextGuardrails.apply")
            raise CustomException(e, sys)