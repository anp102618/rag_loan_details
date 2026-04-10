import re
from src.Utils.logger_setup import get_log
from src.Utils.exception_handler import CustomException


class Normalizer:
    """
    Standardizes and cleans text to improve embedding quality and retrieval accuracy.
    """

    def __init__(self) -> None:
        self.logger = get_log("Normalizer")

    @staticmethod
    def normalize(text: str) -> str:
        """
        Cleans the input text by lowercasing and collapsing all whitespace.

        Args:
            text (str): The raw text string to be normalized.

        Returns:
            str: The sanitized, single-line, lowercase string.
        """
        if not text:
            return ""

        try:
            # 1. Convert to lowercase
            # 2. Replace newlines, tabs, and multiple spaces with a single space
            # 3. Strip leading/trailing whitespace
            return re.sub(r"\s+", " ", text.lower()).strip()

        except Exception as e:
            # Note: Since this is a static method, we usually handle exceptions 
            # in the calling service, but we can wrap it for safety.
            raise CustomException(e)

    def process_batch(self, texts: list[str]) -> list[str]:
        """
        Helper method to normalize a list of strings with logging.
        """
        self.logger.info(f"[NORMALIZE] Processing batch of size={len(texts)}")
        try:
            return [self.normalize(t) for t in texts if t]
        except Exception as e:
            self.logger.error("[NORMALIZE] Batch processing failed")
            raise CustomException(e, logger=self.logger)