import os
import fitz  # PyMuPDF
import asyncio
from typing import List
from src.Utils.logger_setup import get_log
from src.Utils.exception_handler import CustomException


class PDFIngestor:
    """
    Handles the ingestion of PDF documents from a specified directory.
    """

    def __init__(self) -> None:
        self.logger = get_log("Ingestor")

    async def load_pdfs(self, folder_path: str) -> List[str]:
        """
        Scans a folder and extracts text from all PDF files found.

        Args:
            folder_path (str): The path to the directory containing PDFs.

        Returns:
            List[str]: A list of extracted text strings, one per PDF.
        """
        self.logger.info(f"[INGEST] Scanning folder: {folder_path}")

        if not os.path.exists(folder_path):
            self.logger.error(f"[INGEST] Folder not found: {folder_path}")
            return []

        pdf_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(".pdf")
        ]
        
        self.logger.info(f"[INGEST] Found {len(pdf_files)} PDF(s)")

        all_text: List[str] = []

        try:
            for pdf in pdf_files:
                file_path = os.path.join(folder_path, pdf)
                self.logger.info(f"[PDF] Reading file: {pdf}")

                # Offload blocking I/O to a separate thread
                text = await asyncio.to_thread(self._read_pdf, file_path)
                if text.strip():
                    all_text.append(text)

            return all_text

        except Exception as e:
            self.logger.error(f"[INGEST] Extraction failed: {str(e)}")
            raise CustomException(e, logger=self.logger)

    def _read_pdf(self, path: str) -> str:
        """
        Synchronous helper to read PDF content.
        """
        try:
            with fitz.open(path) as doc:
                return "\n".join(page.get_text() for page in doc)
        except Exception as e:
            self.logger.warning(f"[PDF] Could not read {path}: {e}")
            return ""