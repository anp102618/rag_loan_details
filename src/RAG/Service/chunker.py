from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.Utils.logger_setup import get_log
from src.Utils.exception_handler import CustomException


class Chunker:
    """
    Handles the segmentation of long text strings into smaller, overlapping chunks
    optimized for vector embedding and retrieval.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        """
        Initializes the Chunker with specific splitting parameters.

        Args:
            chunk_size (int): The maximum number of characters per chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        self.logger = get_log("Chunker")
        self.logger.info(
            f"[INIT] Configuring splitter | size={chunk_size}, overlap={chunk_overlap}"
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def chunk(self, documents: List[str]) -> List[str]:
        """
        Splits a list of documents into a flattened list of text chunks.

        Args:
            documents (List[str]): A list of full-text strings (from the Extractor).

        Returns:
            List[str]: A list of smaller text segments.
        """
        self.logger.info(f"[CHUNK] Received documents count={len(documents)}")

        if not documents:
            self.logger.warning("[CHUNK] Empty document list provided")
            return []

        try:
            all_chunks: List[str] = []
            total_docs = len(documents)

            for idx, doc in enumerate(documents):
                if not doc or not doc.strip():
                    self.logger.debug(f"[CHUNK] Skipping empty doc at index={idx}")
                    continue

                # Perform the split
                doc_chunks = self.splitter.split_text(doc)
                all_chunks.extend(doc_chunks)

                self.logger.info(
                    f"[CHUNK] Doc {idx + 1}/{total_docs} -> produced {len(doc_chunks)} chunks"
                )

            self.logger.info(f"[CHUNK] Completed | total_chunks_generated={len(all_chunks)}")
            return all_chunks

        except Exception as e:
            self.logger.error("[CHUNK] Critical failure during text segmentation")
            raise CustomException(e, logger=self.logger)