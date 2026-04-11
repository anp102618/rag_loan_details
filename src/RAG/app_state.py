from typing import Optional
from fastapi import HTTPException

from src.RAG.Service.pipeline import RAGPipeline

# Shared application state
rag_container: dict[str, Optional[RAGPipeline]] = {
    "pipeline": None
}


# Dependency to safely access pipeline
def get_pipeline() -> RAGPipeline:
    pipeline = rag_container.get("pipeline")

    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Check server startup."
        )

    return pipeline