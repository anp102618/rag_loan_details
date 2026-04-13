import sys
from typing import Set
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import JSONB

# Internal Imports
from src.RAG.models import ChunkModel
from src.db.main import get_session
from src.Utils.logger_setup import get_log, track_performance
from src.Utils.exception_handler import CustomException

logger = get_log("MetadataExtractor")

class MetadataExtractor:
    """
    Utility class to interface with ChunkModel metadata.
    """
    
    @track_performance
    async def get_unique_sections(self) -> Set[str]:
        """
        Queries the database for all unique section_name values 
        stored within the JSONB chunk_metadata column.
        
        Returns:
            Set[str]: A unique set of section names found in the table.
        """
        try:
            section_names: Set[str] = set()
            
            async for session in get_session():
                # 1. Access the 'section_name' key inside the JSONB column
                # The '->>' operator in Postgres returns the value as text
                stmt = select(
                    func.distinct(ChunkModel.chunk_metadata['section_name'].astext)
                ).where(
                    ChunkModel.chunk_metadata['section_name'] != None
                )
                
                result = await session.execute(stmt)
                
                # 2. Extract and clean the results
                # scalars() returns the individual distinct strings
                for row in result.scalars().all():
                    if row:
                        section_names.add(str(row).strip())
                
                # break after session is utilized
                break

            logger.info(f"Successfully extracted {len(section_names)} unique sections.")
            return section_names

        except Exception as e:
            logger.error("Failed to extract section names from ChunkModel.")
            raise CustomException(e, sys)


    
