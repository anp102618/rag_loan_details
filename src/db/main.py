from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from src.config import config

# Engine configuration
engine = create_async_engine(
    config.DATABASE_URL,
    echo=True,
    future=True,
    # Standard practice for async pg
    pool_pre_ping=True 
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=AsyncSession
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for FastAPI or other frameworks."""
    async with AsyncSessionLocal() as session:
        yield session