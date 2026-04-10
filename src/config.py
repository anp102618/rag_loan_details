from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache


class PathConfig(BaseModel):
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    logs: Path = base_dir / "logs"
    data: Path = base_dir / "data"


class AppConstants(BaseModel):
    app_name: str = "My App"
    api_version: str = "v1"
    default_page_size: int = 20


class Settings(BaseSettings):
    DATABASE_URL: str

    paths: PathConfig = Field(default_factory=PathConfig)
    constants: AppConstants = Field(default_factory=AppConstants)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


config = get_settings()