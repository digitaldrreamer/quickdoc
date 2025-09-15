from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Basic server configuration
    PORT: int = 8005
    MAX_FILE_SIZE_MB: int = 15
    LOG_LEVEL: str = "INFO"

    # * Feature flags - Enable/disable components to save resources
    ENABLE_SUMMARIZATION: bool = True
    ENABLE_EMBEDDING_MODEL: bool = True
    ENABLE_TOKEN_COUNTING: bool = True
    ENABLE_DOCUMENT_PROCESSING: bool = True
    
    # * Document processing feature flags
    ENABLE_PDF_PROCESSING: bool = True
    ENABLE_DOCX_PROCESSING: bool = True
    ENABLE_IMAGE_OCR: bool = True
    ENABLE_MARKDOWN_PROCESSING: bool = True
    ENABLE_EPUB_PROCESSING: bool = True
    
    # * AI Model Configuration
    SUMMARIZATION_MODEL: str = "google/flan-t5-small"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # * External service tokens (optional)
    HUGGING_FACE_HUB_TOKEN: Optional[str] = None
    
    # * Advanced configuration
    SUMMARIZATION_TIMEOUT: int = 300  # 5 minutes
    MAX_SUMMARIZATION_QUEUE_SIZE: int = 100

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()