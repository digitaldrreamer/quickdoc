from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PORT: int = 8002
    MAX_FILE_SIZE_MB: int = 15
    LOG_LEVEL: str = "INFO"

    @property
    def MAX_FILE_SIZE_BYTES(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

settings = Settings()