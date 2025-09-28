from loguru import logger
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    log_level: str = "INFO"


settings = Settings()
logger.level(settings.log_level)
