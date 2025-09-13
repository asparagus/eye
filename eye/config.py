"""Configuration settings for the eye project."""

from typing import Literal

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration settings."""

    TENSORBOARD_PROJECT_NAME: str = "eye"
    TENSORBOARD_URL: str = ""
    TENSORBOARD_LOGS: str = "tb_logs"

    LOG_LEVEL: Literal["ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
