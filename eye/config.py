"""Configuration settings for the eye project."""

from typing import Literal

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Configuration settings."""

    MLFLOW_EXPERIMENT_NAME: str = "eye_experiment"
    MLFLOW_TRACKING_URI: str = "http://localhost:8080"

    LOG_LEVEL: Literal["ERROR", "WARNING", "INFO", "DEBUG"] = "INFO"
