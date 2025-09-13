"""Initialize the global configs."""

from dotenv import load_dotenv

from eye.logs import configure_logger
from eye.config import Config


load_dotenv()
CONFIG = Config()

configure_logger(CONFIG.LOG_LEVEL)
