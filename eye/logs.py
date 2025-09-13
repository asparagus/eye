import logging


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors and enhanced information."""
        # Get color for this log level
        color = self.COLORS.get(record.levelname, "")

        # Create formatted timestamp
        formatted_time = self.formatTime(record, "%Y-%m-%d %H:%M:%S")

        # Format the message with color
        formatted_message = (
            f"{color}[{formatted_time}] "
            f"{record.levelname:<8} "
            f"{record.name}:{record.lineno} - "
            f"{record.getMessage()}{self.RESET}"
        )

        # Add exception info if present
        if record.exc_info:
            formatted_message += f"\n{self.formatException(record.exc_info)}"

        return formatted_message


def configure_logger(config_log_level: str) -> None:
    """Configure logger with enhanced formatting and colors.

    Args:
        config_log_level: Log level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_level = getattr(logging, config_log_level.upper())

    # Create handler with custom formatter
    handler = logging.StreamHandler()
    formatter = ColoredFormatter()
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    handler.setLevel(log_level)

    # Prevent propagation to avoid duplicates from third-party libraries
    root_logger.propagate = False

    # Configure specific loggers that might have their own handlers
    for logger_name in ["lightning", "lightning.pytorch"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False
