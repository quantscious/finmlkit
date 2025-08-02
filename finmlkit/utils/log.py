import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_logging():
    """
    Sets up logging configuration with separate console and file handlers.
    Console logs go to stdout while file logs are stored in the directory specified by LOGGER_DIR.
    """
    root_logger = logging.getLogger()

    # Skip setup if handlers are already configured
    if root_logger.hasHandlers():
        return

    # Get configuration from environment
    file_level = os.getenv('FILE_LOGGER_LEVEL', 'DEBUG').upper()
    console_level = os.getenv('CONSOLE_LOGGER_LEVEL', 'INFO').upper()
    log_file_path = os.getenv('LOG_FILE_PATH')

    # Set root logger to the minimum of console and file levels to capture all needed logs
    root_level = min(logging.getLevelName(file_level), logging.getLevelName(console_level))
    root_logger.setLevel(root_level)

    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = '%(name)s:%(lineno)d | %(levelname)s | %(message)s'
    console_handler.setFormatter(logging.Formatter(console_format))
    root_logger.addHandler(console_handler)

    # Configure file handler if log directory is specified
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)

        file_handler = TimedRotatingFileHandler(
            log_file_path,
            when='midnight',
            backupCount=7,
            delay=True
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(file_level)

        file_format = '%(asctime)s | %(name)s.%(funcName)s:%(lineno)d | %(levelname)s | %(message)s'
        file_handler.setFormatter(logging.Formatter(file_format))

        root_logger.addHandler(file_handler)
        root_logger.info(f"File logging is enabled. Logs are saved to {log_file_path}")

    # Suppress verbose third-party logs
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Prevent propagation to avoid duplicate logs
    root_logger.propagate = False

def get_logger(name: str):
    """
    Returns a logger for the given module name, ensuring logging is configured.
    """
    setup_logging()
    return logging.getLogger(name)
