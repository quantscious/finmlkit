import logging
import sys
import os
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a logger for your library
logger = logging.getLogger('finmlkit')

def configure_logger():
    """
    Configures the logger based on the presence of LOGGER_DIR in the environment.
    If LOGGER_DIR is set, it logs to a file; otherwise, it only logs to the console.
    """
    # Check for LOGGER_DIR from .env or environment variables
    log_dir = os.getenv('LOGGER_DIR')
    disable_logger = os.getenv('DISABLE_LOGGER', 'false').lower() == 'true'

    if disable_logger:
        # Disable logging by setting the level to CRITICAL
        logger.setLevel(logging.CRITICAL)
    else:
        # Set the logger's level based on whether logging to a file is enabled
        logger.setLevel(logging.DEBUG if log_dir else logging.INFO)

    # 1. Console logging
    # Create a console handler for logging to the terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Only show INFO-level and above on the console

    # Create formatter for the console handler
    console_format = logging.Formatter(
        '%(filename)s:%(lineno)d | %(levelname)s | %(message)s')
    console_handler.setFormatter(console_format)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # 2. File logging if LOGGER_DIR is set
    if log_dir:
        # Use LOGGER_DIR for file logging
        log_dir = os.path.join(log_dir, 'logs')

        # Create the log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Set the base log file name (the handler will manage rotations)
        log_file_path = os.path.join(log_dir, 'finmlkit.log')

        # Create a file handler with rotation
        file_handler = TimedRotatingFileHandler(
            log_file_path, when='midnight', backupCount=7, delay=True)
        file_handler.suffix = "%Y-%m-%d"  # Adds the date to the log file name
        file_handler.setLevel(logging.DEBUG)

        # Create formatter for the file handler
        file_format = logging.Formatter(
            '%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_format)

        # Add file handler to the logger
        logger.addHandler(file_handler)

        # Now that the logger's level is set, this message will be logged
        logger.info(f"File logging is enabled. Logs are saved to {log_file_path}")

    # Prevent propagation to the root logger to avoid duplicate logs
    logger.propagate = False


# Call this function to configure the logger when the module is imported
configure_logger()


if __name__ == "__main__":
    # Test the logger
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")