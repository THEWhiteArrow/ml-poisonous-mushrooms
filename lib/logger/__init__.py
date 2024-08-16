import logging
import sys


def setup_logger(name: str | None = None) -> logging.Logger:
    """A function to setup the logger

    Args:
        name (str | None, optional): Logger name. Defaults to None.

    Returns:
        logging.Logger: Configured logger
    """

    # Create a logger object
    logger = logging.getLogger(name or __name__)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler("logs.log", mode="a")
    file_handler.setLevel(logging.DEBUG)  # Adjust the log level as needed
    file_formatter = logging.Formatter(
        '%(levelname)s %(name)s %(asctime)s | %(message)s')
    file_handler.setFormatter(file_formatter)

    # Create stream handler (for console output)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.INFO)  # Adjust the log level as needed
    stream_formatter = logging.Formatter(
        '%(levelname)s %(name)s %(asctime)s | %(message)s')
    stream_handler.setFormatter(stream_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Set logger level
    logger.setLevel(logging.DEBUG)  # Adjust the log level as needed

    return logger


# Initialize logger
logger = setup_logger("Init")

# Example usage
logger.info("Logger setup complete")
