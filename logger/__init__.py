import logging
import sys


def setup_logger(name: str | None = None) -> logging.Logger:
    """A function to setup the logger

    Args:
        name (str | None, optional): Logger name. Defaults to None.

    Returns:
        logging.Logger: Configured logger
    """

    file_handler = logging.FileHandler("logs.log", mode="a")
    stream_handler = logging.StreamHandler(stream=sys.stdout)

    logger = logging.getLogger(name or __name__)
    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s %(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[file_handler, stream_handler],
    )

    logger.setLevel(logging.INFO)

    return logger


logger = setup_logger("Init")
logger.info("Logger setup complete")
