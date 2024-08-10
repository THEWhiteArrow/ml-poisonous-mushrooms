import sys
import logging

file_handler = logging.FileHandler("logs.log", mode="w")
stream_handler = logging.StreamHandler(stream=sys.stdout)


logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s %(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[file_handler, stream_handler],
)

logging.info("Logging has been setup.")
