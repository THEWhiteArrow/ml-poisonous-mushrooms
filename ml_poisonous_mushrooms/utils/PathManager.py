from enum import Enum
from pathlib import Path


class PathManager(Enum):
    OUTPUT_DIR_PATH = Path(__file__).parent.parent.parent / "results"
