import os
from typing import Tuple
import pandas as pd

from data import DATA_DIR_PATH
from lib.logger import setup_logger


logger = setup_logger(__name__)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function that is to load the data.

    Raises:
        FileNotFoundError: If the data is not found in the data folder

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test data.

    """

    train_path = os.path.join(DATA_DIR_PATH, "train.csv")
    test_path = os.path.join(DATA_DIR_PATH, "test.csv")

    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

    except Exception as e:
        logger.error(e)
        raise FileNotFoundError(
            "Data not found. Please download the data from: https://www.kaggle.com/competitions/playground-series-s4e8/data and put it in the data folder."
        )

    return train, test
