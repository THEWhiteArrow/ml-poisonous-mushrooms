import json
import os
from typing import List, Tuple, TypedDict, cast
import pandas as pd
from dataclasses import dataclass

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


@dataclass
class EnsembleConfigDict(TypedDict):
    model_run: str
    model_combination_names: List[str]


def load_ensemble_config() -> EnsembleConfigDict:
    """A function that is to load the ensemble config.

    Raises:
        FileNotFoundError: If the ensemble config is not found in the data folder

    Returns:
        EnsembleConfigDict: The ensemble config dictionary.
    """

    ensemble_config_path = (
        os.path.dirname(os.path.abspath(__file__)) + "/../ensemble_config.json"
    )

    try:
        ensemble_config = cast(
            EnsembleConfigDict, json.load(open(ensemble_config_path))
        )

    except FileNotFoundError as e:
        logger.error(e)
        raise FileNotFoundError(
            "Ensemble config not found. Please create a ensemble_config.json file in the root of the project."
        )
    except Exception as e:
        logger.error(e)
        raise Exception(
            "An error occured while loading the ensemble config. Please check the ensemble_config.json file."
        )

    return ensemble_config
