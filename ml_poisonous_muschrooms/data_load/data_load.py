from typing import Tuple
import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A function that is to load the data.

    Raises:
        FileNotFoundError: If the data is not found in the data folder

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The train and test data.

    """

    try:
        train = pd.read_csv("./data/train.csv")
        test = pd.read_csv("./data/test.csv")

    except FileNotFoundError:
        raise FileNotFoundError(
            "Data not found. Please download the data from: https://www.kaggle.com/competitions/playground-series-s4e8/data and put it in the data folder."
        )

    return train, test
