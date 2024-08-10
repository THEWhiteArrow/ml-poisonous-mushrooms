from typing import Tuple
import pandas as pd


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("./data/train.csv")
    test = pd.read_csv("./data/test.csv")
    return train, test
