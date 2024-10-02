from typing import List, cast
import pandas as pd
from lib.logger import setup_logger

logger = setup_logger(__name__)


def clean_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """A function that cleans the categorical data in the dataset.
    It will remove the outliers (categories that are less than 1% of the total data) and replace them with "nan".
    Additionally it will fill "real" NaNs with "nan" as well.

    Args:
        X (pd.DataFrame): A dataframe to clean.
        columns (List[str]): Columns to clean.

    Returns:
        pd.DataFrame: A cleaned dataframe.
    """
    data = X.copy()
    categorical_outliers_frequency_limit = 0.01
    logger.info("Outliers frequency limit is" +
                f"{categorical_outliers_frequency_limit}")

    categorical_columns: List[str] = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    for column in categorical_columns:
        value_counts = data[column].value_counts().to_frame()
        sum_value_counts = value_counts["count"].sum()

        outliers = cast(
            pd.DataFrame,
            value_counts[
                value_counts["count"]
                < sum_value_counts * categorical_outliers_frequency_limit
            ],
        ).index.to_list()
        logger.info(f"Outliers for column '{column}' are {outliers}")

        data.loc[:, column] = (
            pd.Series(
                data[column].apply(
                    lambda el: el if el not in outliers else "dna")
            )
            .rename(column)
            .fillna("dna")
            .astype("category")
        )

    return data


def label_targets(X: pd.DataFrame) -> pd.DataFrame:
    """A function that labels the targets in the dataset.

    Args:
        X (pd.DataFrame): A dataframe to label.

    Returns:
        pd.DataFrame: A labeled dataframe.
    """
    data = X.copy()
    data["class"] = data["class"].map({"p": 1, "e": 0})

    return data


def unlabel_targets(X: pd.DataFrame) -> pd.DataFrame:
    """A function that unlabels the targets in the dataset.

    Args:
        X (pd.DataFrame): A dataframe to unlabel.

    Returns:
        pd.DataFrame: An unlabeled dataframe.
    """
    data = X.copy()
    data["class"] = data["class"].map({1: "p", 0: "e"})

    return data


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """A function that engineers the features.

    Args:
        data (pd.DataFrame): Before engineered data.

    Returns:
        pd.DataFrame: After engineered data.
    """
    data = data.copy()

    cleaned_cat_data = clean_categorical(X=data)
    cleaned_labeled_data = label_targets(X=cleaned_cat_data)
    return cleaned_labeled_data
