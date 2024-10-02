from typing import Optional, cast

import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.data_load.data_load import load_data
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features


def test_pipeline_processing_default_numeric_no_string_encoding():
    """
    Test to verify that if no model is passed then all numerical columns will have
    applied scaling and string columns will be attached without one hot encoding.
    """

    # --- DATA ---
    train, test = load_data()
    train = train.head(int(len(train) * 0.12))
    engineered_data = engineer_features(train).set_index("id")

    # --- ACT ---
    pipeline = ProcessingPipelineWrapper(pandas_output=True).create_pipeline(
        allow_strings=True
    )

    output = cast(pd.DataFrame, pipeline.fit_transform(engineered_data))

    # --- ASSERT ---
    assert len(output.columns) == len(engineered_data.columns)


def test_pipeline_processing_encodes_strings():
    """
    Test to verify that if no model is passed then all numerical columns will have
    applied scaling and string columns will be attached with one hot encoding.
    """

    # --- DATA ---
    engineered_data = pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "col3": [None, "aa", "aa", "bb", "bb"],
        }
    )
    # --- ACT ---
    pipeline = ProcessingPipelineWrapper(pandas_output=True).create_pipeline(
        allow_strings=False
    )

    output = cast(pd.DataFrame, pipeline.fit_transform(engineered_data))

    # --- ASSERT ---
    assert len(output.columns) == 1 + (5 - 1) + (3 - 1)


@pytest.mark.dev
@pytest.mark.parametrize(
    "model, expected",
    [
        (RidgeClassifier(), False),
        (KNeighborsClassifier(), False),
        (SVC(), False),
        (LGBMClassifier(), True),
        (RandomForestClassifier(), True),
        (None, False),
    ],
)
def test_allow_strings_for_correct_models(
    model: Optional[BaseEstimator], expected: bool
):
    assert (
        ProcessingPipelineWrapper()._is_string_allowed(model=model, allow_strings=None)
        == expected
    )
