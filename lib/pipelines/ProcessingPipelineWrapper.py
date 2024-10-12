from typing import List, Optional, Tuple, cast
from joblib import Memory

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder

from lib.logger import setup_logger

logger = setup_logger(__name__)


def are_strings_allowed(
    model: Optional[BaseEstimator], allow_strings: Optional[bool]
) -> bool:
    """A function that checks if the model is allowed to have string columns.

    Args:
        model (Optional[BaseEstimator]): A model that is supposed to make predictions.
        allow_strings (bool): Whether or not all the data should have float types.

    Returns:
        bool: Whether or not the model is allowed to have string columns. Defaults to False if the model is None and allow_strings is None.
    """

    purely_numerical_model_names: List[str] = [
        "ridge",
        "kneighbors",
        "svc",
        "randomforest",
        "lgb",
        "xgb",
        "catboost",
    ]

    # --- NOTE ---
    # LGBoost, XGBoost, and CatBoost cold handle string columns but the setup would have to be customized for that.

    if model is not None:
        model_only_numerical = any(
            numerical_model_name.lower() in model.__class__.__name__.lower()
            for numerical_model_name in purely_numerical_model_names
        )

        if model_only_numerical and allow_strings is True:
            raise ValueError(
                "Model is purely numerical and does not allow string columns, but allow_strings is True."
            )

        return not model_only_numerical

    elif allow_strings is not None:
        return allow_strings
    else:
        return False


def convert_to_dataframe(X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
    """A function that converts the input to a pandas DataFrame.

    Args:
        X (pd.DataFrame | np.ndarray): The input data.

    Returns:
        pd.DataFrame: The input data as a pandas DataFrame.
    """

    if isinstance(X, pd.DataFrame):
        return X

    else:
        # Convert the input to a pandas DataFrame including proper type conversion.
        new_X = pd.DataFrame(X)
        logger.info(new_X.dtypes)
        for column in new_X.columns:
            try:
                new_X[column] = pd.to_numeric(new_X[column])
            except ValueError:
                logger.info(f"Could not convert column {column} to numeric.")
                logger.info(new_X[column].value_counts())

        return new_X


def create_pipeline(
    model: Optional[BaseEstimator] = None,
    features_in: Optional[List[str]] = None,
    memory: Optional[Memory | str] = None,
    enforce_input_dataframe: bool = False,
    allow_strings: Optional[bool] = None,
    pandas_output: bool = False,
) -> Pipeline:
    """A function that is to automate the process of processing the data so that it is ready to be trained on made the prediction.

    Args:
        allow_strings (bool, optional): Whether or not all the data should have float types. Meaning if the encoding of the categorical columns is crucial. Defaults to False.
        model (Optional[BaseEstimator], optional): A model that is supposed to make predictions. Defaults to None.

    Returns:
        Pipeline: A processing pipeline.
    """

    main_pipeline_steps: List[Tuple[str, BaseEstimator]] = []

    if enforce_input_dataframe and features_in is not None:
        raise Exception(
            "Cannot enforce input dataframe and use features_in at the same time -> name of the columns will be lost"
        )

    if enforce_input_dataframe:
        main_pipeline_steps.append(
            (
                "array_to_dataframe",
                FunctionTransformer(
                    convert_to_dataframe,
                    validate=False,
                ),
            )
        )

    string_pipeline_steps: List[TransformerMixin] = [
        SimpleImputer(strategy="constant", fill_value="dna"),
    ]

    if not are_strings_allowed(model, allow_strings):

        string_pipeline_steps.append(
            OneHotEncoder(
                drop="first",
                handle_unknown="infrequent_if_exist",
                sparse_output=pandas_output is False,
            )
        )

    main_pipeline_steps += [
        (
            "column_selector",
            FunctionTransformer(
                lambda X: X if features_in is None else X[features_in],
                validate=False,
            ),
        ),
        (
            "feature_union",
            FeatureUnion(
                transformer_list=[
                    (
                        "numerical",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "numerical_pipeline",
                                    make_pipeline(
                                        SimpleImputer(strategy="mean"),
                                        StandardScaler(),
                                    ),
                                    make_column_selector(dtype_include=np.number),  # type: ignore
                                ),
                            ],
                            remainder="drop",
                        ),
                    ),
                    (
                        "string",
                        ColumnTransformer(
                            transformers=[
                                (
                                    "string_pipeline",
                                    make_pipeline(*string_pipeline_steps),
                                    make_column_selector(dtype_include=object),  # type: ignore
                                ),
                            ],
                            remainder="drop",
                        ),
                    ),
                ]
            ),
        ),
    ]

    if model is not None:
        main_pipeline_steps.append(("model", model))

    # --- NOTE ---
    # Create the pipeline with the steps and set the output to be pandas DataFrame if needed.
    pipeline = cast(
        Pipeline,
        Pipeline(steps=main_pipeline_steps, memory=memory).set_output(
            transform="pandas" if pandas_output else "default"
        ),
    )

    return pipeline
