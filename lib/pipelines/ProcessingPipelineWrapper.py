from dataclasses import dataclass
from typing import List, Optional, Tuple, cast
from joblib import Memory
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ProcessingPipelineWrapper:
    """A class that is to help the creation of the processing pipeline."""

    memory: Optional[Memory] = None
    """A memory caching of pipeline."""
    pandas_output: bool = False
    """Whether or not the pipeline should output pandas DataFrame."""

    def _is_string_allowed(
        self, model: Optional[BaseEstimator], allow_strings: Optional[bool]
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

    def create_pipeline(
        self,
        model: Optional[BaseEstimator] = None,
        allow_strings: Optional[bool] = None,
        features_in: Optional[List[str]] = None,
    ) -> Pipeline:
        """A function that is to automate the process of processing the data so that it is ready to be trained on made the prediction.

        Args:
            allow_strings (bool, optional): Whether or not all the data should have float types. Meaning if the encoding of the categorical columns is crucial. Defaults to False.
            model (Optional[BaseEstimator], optional): A model that is supposed to make predictions. Defaults to None.

        Returns:
            Pipeline: A processing pipeline.
        """

        # --- NOTE ---
        # Impute mean for numerical columns and one hot encode for string columns.
        numerical_column_transformer = ColumnTransformer(
            transformers=[
                (
                    "numerical_pipeline",
                    make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
                    make_column_selector(dtype_include=np.number),  # type: ignore
                )
            ],
            remainder="drop",
        )

        # --- NOTE ---
        # If the model is purely numerical, then we should not allow string columns.
        # Or if the user wants to force encoding, then we should encode the string columns.

        string_pipeline_steps: List[BaseEstimator] = [
            SimpleImputer(strategy="constant", fill_value="dna"),
        ]
        if not self._is_string_allowed(model=model, allow_strings=allow_strings):
            string_pipeline_steps.append(
                OneHotEncoder(
                    drop="first",
                    handle_unknown="infrequent_if_exist",
                    sparse_output=self.pandas_output is False,
                )
            )

        string_column_transformer = ColumnTransformer(
            transformers=[
                (
                    "string_pipeline",
                    make_pipeline(*string_pipeline_steps),
                    make_column_selector(dtype_include=object),  # type: ignore
                )
            ],
            remainder="drop",
        )

        # --- NOTE ---
        # Create steps for the pipeline.
        # If model is not None, then add the model to the pipeline.

        steps: List[Tuple[str, BaseEstimator]] = []

        if features_in is not None:
            steps.append(
                (
                    "column_selector",
                    FunctionTransformer(lambda X: X[features_in], validate=False),
                )
            )

        steps.append(
            (
                "feature_union",
                FeatureUnion(
                    transformer_list=[
                        ("numerical", numerical_column_transformer),
                        ("string", string_column_transformer),
                    ]
                ),
            )
        )
        if model is not None:
            steps.append(("model", model))

        # --- NOTE ---
        # Create the pipeline with the steps and set the output to be pandas DataFrame if needed.
        pipeline = cast(
            Pipeline,
            Pipeline(steps=steps, memory=self.memory).set_output(
                transform="pandas" if self.pandas_output else "default"
            ),
        )

        return pipeline
