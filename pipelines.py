from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union, cast, Optional
from joblib import Memory
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline


def separate_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """A function that is to seperate the columns into numerical and categorical.

    Args:
        X (pd.DataFrame): Dataframe of interest.

    Returns:
        Tuple[List[str], List[str]]: A tuple that has the first element as list of numercial columns and second one as categorical columns list.
    """
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()
    return numerical_columns, categorical_columns


@dataclass
class ProcessingPipelineWrapper:
    """A class that is to help the creation of the processing pipeline."""

    numerical_columns: List[str]
    """Numerical columns of the data."""
    categorical_columns: List[str]
    """Categotical columns of the data."""
    memory: Optional[Memory] = None
    """A memory caching of pipeline."""

    def create_pipeline(
        self, allow_strings: bool = True, model: Optional[BaseEstimator] = None
    ) -> Pipeline:
        """A function that is to automate the process of processing the data so that it is ready to be trained on made the prediction.

        Args:
            force_numerical (bool, optional): Whether or not all the data should have float types. Meaning if the encoding of the categorical columns is crucial. Defaults to True.
            model (Optional[BaseEstimator], optional): A model that is supposed to make predictions. Defaults to None.

        Returns:
            Pipeline: A processing pipeline.
        """

        # transformers: List[Tuple[str, BaseEstimator, List[str]]] = [
        #     ("imputer", SimpleImputer(strategy="mean"), self.numerical_columns),
        #     ("scaler", StandardScaler(), self.numerical_columns),
        # ]

        # if allow_strings is False:
        #     transformers.append(
        #         (
        #             "encoder",
        #             OneHotEncoder(
        #                 drop="first",
        #                 handle_unknown="infrequent_if_exist",
        #                 sparse_output=False,
        #             ),
        #             self.categorical_columns,
        #         )
        #     )

        # ct = ColumnTransformer(transformers=transformers, remainder="drop")

        # steps: List[Tuple[str, BaseEstimator]] = [("transformer", ct)]

        transformer_list: Sequence[Tuple[str, TransformerMixin | Pipeline]] = []

        numerical_transformer = ColumnTransformer(
            transformers=[
                (
                    "numerical_pipeline",
                    make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()),
                    make_column_selector(dtype_include=np.number),  # type: ignore
                )
            ],
            remainder="drop",
        )

        transformer_list.append(("numerical_transformer", numerical_transformer))

        if allow_strings is False:
            string_transformer = ColumnTransformer(
                transformers=[
                    (
                        "encoder",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="infrequent_if_exist",
                            sparse_output=False,
                        ),
                        make_column_selector(dtype_include=object),  # type: ignore
                    )
                ],
                remainder="drop",
            )

            transformer_list.append(("string_transformer", string_transformer))

        steps: List[Tuple[str, BaseEstimator]] = [
            ("feature_union", FeatureUnion(transformer_list=transformer_list))
        ]

        if model is not None:
            steps.append(("model", model))

        pipeline = cast(
            Pipeline,
            Pipeline(steps=steps, memory=self.memory).set_output(transform="pandas"),
        )

        return pipeline


@dataclass
class PreprocessingPipelineWrapper:
    memory: Optional[Memory] = None
    """A memory caching of pipeline."""

    def create_preprocessing_pipeline(
        self, steps: List[Tuple[str, Union[TransformerMixin, BaseEstimator]]]
    ) -> Pipeline:
        """A function that creates a preprocessing pipeline from predefined steps.

        Args:
            steps (List[Tuple[str, Union[TransformerMixin, BaseEstimator]]]): A list of tuples in a form of the name of the step and a pipeline actuall pipeline step.

        Returns:
            Pipeline: A preprocessing pipeline.
        """
        pipeline = cast(
            Pipeline,
            Pipeline(
                steps=steps,
                memory=self.memory,
            ).set_output(transform="pandas"),
        )

        return pipeline
