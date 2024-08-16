from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, cast
from joblib import Memory
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class ProcessingPipelineWrapper:
    """A class that is to help the creation of the processing pipeline."""

    memory: Optional[Memory] = None
    """A memory caching of pipeline."""
    pandas_output: bool = False
    """Whether or not the pipeline should output pandas DataFrame."""

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
                            sparse_output=self.pandas_output is False,
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
            Pipeline(steps=steps, memory=self.memory).set_output(
                transform="pandas" if self.pandas_output else "default"
            ),
        )

        return pipeline
