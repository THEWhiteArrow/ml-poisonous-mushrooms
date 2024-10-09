import gc
from typing import List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from lib.logger import setup_logger
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper

logger = setup_logger(__name__)


@dataclass
class EnsembleModel(BaseEstimator):
    models: List[BaseEstimator]
    combination_names: List[str]
    combination_feature_lists: List[List[str]]
    scores: List[float]
    processing_pipelines: Optional[List[Pipeline]] = None
    predictions: Optional[List[pd.Series]] = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> "EnsembleModel":
        logger.info("Fitting ensemble model")
        self.processing_pipelines = []

        for i, model in enumerate(self.models):
            logger.info(f"Fitting model : {self.combination_names[i]}")
            pipeline = ProcessingPipelineWrapper().create_pipeline(model=model)

            pipeline.fit(X[self.combination_feature_lists[i]], y)

            self.processing_pipelines.append(pipeline)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        logger.info("Predicting with ensemble model")
        self.predictions = []

        if self.processing_pipelines is None or len(self.processing_pipelines) != len(
            self.models
        ):
            raise ValueError(
                "The ensemble model either has not been fitted or is wrongly fitted"
            )

        for i, model in enumerate(self.models):
            logger.info(f"Predicting with : {self.combination_names[i]}")

            raw_prediction = self.processing_pipelines[i].predict(
                X[self.combination_feature_lists[i]]
            )

            prediction = pd.Series(
                raw_prediction, index=X.index, name=self.combination_names[i]
            )

            self.predictions.append(prediction)

        combined_prediction = self._combine_classification_predictions()

        return combined_prediction

    def _combine_classification_predictions(
        self, bitmap: Optional[int] = None
    ) -> pd.Series:
        """A function that combines the classification predictions of multiple models into a single prediction based on the scores of the models in the ensemble.
        It takes into account how much of the total score each model has and assigns the final prediction based on that.

        Args:
            data (pd.DataFrame): A DataFrame with the predictions of each model in the ensemble

        Returns:
            pd.Series: A Series with the final prediction
        """
        if self.predictions is None:
            raise ValueError("The ensemble model has not been predicted yet")

        if bitmap is not None:
            data = pd.concat(
                [
                    self.predictions[i]
                    for i in range(len(self.predictions))
                    if bitmap & (1 << i)
                ],
                axis=1,
            )
        else:
            data = pd.concat(self.predictions, axis=1)

        unique_targets = pd.unique(data.values.ravel())

        scaled_scores = np.array(self.scores) / np.sum(self.scores)

        votes = np.zeros((data.shape[0], len(unique_targets)), dtype=np.float32)

        target_to_idx = {target: idx for idx, target in enumerate(unique_targets)}

        for model_idx, model_name in enumerate(self.combination_names):

            if bitmap is not None and not bitmap & (1 << model_idx):
                continue

            model_predictions = data[model_name]

            for target_value, target_idx in target_to_idx.items():
                votes[:, target_idx] += np.where(
                    model_predictions == target_value, scaled_scores[model_idx], 0
                )

        final_vote_idx = np.argmax(votes, axis=1)
        final_vote = pd.Series(unique_targets[final_vote_idx], index=data.index)

        del data
        del votes
        gc.collect()

        return final_vote
