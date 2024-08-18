from typing import List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from lib.logger import setup_logger
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper

logger = setup_logger(__name__)


@dataclass
class EnsembleModel(BaseEstimator):
    models: List[BaseEstimator]
    combination_names: List[str]
    combination_feature_lists: List[List[str]]
    scores: List[float]

    # --- NOTE IMPORTANT ---
    # Data needs to be processed both for fitting and trasforming

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series) -> "EnsembleModel":
        for i, model in enumerate(self.models):
            logger.info(f"Fitting model : {self.combination_names[i]}")
            processing_pipeline = ProcessingPipelineWrapper().create_pipeline()

            model.fit(processing_pipeline.fit_transform(X[self.combination_feature_lists[i]]), y)  # type: ignore

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:

        predictions: pd.DataFrame = pd.DataFrame(index=X.index)

        for i, model in enumerate(self.models):
            logger.info(f"Predicting with : {self.combination_names[i]}")
            processing_pipeline = ProcessingPipelineWrapper().create_pipeline()

            prediction_series = pd.Series(index=X.index, data=model.predict(processing_pipeline.fit_transform(X[self.combination_feature_lists[i]])))  # type: ignore

            predictions = predictions.join(prediction_series, how="left")

        combined_prediction = self._combine_classification_predictions(predictions)

        return combined_prediction

    def _combine_classification_predictions(self, data: pd.DataFrame) -> pd.Series:
        """A function that combines the classification predictions of multiple models into a single prediction based on the scores of the models in the ensemble.
        It takes into account how much of the total score each model has and assigns the final prediction based on that.

        Args:
            data (pd.DataFrame): A DataFrame with the predictions of each model in the ensemble

        Returns:
            pd.Series: A Series with the final prediction
        """
        unique_targets = list(set(data.values.flatten()))

        scaled_scores = np.array(self.scores) / sum(self.scores)

        votes = pd.DataFrame(index=data.index, columns=unique_targets)
        votes.loc[:, :] = 0.0

        for name in self.combination_names:
            for target in unique_targets:
                votes.loc[
                    data.loc[data[name].eq(target)].index, target
                ] += scaled_scores[self.combination_names.index(name)]

        final_vote = votes.idxmax(axis=1)
        return final_vote
