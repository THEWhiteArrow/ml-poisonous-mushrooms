from typing import List, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
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

            self.predictions.append(prediction.copy())

        combined_prediction = self._combine_classification_predictions()

        return combined_prediction

    def _combine_classification_predictions(self) -> pd.Series:
        """A function that combines the classification predictions of multiple models into a single prediction based on the scores of the models in the ensemble.
        It takes into account how much of the total score each model has and assigns the final prediction based on that.

        Args:
            data (pd.DataFrame): A DataFrame with the predictions of each model in the ensemble

        Returns:
            pd.Series: A Series with the final prediction
        """
        if self.predictions is None:
            raise ValueError("The ensemble model has not been predicted yet")
        data = pd.DataFrame(index=self.predictions[0].index)

        for i, prediction in enumerate(self.predictions):
            data = data.join(prediction, how="left")

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

    def _evaluate_combination(self, j: int, y_test: pd.Series) -> Tuple[str, float]:

        if self.processing_pipelines is None:
            raise ValueError("The ensemble model has not been fitted yet")

        if self.predictions is None:
            raise ValueError("The ensemble model has not been predicted yet")

        temp_combination_names = [
            self.combination_names[k] for k in range(len(self.models)) if j & (1 << k)
        ]
        temp_combination_names_string = "-".join(temp_combination_names)

        temp_ensemble = EnsembleModel(
            models=[self.models[k] for k in range(len(self.models)) if j & (1 << k)],
            combination_feature_lists=[
                self.combination_feature_lists[k]
                for k in range(len(self.models))
                if j & (1 << k)
            ],
            combination_names=temp_combination_names,
            scores=[self.scores[k] for k in range(len(self.models)) if j & (1 << k)],
            processing_pipelines=[
                self.processing_pipelines[k]
                for k in range(len(self.models))
                if j & (1 << k)
            ],
            predictions=[
                self.predictions[k] for k in range(len(self.models)) if j & (1 << k)
            ],
        )

        y_pred = temp_ensemble._combine_classification_predictions()

        temp_accuracy = (y_pred == y_test).sum() / len(y_test)

        return temp_combination_names_string, temp_accuracy

    def optimize(
        self,
        n_cv: int,
        X: pd.DataFrame,
        y: pd.Series,
        processes: int = -1,
    ) -> Tuple["EnsembleModel", pd.DataFrame]:
        logger.info("Optimizing ensemble model | " + "Optimizing with bitmap")

        processes = mp.cpu_count() if processes == -1 else processes
        kfold = KFold(n_splits=n_cv)

        results_list: List[Tuple[str, int, float]] = []

        for i, (train_index, test_index) in enumerate(kfold.split(X)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            logger.info(f"Optimizing fold {i + 1}")

            self.fit(X_train, y_train)
            if self.processing_pipelines is None:
                raise ValueError("The ensemble model has not been fitted yet")

            self.predict(X_test)
            if self.predictions is None:
                raise ValueError("The ensemble model has not been predicted yet")

            with mp.Pool(processes) as pool:
                tasks = [
                    pool.apply_async(
                        self._evaluate_combination,
                        args=(j, y_test),
                    )
                    for j in range(1, 2 ** len(self.models))
                ]

                for k, task in enumerate(tasks, start=1):
                    log_interval = max(1, (2 ** len(self.models)) // 10)
                    if k % log_interval == 0:
                        logger.info(f"Checking combination {k}/{2 ** len(self.models)}")

                    temp_combination_names_string, temp_accuracy = task.get()

                    results_list.append(
                        (temp_combination_names_string, i, temp_accuracy)
                    )

        results_df = pd.DataFrame(
            results_list,
            columns=["combination", "fold", "score"],
        )

        results_df = results_df.groupby("combination")["score"].mean()
        logger.info(f"Optimization results: {results_df}")

        best_combination = str(results_df.idxmax())

        best_combination_names = best_combination.split("-")

        best_ensemble = EnsembleModel(
            models=[
                self.models[self.combination_names.index(name)]
                for name in best_combination_names
            ],
            combination_feature_lists=[
                self.combination_feature_lists[self.combination_names.index(name)]
                for name in best_combination_names
            ],
            combination_names=best_combination_names,
            scores=[
                self.scores[self.combination_names.index(name)]
                for name in best_combination_names
            ],
        )

        return best_ensemble, results_df.to_frame()
