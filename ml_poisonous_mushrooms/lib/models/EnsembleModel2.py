import gc
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ml_poisonous_mushrooms.lib.logger import setup_logger
from ml_poisonous_mushrooms.lib.pipelines.ProcessingPipelineWrapper import (
    create_pipeline,
)

logger = setup_logger(__name__)


class EnsembleModel2(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        models: List[BaseEstimator],
        combination_features: List[List[str]],
        combination_names: List[str],
        task: Literal["classification", "regression"],
        weights: Optional[List[float]] = None,
        meta_model: Optional[BaseEstimator] = None,
        prediction_method: Literal["predict", "predict_proba"] = "predict",
        prediction_proba_target: Optional[str] = None,
    ) -> None:
        self.estimators: List[Pipeline] = []
        for i, model in enumerate(models):
            pipeline = create_pipeline(
                model=model,
                features_in=combination_features[i],
                enforce_input_dataframe=False,
                allow_strings=None,
            )
            self.estimators.append(pipeline)

        self.combination_names = combination_names
        if meta_model is not None:
            self.meta_model = create_pipeline(model=meta_model)
        else:
            self.meta_model = None

        self.prediction_method = prediction_method
        self.prediction_proba_target = prediction_proba_target
        self.predictions: List[pd.Series | pd.DataFrame] = []
        self.task = task
        if weights is None:
            self.weights = [1 / len(self.estimators)] * len(self.estimators)
        else:
            self.weights = weights

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EnsembleModel2":
        if self.meta_model is None:
            logger.info("Meta model is not present...")
            for i, estimator in enumerate(self.estimators):
                logger.info(f"Fitting model : {self.combination_names[i]}")
                estimator.fit(X, y)

        else:
            logger.info("Meta model is present...")

            kfold = KFold(n_splits=5, shuffle=True, random_state=1000000007)

            all_predictions_out_of_fold = pd.DataFrame(
                columns=self.combination_names, index=X.index
            )
            logger.info("Fitting models and making predictions on out of fold data...")
            for i, (train_idx, test_idx) in enumerate(kfold.split(X)):
                logger.info(f"Fold {i + 1}")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, _ = y.iloc[train_idx], y.iloc[test_idx]

                for j, estimator in enumerate(self.estimators):
                    logger.info(f"Fitting model : {self.combination_names[j]}")
                    estimator.fit(X_train, y_train)

                    logger.info(f"Predicting model : {self.combination_names[j]}")
                    all_predictions_out_of_fold.loc[
                        X_test.index, self.combination_names[j]
                    ] = pd.Series(estimator.predict(X_test), index=X_test.index)

            logger.info(
                "Completed fitting all models and making predictions on out of fold data"
            )
            logger.info("Fitting meta model...")
            self.meta_model.fit(pd.concat([X, all_predictions_out_of_fold], axis=1), y)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:

        if self.prediction_method == "predict":
            self.predictions = [
                pd.Series(estimator.predict(X), name=name, index=X.index)
                for estimator, name in zip(self.estimators, self.combination_names)
            ]
        elif self.prediction_method == "predict_proba":
            self.predictions = [
                pd.DataFrame(
                    estimator.predict_proba(X),
                    columns=[f"{class_name}" for class_name in estimator.classes_],
                    index=X.index,
                )
                for estimator, _ in zip(self.estimators, self.combination_names)
            ]
        else:
            raise ValueError(
                f"Prediction method {self.prediction_method} not recognized. Please use appropriate 'predict' or 'predict_proba'"
            )

        if self.task == "classification":

            if self.meta_model is not None:
                meta_X = pd.concat([X, *self.predictions], axis=1)
                final_pred_raw = self.meta_model.predict(meta_X)

                final_pred = pd.Series(
                    final_pred_raw, index=meta_X.index, name="prediction"
                )

            else:
                final_pred = self._combine_weighted_voting(
                    predictions=self.predictions,
                    weights=self.weights,
                    prediction_proba_target=self.prediction_proba_target,
                )

        else:
            raise ValueError("Only classification task is supported")

        return final_pred

    @classmethod
    def _combine_weighted_voting(
        cls,
        predictions: List[pd.Series | pd.DataFrame],
        weights: List[float],
        prediction_proba_target: Optional[str] = None,
    ) -> pd.Series:

        data = pd.concat(
            [
                (
                    prediction.rename("prediction")
                    .to_frame()
                    .assign(probability=weights[i])
                    .pivot(columns="prediction", values="probability")
                    .fillna(0.0)
                    if isinstance(prediction, pd.Series)
                    else prediction * weights[i]
                )
                for i, prediction in enumerate(predictions)
            ],
            axis=1,
        )

        unique_targets = data.columns.unique().to_list()
        votes = pd.DataFrame(
            np.zeros((data.shape[0], len(unique_targets)), dtype=np.float32),
            columns=unique_targets,
            index=data.index,
        )

        for target in unique_targets:
            votes[target] = data[[target]].sum(axis=1)

        if prediction_proba_target is not None:
            final_vote = votes[prediction_proba_target]
        else:
            final_vote = votes.idxmax(axis=1).astype(int)

        final_vote.name = "prediction"

        del data
        del votes
        gc.collect()

        return final_vote
