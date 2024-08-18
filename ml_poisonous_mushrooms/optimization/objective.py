from typing import Callable
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from lib.models.HyperOptCombination import HyperOptCombination
from lib.optymization.TrialParamWrapper import TrialParamWrapper
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper


def create_objective(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    model_combination: HyperOptCombination,
    n_cv: int,
) -> Callable[[optuna.Trial], float]:

    processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=False)
    model = model_combination.model

    def objective(
        trial: optuna.Trial,
    ) -> float:

        params = TrialParamWrapper().get_params(
            model_name=model_combination.model.__class__.__name__,
            trial=trial,
        )

        model.set_params(**params)

        pipeline = processing_pipeline_wrapper.create_pipeline(model=model)

        scores = cross_val_score(
            estimator=pipeline, X=X, y=y, cv=n_cv, scoring="accuracy"
        )

        avg_acc = scores.mean()
        return avg_acc

    return objective