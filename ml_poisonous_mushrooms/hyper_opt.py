from typing import Callable, Optional
import datetime as dt
import multiprocessing as mp

import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from lib.features.FeatureManager import FeatureManager
from lib.features.FeatureSet import FeatureSet
from lib.logger import setup_logger
from lib.models.HyperOptCombination import HyperOptCombination
from lib.models.HyperOptManager import HyperOptManager
from lib.models.ModelManager import ModelManager
from lib.optymization.TrialParamWrapper import TrialParamWrapper
from lib.optymization.parrarel_optimization import run_parallel_optimization
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.utils.data_load import load_data
from models.existing_models import get_existing_models
from models import MODELS_DIR_PATH

logger = setup_logger(__name__)


def create_objective(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    model_combination: HyperOptCombination,
    n_cv: int,
) -> Callable[[optuna.Trial], float]:

    processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=False)
    model_wrapper = model_combination.model_wrapper
    model = model_wrapper.model
    allow_strings = model_wrapper.allow_strings

    def objective(
        trial: optuna.Trial,
    ) -> float:

        params = TrialParamWrapper().get_params(
            model_name=model_combination.model_wrapper.model.__class__.__name__,
            trial=trial,
        )

        model.set_params(**params)

        pipeline = processing_pipeline_wrapper.create_pipeline(
            model=model,
            allow_strings=allow_strings,
        )

        scores = cross_val_score(
            estimator=pipeline, X=X, y=y, cv=n_cv, scoring="accuracy"
        )

        avg_acc = scores.mean()
        return avg_acc

    return objective


def hyper_opt(
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    model_run: Optional[str],
    processes: Optional[int] = None,
):
    if model_run is None:
        model_run = dt.datetime.now().strftime("%Y%m%d%H%M")

    logger.info(f"Starting hyper opt for run {model_run}...")

    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering features...")
    # --- NOTE ---
    # This could be a class that is injected with injector.
    engineered_data = engineer_features(train).set_index("id")

    logger.info(f"Training data has {len(engineered_data)} rows.")

    # --- NOTE ---
    # This could be a class that is injected with injector.
    feature_manager = FeatureManager(
        feature_sets=[
            FeatureSet(
                name="stem",
                is_optional=False,
                features=[
                    "stem-height",
                    "stem-width",
                    "stem-root",
                    "stem-surface",
                    "stem-color",
                ],
            ),
            FeatureSet(
                name="cap",
                is_optional=True,
                features=["cap-diameter", "cap-shape", "cap-surface", "cap-color"],
            ),
            FeatureSet(
                name="gill",
                is_optional=True,
                features=["gill-spacing", "gill-attachment", "gill-color"],
            ),
            FeatureSet(
                name="ring_and_veil",
                is_optional=True,
                features=["has-ring", "ring-type", "veil-type", "veil-color"],
            ),
            FeatureSet(
                name="other",
                is_optional=True,
                features=[
                    "spore-print-color",
                    "habitat",
                    "season",
                    "does-bruise-or-bleed",
                ],
            ),
        ]
    )

    model_manager = ModelManager(task="classification")

    hyper_manager = HyperOptManager(
        feature_manager=feature_manager,
        model_wrappers=model_manager.get_model_wrappers(use_sv=False),
    )

    all_model_combinations = hyper_manager.get_model_combinations()
    logger.info(f"Training {len(all_model_combinations)} combinations.")

    logger.info("Checking for existing models...")

    omit_names = get_existing_models(model_run)

    logger.info(f"Omitting {len(omit_names)} combinations.")

    logger.info("Starting parallel optimization...")
    _ = run_parallel_optimization(
        model_run=model_run,
        direction="maximize",
        all_model_combinations=all_model_combinations,
        X=engineered_data,
        y=engineered_data["class"],
        n_optimization_trials=n_optimization_trials,
        n_cv=n_cv,
        n_patience=n_patience,
        model_dir_path=MODELS_DIR_PATH,
        create_objective=create_objective,
        omit_names=omit_names,
        processes=processes,
    )

    logger.info("Models has been pickled and saved to disk.")


if __name__ == "__main__":
    logger.info("Starting hyper opt script...")
    hyper_opt(
        n_optimization_trials=25,
        n_cv=5,
        n_patience=7,
        model_run=None,
        processes=mp.cpu_count() // 2,
    )
    logger.info("Hyper opt script complete.")
