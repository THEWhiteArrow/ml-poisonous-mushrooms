import os
import signal
from typing import Callable, List, Optional
import multiprocessing as mp
import pandas as pd
import optuna
import datetime as dt
from sklearn.model_selection import cross_val_score
from ml_poisonous_mushrooms.data.data_load import load_data
from ml_poisonous_mushrooms.logger import setup_logger
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.utils.features import FeatureManager, FeatureSet
from ml_poisonous_mushrooms.utils.models import (
    HyperOptManager,
    HyperOptModelCombination,
    ModelManager,
    HyperOptResult
)
from ml_poisonous_mushrooms.utils.pipelines import (
    EarlyStoppingCallback,
    ProcessingPipelineWrapper,
)


logger = setup_logger(__name__)


def create_objective(
    data: pd.DataFrame, model_combination: HyperOptModelCombination, n_cv: int
) -> Callable[[optuna.Trial], float]:

    data = data.copy()

    X = data.drop(columns=["class"])
    y = data["class"]

    processing_pipeline_wrapper = ProcessingPipelineWrapper(
        pandas_output=False)
    model_wrapper = model_combination.model_wrapper
    model = model_wrapper.model
    allow_strings = model_wrapper.allow_strings

    def objective(
        trail: optuna.Trial,
    ) -> float:

        params = model_wrapper.get_params(trail)

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


def get_existing_models(run_str: str) -> List[str]:
    existing_models = []

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_run_dir_path = os.path.join(dir_path, "models", f"hyper_opt_{run_str}")

    if not os.path.exists(model_run_dir_path):
        return existing_models

    for file in os.listdir(model_run_dir_path):
        if file.endswith(".pkl"):
            existing_models.append(file.split(".")[0])

    return existing_models


def optimize_model(
    model_run: str,
    model_combination: HyperOptModelCombination,
    data: pd.DataFrame,
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    i: int,
) -> None:
    logger.info(f"Optimizing model combination {i}: {model_combination.name}")
    early_stopping = EarlyStoppingCallback(
        name=model_combination.name, patience=n_patience
    )

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(
        direction="maximize", study_name=f"optuna_{model_combination.name}"
    )

    study.optimize(
        func=create_objective(
            data=data[["class", *model_combination.feature_combination.features]],
            model_combination=model_combination,
            n_cv=n_cv,
        ),
        n_trials=n_optimization_trials,
        callbacks=[early_stopping],  # type: ignore
    )

    # Save the best parameters and score
    best_params = study.best_params
    best_score = study.best_value

    logger.info(
        f"Model combination: {model_combination.name} has scored: {
            best_score} with params: {best_params}"
    )

    result = HyperOptResult(
        name=model_combination.name,
        score=best_score,
        params=best_params,
        model=model_combination.model_wrapper.model,
        features=model_combination.feature_combination.features,
    )

    result.pickle(
        path=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "models",
            f"hyper_opt_{model_run}",
            f"{model_combination.name}.pkl",
        )
    )


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def parallel_optimization(
    all_model_combinations: List[HyperOptModelCombination],
    engineered_data: pd.DataFrame,
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    model_run: str,
    omit_names: List[str] = [],
):
    # Set up multiprocessing pool
    with mp.Pool(
        processes=mp.cpu_count() // 2, initializer=init_worker
    ) as pool:
        # Map each iteration of the loop to a process
        _ = pool.starmap(
            optimize_model,
            [
                (
                    model_run,
                    model_combination,
                    engineered_data,
                    n_optimization_trials,
                    n_cv,
                    n_patience,
                    i,
                )
                for i, model_combination in enumerate(all_model_combinations)
                if model_combination.name not in omit_names
            ],
        )


def hyper_opt(
    n_optimization_trials: int,
    n_cv: int,
    n_patience: int,
    model_run: Optional[str],
):
    if model_run is None:
        model_run = dt.datetime.now().strftime("%Y%m%d%H%M")

    logger.info(f"Starting hyper opt for run {model_run}...")

    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering features...")
    # --- NOTE ---
    # This could be a class that is injected with injector.
    engineered_data = engineer_features(train.head(10 * 1000)).set_index("id")

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
                features=["cap-diameter", "cap-shape",
                          "cap-surface", "cap-color"],
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
        model_wrappers=model_manager.get_model_wrappers(),
    )

    all_model_combinations = hyper_manager.get_model_combinations()
    logger.info(f"Training {len(all_model_combinations)} combinations.")

    logger.info("Checking for existing models...")

    omit_names = get_existing_models(model_run)

    logger.info(f"Omitting {len(omit_names)} combinations.")

    logger.info("Starting parallel optimization...")
    _ = parallel_optimization(
        all_model_combinations=all_model_combinations,
        engineered_data=engineered_data,
        n_optimization_trials=n_optimization_trials,
        n_cv=n_cv,
        n_patience=n_patience,
        model_run=model_run,
        omit_names=omit_names,
    )

    logger.info("Models has been pickled and saved to disk.")


if __name__ == "__main__":
    logger.info("Starting hyper opt script...")
    hyper_opt(
        n_optimization_trials=25,
        n_cv=5,
        n_patience=7,
        model_run=None,
    )
    logger.info("Hyper opt script complete.")
