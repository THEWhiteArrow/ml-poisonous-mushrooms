import json
from typing import Callable
from dataclasses import dataclass
from joblib import Memory
from logger import setup_logger
import pandas as pd
import optuna
import datetime as dt
from sklearn.model_selection import cross_val_score
from data_load import load_data
from engineering_features import engineer_features
from features import FeatureSet, FeatureManager
from models import HyperOptManager, ModelManager, HyperOptModelCombination
from pipelines import separate_column_types, ProcessingPipelineWrapper

logger = setup_logger()
N_OPTIMIZE_TRIALS: int = 25
N_CV: int = 5
START_ITERATION : int = 0


def create_objective(
    data: pd.DataFrame, model_combination: HyperOptModelCombination
) -> Callable[[optuna.Trial], float]:

    data = data.copy()

    X = data.drop(columns=["class"])
    y = data["class"]

    numerical_features, categorical_features = separate_column_types(X)

    processing_pipeline_wrapper = ProcessingPipelineWrapper(
        numerical_columns=numerical_features, categorical_columns=categorical_features, 
        memory=Memory(location=f"./cache/processing/{model_combination.name}", verbose=0),
    )
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
            estimator=pipeline, X=X, y=y, cv=N_CV, scoring="accuracy"
        )

        avg_acc = scores.mean()
        return avg_acc

    return objective


@dataclass
class EarlyStoppingCallback:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_value = None
        self.no_improvement_count = 0

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        # Get the current best value
        current_best_value = study.best_value

        # Check if the best value has improved
        if self.best_value is None or current_best_value > self.best_value:
            self.best_value = current_best_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Stop study if there has been no improvement for `self.patience` trials
        if self.no_improvement_count >= self.patience:
            study.stop()


def hyper_opt():
    logger.info("Starting main...")
    train, test = load_data()

    engineered_data = engineer_features(train.head(1500 * 1000)).set_index("id")
    # engineered_data = engineer_features(train).set_index("id")
    logger.info(f"Training data has {len(engineered_data)} rows.")
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
                name="veil", is_optional=True, features=["veil-type", "veil-color"]
            ),
            FeatureSet(
                name="ring", is_optional=True, features=["has-ring", "ring-type"]
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
    cv_output_df = pd.DataFrame(data={
        "combination": [], "accuracy": [], "params": []
    })

    for i, model_combination in enumerate(all_model_combinations):
        if i < START_ITERATION:
            continue

        logger.info(f"Training {i} iteration")

        early_stopping = EarlyStoppingCallback(patience=10)

        study = optuna.create_study(
            direction="maximize", study_name=f"optuna_{model_combination.name}"
        )

        study.optimize(
            func=create_objective(
                data=engineered_data[["class", *model_combination.feature_combination.features]],
                model_combination=model_combination,
            ),
            n_trials=N_OPTIMIZE_TRIALS,
            callbacks=[early_stopping],  # type: ignore
        )

        best_params = study.best_params
        best_score = study.best_value

        model_combination.score = best_score
        model_combination.hyper_parameters = best_params

        logger.info(
            f"Model combination: {model_combination.name} has scored: {best_score} with params: {best_params}"
        )

        cv_output_df = pd.concat(
            [
                cv_output_df,
                pd.DataFrame(data={
                    "combination": [model_combination.name],
                    "accuracy": [best_score],
                    "params": [json.dumps(best_params)]
                }),
            ],
            axis=0,
            ignore_index=True
        )

        # model_combination.pickle("./data/")
    cv_output_df.to_csv(f"cv_output_df_{dt.datetime.now().isoformat()}.csv")


def debug():
    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering data...")
    engineered_data = engineer_features(train.head(10000 * 1000)).set_index("id")

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
                name="veil", is_optional=True, features=["veil-type", "veil-color"]
            ),
            FeatureSet(
                name="ring", is_optional=True, features=["has-ring", "ring-type"]
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

    choosen_combination = all_model_combinations[13]
    logger.info(choosen_combination.name)

    X = engineered_data.copy()[choosen_combination.feature_combination.features]
    y = engineered_data.copy()["class"]

    numerical_features, categorical_features = separate_column_types(X)

    processing_pipeline_wrapper = ProcessingPipelineWrapper(
        numerical_columns=numerical_features, categorical_columns=categorical_features
    )
    model_wrapper = choosen_combination.model_wrapper
    model = model_wrapper.model

    allow_strings = model_wrapper.allow_strings

    logger.info(f"The model {model.__class__.__name__} {"allows strings" if allow_strings is True else "does NOT allow strings"}")
    pipeline = processing_pipeline_wrapper.create_pipeline(
        model=model, allow_strings=allow_strings
    )

    # processed_X = pipeline.fit_transform(X=X.copy())

    scores = cross_val_score(
        estimator=pipeline, X=X, y=y, cv=N_CV, scoring="accuracy"
    )

    logger.info(f"Accuracy scores: {scores}")
    logger.info(f"Avg: {scores.mean()}")

    # --- TODO ---
    # The data that is processed includes the imputer fields and scaled fields
    # (it should impute the fields and them scaled them without creating new columns)
    logger.info("Data has been processed")


if __name__ == "__main__":
    hyper_opt()
    # debug()
