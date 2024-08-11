from typing import Callable
from logger import setup_logger
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from data_load import load_data
from engineering_features import engineer_features
from features import FeatureSet, FeatureManager
from models import HyperOptManager, ModelManager, HyperOptModelCombination
from pipelines import separate_column_types, ProcessingPipelineWrapper

logger = setup_logger()
N_OPTIMIZE_TRIALS: int = 10
N_CV: int = 5


def create_objective(
    data: pd.DataFrame, model_combination: HyperOptModelCombination
) -> Callable[[optuna.Trial], float]:

    data = data.copy().set_index("id")

    X = data.drop(columns=["class"])
    y = data["class"]

    numerical_features, categorical_features = separate_column_types(X)

    processing_pipeline_wrapper = ProcessingPipelineWrapper(
        numerical_columns=numerical_features, categorical_columns=categorical_features
    )
    model_wrapper = model_combination.model_wrapper
    model = model_wrapper.model

    def objective(
        trail: optuna.Trial,
    ) -> float:

        params = model_wrapper.get_params(trail)
        allow_strings = model_wrapper.allow_strings

        model.set_params(params=params)

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


def main():
    logger.info("Starting main...")
    train, test = load_data()

    engineered_data = engineer_features(train.head(10 * 1000))

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

    for model_combination in all_model_combinations:

        study = optuna.create_study(
            direction="minimize", study_name=f"optuna_{model_combination.name}"
        )

        study.optimize(
            func=create_objective(
                data=engineered_data[model_combination.feature_combination.features],
                model_combination=model_combination,
            ),
            n_trials=N_OPTIMIZE_TRIALS,
        )

        best_params = study.best_params
        best_score = study.best_value

        model_combination.score = best_score
        model_combination.hyper_parameters = best_params

        logger.info(
            f"Model combination: {model_combination.name} has scored: {best_score}"
        )
        model_combination.pickle("./data/")


if __name__ == "__main__":
    logger.info("Running main...")
    main()
