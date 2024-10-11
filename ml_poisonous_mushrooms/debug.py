from typing import cast
import time
import json
import numpy as np

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

from lib.features.FeatureManager import FeatureManager
from lib.features.FeatureSet import FeatureSet
from lib.models.HyperOptManager import HyperOptManager
from lib.models.ModelManager import ModelManager
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.data_load.data_load import load_data
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from lib.logger import setup_logger

logger = setup_logger(__name__)


def debug_pipelines():
    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering data...")
    engineered_data = engineer_features(train.head(10 * 1000)).set_index("id")

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
        models=model_manager.get_models(use_additional=[]),
    )

    all_model_combinations = hyper_manager.get_model_combinations()

    choosen_combination = all_model_combinations[13]
    logger.info(choosen_combination.name)

    X = engineered_data.copy().drop(columns=["class"])
    y = engineered_data.copy()["class"]

    processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=True)
    model = choosen_combination.model

    pipeline0 = processing_pipeline_wrapper.create_pipeline()
    pipeline1 = processing_pipeline_wrapper.create_pipeline(
        features_in=choosen_combination.feature_combination.features
    )
    pipeline2 = processing_pipeline_wrapper.create_pipeline(
        model=model, features_in=choosen_combination.feature_combination.features
    )

    processedX0 = pipeline0.fit_transform(
        X[choosen_combination.feature_combination.features]
    )
    processedX1 = pipeline1.fit_transform(X)

    scores = cross_val_score(estimator=pipeline2, X=X, y=y, cv=5, scoring="accuracy")

    logger.info(f"Accuracy scores: {scores}")
    logger.info(f"Avg: {scores.mean()}")

    # --- TODO ---
    # The data that is processed includes the imputer fields and scaled fields
    # (it should impute the fields and them scaled them without creating new columns)
    logger.info("Data has been processed")


def debug_ensamble():
    unique_names = ["RandomForest", "SVM", "LogisticRegression"]
    unique_targets = ["e", "f"]
    scores = [0.5, 0.8, 0.3]

    def fake_out(i):
        fake_arr = []
        for i in range(i):
            fake_arr.append(unique_targets[np.random.randint(1, 10) % 2])

        return fake_arr

    predictions = pd.DataFrame(
        index=range(10), data={name: fake_out(10) for name in unique_names}
    )

    scaled_scores = np.array(scores) / sum(scores)

    votes = pd.DataFrame(index=range(10), columns=unique_targets)
    votes.loc[:, :] = 0.0

    for name in unique_names:
        for target in unique_targets:
            votes.loc[
                predictions.loc[predictions[name].eq(target)].index, target
            ] += scaled_scores[unique_names.index(name)]

    votes["decision"] = votes.apply(lambda row: row[unique_targets].idxmax(), axis=1)

    print(votes)

    return votes["decision"]


def debug_enumerate():
    class Test:
        def __init__(self):
            self.a = 1
            self.b = 2

        def __str__(self):
            return f"{self.a} {self.b}"

    arr = [Test(), Test(), Test()]
    print([str(item) for item in arr])
    for i, item in enumerate(arr):
        item.a = i

    print([str(item) for item in arr])


def debug_xgboost():
    logger.info("Loading data...")
    train, test = load_data()

    logger.info("Engineering data...")
    engineered_data = engineer_features(train.head(1100 * 1000)).set_index("id")

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
        models=model_manager.get_models(use_additional=[]),
    )

    all_model_combinations = hyper_manager.get_model_combinations()

    choosen_combination = list(
        filter(
            lambda combination: "XGBClassifier" in combination.name,
            all_model_combinations,
        )
    )[0]

    logger.info(choosen_combination.name)
    model = choosen_combination.model

    X = engineered_data.copy()[choosen_combination.feature_combination.features]
    y = engineered_data.copy()["class"]

    processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=False)
    pipeline = processing_pipeline_wrapper.create_pipeline(model=model)

    acc = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

    logger.info(f"Accuracy: {acc.mean()}")


def debug_cv():
    logger.info("Loading data...")
    train, test = load_data()

    engineered_data = engineer_features(train.head(int(len(train) * 0.1))).set_index(
        "id"
    )

    # model = RandomForestClassifier()
    model = RidgeClassifier()

    pipeline = ProcessingPipelineWrapper(pandas_output=False).create_pipeline(
        model=model
    )

    X = engineered_data.copy().drop(columns=["class"])
    y = engineered_data.copy()["class"]

    # output = pipeline[:-1].fit_transform(X_train, y_train)
    results = {}

    score_methods = ["accuracy", "f1", "precision", "recall"]
    for score_method in score_methods:
        logger.info(f"Calculating {score_method}...")
        results[score_method] = cross_val_score(
            estimator=pipeline, X=X, y=y, cv=5, scoring=score_method
        ).tolist()

    logger.info("Results:")
    for score_method in score_methods:
        logger.info(f"{score_method}: {json.dumps(results[score_method])}")
    logger.info("done")


def measure_time() -> float:
    # Generate two large random matrices
    matrix_size = 15000  # Adjust size if it runs too fast or too slow
    A = np.random.rand(matrix_size, matrix_size)
    B = np.random.rand(matrix_size, matrix_size)
    # Measure time to multiply the matrices
    start_time = time.time()
    # Perform matrix multiplication
    _ = np.dot(A, B)
    # Measure end time
    end_time = time.time()
    # Calculate time taken
    time_taken = end_time - start_time
    # print(f"Matrix multiplication took {time_taken:.2f} seconds")
    return time_taken


def measure_time_n_times(n: int) -> float:
    times = []
    for i in range(n):
        times.append(measure_time())
    return cast(float, np.mean(times))


if __name__ == "__main__":
    debug_pipelines()
    # debug_ensamble()
    # debug_enumerate()
    # debug_xgboost()
    # debug_cv()

    print(measure_time_n_times(5))
    logger.info("Debugging done")
