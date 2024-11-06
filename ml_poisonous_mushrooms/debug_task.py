# from copy import deepcopy
# from pathlib import Path
# import pickle
# from typing import List, cast
# import time
# import json
# import numpy as np

# import pandas as pd
# from sklearn.ensemble import StackingClassifier, VotingClassifier
# from sklearn.linear_model import RidgeClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.pipeline import Pipeline

# from ml_poisonous_mushrooms.lib.features.FeatureManager import FeatureManager
# from ml_poisonous_mushrooms.lib.features.FeatureSet import FeatureSet
# from ml_poisonous_mushrooms.lib.models.HyperOptManager import HyperOptManager
# from ml_poisonous_mushrooms.lib.models.HyperOptResultDict import HyperOptResultDict
# from ml_poisonous_mushrooms.lib.models.ModelManager import ModelManager
# from ml_poisonous_mushrooms.lib.pipelines.ProcessingPipelineWrapper import (
#     create_pipeline,
# )
# from ml_poisonous_mushrooms.data_load.data_load import load_data
# from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
# from ml_poisonous_mushrooms.utils.PathManager import PathManager

from ml_poisonous_mushrooms.lib.logger import setup_logger

logger = setup_logger(__name__)


# def debug_pipelines():
#     logger.info("Loading data...")
#     train, test = load_data()

#     logger.info("Engineering data...")
#     engineered_data = engineer_features(train.head(10 * 1000)).set_index("id")

#     feature_manager = FeatureManager(
#         feature_sets=[
#             FeatureSet(
#                 name="stem",
#                 is_optional=False,
#                 features=[
#                     "stem-height",
#                     "stem-width",
#                     "stem-root",
#                     "stem-surface",
#                     "stem-color",
#                 ],
#             ),
#             FeatureSet(
#                 name="cap",
#                 is_optional=True,
#                 features=["cap-diameter", "cap-shape", "cap-surface", "cap-color"],
#             ),
#             FeatureSet(
#                 name="gill",
#                 is_optional=True,
#                 features=["gill-spacing", "gill-attachment", "gill-color"],
#             ),
#             FeatureSet(
#                 name="veil", is_optional=True, features=["veil-type", "veil-color"]
#             ),
#             FeatureSet(
#                 name="ring", is_optional=True, features=["has-ring", "ring-type"]
#             ),
#             FeatureSet(
#                 name="other",
#                 is_optional=True,
#                 features=[
#                     "spore-print-color",
#                     "habitat",
#                     "season",
#                     "does-bruise-or-bleed",
#                 ],
#             ),
#         ]
#     )
#     model_manager = ModelManager(task="classification")
#     hyper_manager = HyperOptManager(
#         feature_manager=feature_manager,
#         models=model_manager.get_models(use_additional=[]),
#     )

#     all_model_combinations = hyper_manager.get_model_combinations()

#     choosen_combination = all_model_combinations[13]
#     logger.info(choosen_combination.name)

#     X = engineered_data.copy().drop(columns=["class"])
#     y = engineered_data.copy()["class"]

#     processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=True)
#     model = choosen_combination.model

#     pipeline0 = processing_pipeline_wrapper.create_pipeline()
#     pipeline1 = processing_pipeline_wrapper.create_pipeline(
#         features_in=choosen_combination.feature_combination.features
#     )
#     pipeline2 = processing_pipeline_wrapper.create_pipeline(
#         model=model, features_in=choosen_combination.feature_combination.features
#     )
#     pipeline3 = processing_pipeline_wrapper.create_pipeline(model=model)

#     processedX0 = pipeline0.fit_transform(
#         X[choosen_combination.feature_combination.features]
#     )
#     processedX1 = pipeline1.fit_transform(X)

#     scores2 = cross_val_score(
#         estimator=pipeline2, X=X.copy(), y=y, cv=5, scoring="accuracy"
#     )
#     scores3 = cross_val_score(
#         estimator=pipeline3, X=X.copy(), y=y, cv=5, scoring="accuracy"
#     )

#     logger.info(f"Accuracy scores: {scores2}")
#     logger.info(f"Avg: {scores2.mean()}")
#     logger.info(f"Accuracy scores: {scores3}")
#     logger.info(f"Avg: {scores3.mean()}")

#     # --- TODO ---
#     # The data that is processed includes the imputer fields and scaled fields
#     # (it should impute the fields and them scaled them without creating new columns)
#     logger.info("Data has been processed")


# def debug_ensamble():
#     unique_names = ["RandomForest", "SVM", "LogisticRegression"]
#     unique_targets = ["e", "f"]
#     scores = [0.5, 0.8, 0.3]

#     def fake_out(i):
#         fake_arr = []
#         for i in range(i):
#             fake_arr.append(unique_targets[np.random.randint(1, 10) % 2])

#         return fake_arr

#     predictions = pd.DataFrame(
#         index=range(10), data={name: fake_out(10) for name in unique_names}
#     )

#     scaled_scores = np.array(scores) / sum(scores)

#     votes = pd.DataFrame(index=range(10), columns=unique_targets)
#     votes.loc[:, :] = 0.0

#     for name in unique_names:
#         for target in unique_targets:
#             votes.loc[
#                 predictions.loc[predictions[name].eq(target)].index, target
#             ] += scaled_scores[unique_names.index(name)]

#     votes["decision"] = votes.apply(lambda row: row[unique_targets].idxmax(), axis=1)

#     print(votes)

#     return votes["decision"]


# def debug_enumerate():
#     class Test:
#         def __init__(self):
#             self.a = 1
#             self.b = 2

#         def __str__(self):
#             return f"{self.a} {self.b}"

#     arr = [Test(), Test(), Test()]
#     print([str(item) for item in arr])
#     for i, item in enumerate(arr):
#         item.a = i

#     print([str(item) for item in arr])


# def debug_xgboost():
#     logger.info("Loading data...")
#     train, test = load_data()

#     logger.info("Engineering data...")
#     engineered_data = engineer_features(train.head(1100 * 1000)).set_index("id")

#     feature_manager = FeatureManager(
#         feature_sets=[
#             FeatureSet(
#                 name="stem",
#                 is_optional=False,
#                 features=[
#                     "stem-height",
#                     "stem-width",
#                     "stem-root",
#                     "stem-surface",
#                     "stem-color",
#                 ],
#             ),
#             FeatureSet(
#                 name="cap",
#                 is_optional=True,
#                 features=["cap-diameter", "cap-shape", "cap-surface", "cap-color"],
#             ),
#             FeatureSet(
#                 name="gill",
#                 is_optional=True,
#                 features=["gill-spacing", "gill-attachment", "gill-color"],
#             ),
#             FeatureSet(
#                 name="veil", is_optional=True, features=["veil-type", "veil-color"]
#             ),
#             FeatureSet(
#                 name="ring", is_optional=True, features=["has-ring", "ring-type"]
#             ),
#             FeatureSet(
#                 name="other",
#                 is_optional=True,
#                 features=[
#                     "spore-print-color",
#                     "habitat",
#                     "season",
#                     "does-bruise-or-bleed",
#                 ],
#             ),
#         ]
#     )
#     model_manager = ModelManager(task="classification")
#     hyper_manager = HyperOptManager(
#         feature_manager=feature_manager,
#         models=model_manager.get_models(use_additional=[]),
#     )

#     all_model_combinations = hyper_manager.get_model_combinations()

#     choosen_combination = list(
#         filter(
#             lambda combination: "XGBClassifier" in combination.name,
#             all_model_combinations,
#         )
#     )[0]

#     logger.info(choosen_combination.name)
#     model = choosen_combination.model

#     X = engineered_data.copy()[choosen_combination.feature_combination.features]
#     y = engineered_data.copy()["class"]

#     processing_pipeline_wrapper = ProcessingPipelineWrapper(pandas_output=False)
#     pipeline = processing_pipeline_wrapper.create_pipeline(model=model)

#     acc = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")

#     logger.info(f"Accuracy: {acc.mean()}")


# def debug_cv():
#     logger.info("Loading data...")
#     train, test = load_data()

#     engineered_data = engineer_features(train.head(int(len(train) * 0.1))).set_index(
#         "id"
#     )

#     # model = RandomForestClassifier()
#     model = RidgeClassifier()

#     pipeline = ProcessingPipelineWrapper(pandas_output=False).create_pipeline(
#         model=model
#     )

#     X = engineered_data.copy().drop(columns=["class"])
#     y = engineered_data.copy()["class"]

#     # output = pipeline[:-1].fit_transform(X_train, y_train)
#     results = {}

#     score_methods = ["accuracy", "f1", "precision", "recall"]
#     for score_method in score_methods:
#         logger.info(f"Calculating {score_method}...")
#         results[score_method] = cross_val_score(
#             estimator=pipeline, X=X, y=y, cv=5, scoring=score_method
#         ).tolist()

#     logger.info("Results:")
#     for score_method in score_methods:
#         logger.info(f"{score_method}: {json.dumps(results[score_method])}")
#     logger.info("done")


# def measure_time() -> float:
#     # Generate two large random matrices
#     matrix_size = 15000  # Adjust size if it runs too fast or too slow
#     A = np.random.rand(matrix_size, matrix_size)
#     B = np.random.rand(matrix_size, matrix_size)
#     # Measure time to multiply the matrices
#     start_time = time.time()
#     # Perform matrix multiplication
#     _ = np.dot(A, B)
#     # Measure end time
#     end_time = time.time()
#     # Calculate time taken
#     time_taken = end_time - start_time
#     # print(f"Matrix multiplication took {time_taken:.2f} seconds")
#     return time_taken


# def measure_time_n_times(n: int) -> float:
#     times = []
#     for i in range(n):
#         times.append(measure_time())
#     return cast(float, np.mean(times))


# def debug_stacking() -> None:
#     train, test = load_data()
#     train = train.head(int(len(train) * 0.02))
#     data = engineer_features(train)
#     X = data.drop(columns=["class", "id"])
#     y = data["class"]

#     logger.info("CV stacking ensemble")
#     stacking = StackingClassifier(
#         estimators=[
#             (
#                 "ridge",
#                 create_pipeline(
#                     model=RidgeClassifier(),
#                     features_in=X.columns.to_list(),
#                     pandas_output=False,
#                 ),
#             ),
#         ],
#         final_estimator=create_pipeline(
#             model=KNeighborsClassifier(n_jobs=-1), enforce_input_dataframe=True
#         ),
#         passthrough=True,
#         verbose=1,
#     )

#     stacking_scores = cross_val_score(
#         stacking, X.copy(), y, cv=5, error_score="raise", verbose=2
#     )

#     logger.info(f"Stacking scores: {stacking_scores}")


# def ensemble_v2(
#     n_cv: int, hyper_models_dir_path: Path, selected_model_names: List[str]
# ) -> pd.DataFrame:

#     logger.info("Loading models from the config")
#     hyper_opt_results: List[HyperOptResultDict] = []
#     for model_name in selected_model_names:
#         model_path = hyper_models_dir_path / f"{model_name}.pkl"

#         model_data = cast(HyperOptResultDict, pickle.load(open(model_path, "rb")))
#         model_data["name"] = model_name
#         hyper_opt_results.append(model_data)

#     logger.info("Creating ensemble model")
#     model_names: List[str] = []
#     model_pipelines: List[Pipeline] = []
#     model_scores: List[float] = []
#     for model_data in hyper_opt_results:
#         pipeline = create_pipeline(
#             model=model_data["model"], features_in=model_data["features"]
#         )

#         model_names.append(model_data["name"].replace("__", "_"))
#         model_pipelines.append(pipeline)
#         model_scores.append(model_data["score"])

#     model_weights: List[float] = [score / sum(model_scores) for score in model_scores]

#     ensemble_voting_model = VotingClassifier(
#         estimators=[
#             (name, deepcopy(pipeline))
#             for name, pipeline in zip(model_names, model_pipelines)
#         ],
#         weights=model_weights,
#         voting="soft",
#         verbose=True,
#     )

#     ensemble_stacking_model = StackingClassifier(
#         estimators=[
#             (name, deepcopy(pipeline))
#             for name, pipeline in zip(model_names, model_pipelines)
#         ],
#         final_estimator=create_pipeline(model=LGBMClassifier(n_jobs=-1, verbose=-1), enforce_input_dataframe=True),  # type: ignore
#         passthrough=True,
#         verbose=1,
#     )

#     train, test = load_data()
#     train = train.head(int(len(train) * 0.02))
#     data = engineer_features(train).set_index("id")

#     X = data.drop(columns=["class"])
#     y = data["class"]

#     logger.info("CV stacking ensemble")
#     stacking_scores = cross_val_score(
#         ensemble_stacking_model, X.copy(), y, cv=n_cv, error_score="raise", verbose=2
#     )
#     logger.info(f"Stacking scores: {stacking_scores}")

#     logger.info("CV voting ensemble")
#     voting_scores = cross_val_score(
#         ensemble_voting_model, X, y, cv=n_cv, error_score="raise"
#     )
#     logger.info(f"Voting scores: {voting_scores}")

#     results = pd.DataFrame(
#         {
#             "model": ["voting", "stacking"],
#             "score": [np.mean(voting_scores), np.mean(stacking_scores)],
#         }
#     )

#     return results


# def predict_v2(
#     selected_model_names: List[str],
#     hyper_models_dir_path: Path = PathManager.OUTPUT_DIR_PATH.value / "hyper_opt_",
# ) -> None:
#     logger.info("Loading models from the config")
#     hyper_opt_results: List[HyperOptResultDict] = []
#     for model_name in selected_model_names:
#         model_path = hyper_models_dir_path / f"{model_name}.pkl"

#         model_data = cast(HyperOptResultDict, pickle.load(open(model_path, "rb")))
#         model_data["name"] = model_name
#         hyper_opt_results.append(model_data)

#     logger.info("Creating ensemble model")
#     model_names: List[str] = []
#     model_pipelines: List[Pipeline] = []
#     model_scores: List[float] = []
#     for model_data in hyper_opt_results:
#         pipeline = create_pipeline(
#             model=model_data["model"], features_in=model_data["features"]
#         )

#         model_names.append(model_data["name"].replace("__", "_"))
#         model_pipelines.append(pipeline)
#         model_scores.append(model_data["score"])

#     model_weights: List[float] = [score / sum(model_scores) for score in model_scores]

#     ensemble_voting_model = VotingClassifier(
#         estimators=[
#             (name, deepcopy(pipeline))
#             for name, pipeline in zip(model_names, model_pipelines)
#         ],
#         weights=model_weights,
#         voting="soft",
#         verbose=True,
#     )

#     ensemble_stacking_model = StackingClassifier(
#         estimators=[
#             (name, deepcopy(pipeline))
#             for name, pipeline in zip(model_names, model_pipelines)
#         ],
#         final_estimator=create_pipeline(model=LGBMClassifier(n_jobs=-1, verbose=-1), enforce_input_dataframe=True),  # type: ignore
#         passthrough=True,
#         verbose=1,
#     )

#     train, test = load_data()

#     engineered_data_train = engineer_features(train).set_index("id")
#     engineered_data_test = engineer_features(test).set_index("id")

#     X_train = engineered_data_train.drop(columns=["class"])
#     y_train = engineered_data_train["class"]
#     X_test = engineered_data_test

#     # y_pred_voting_raw = ensemble_voting_model.fit(X_train, y_train).predict(X_test)
#     # y_pred_voting_df = pd.DataFrame(
#     #     y_pred_voting_raw, index=X_test.index, columns=["class"]  # type: ignore
#     # )

#     # y_pred_voting_df.to_csv(ensemble_model_dir_path / "predictions_voting.csv")

#     y_pred_stacking_raw = ensemble_stacking_model.fit(X_train, y_train).predict(X_test)
#     y_pred_stacking_df = pd.DataFrame(
#         y_pred_stacking_raw, index=X_test.index, columns=["class"]  # type: ignore
#     )

#     y_pred_stacking_df.to_csv(ensemble_model_dir_path / "predictions_stacking.csv")


if __name__ == "__main__":
    logger.info("Starting debugging...")

    logger.info("Debugging done")
