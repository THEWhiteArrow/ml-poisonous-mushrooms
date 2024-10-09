import gc
from pathlib import Path
import pickle
from typing import Callable, List, Tuple, TypedDict, cast
import multiprocessing as mp


import pandas as pd
import psutil
from sklearn.model_selection import KFold

from lib.logger import setup_logger
from lib.models.EnsembleModel import EnsembleModel
from lib.models.HyperOptResultDict import HyperOptResultDict

logger = setup_logger(__name__)


class EnsembleFunctionDto(TypedDict):
    load_data_func: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]
    engineer_features_func: Callable[[pd.DataFrame], pd.DataFrame]


class EnsembleSetupDto(TypedDict):
    hyper_models_dir_path: Path
    ensemble_model_dir_path: Path
    selected_model_names: List[str]
    hyper_model_run: str
    n_cv: int
    id_column: List[str] | str
    limit_data_percentage: float
    processes: int


def log_system_usage(message=""):
    # Get memory info
    mem = psutil.virtual_memory()
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    logger.info(f"{message} | Memory used: {mem.percent}% | CPU used: {cpu_percent}%")


def create_ensemble_model(
    selected_model_names: List[str],
    hyper_models_dir_path: Path,
    limit_data_percentage: float,
    n_cv: int,
    processes: int,
    function_dto: EnsembleFunctionDto,
) -> Tuple[EnsembleModel, pd.DataFrame]:

    logger.info("Loading models from the config")

    hyper_opt_results: List[HyperOptResultDict] = []

    for model_name in selected_model_names:
        model_path = hyper_models_dir_path / f"{model_name}.pkl"

        model_data = cast(HyperOptResultDict, pickle.load(open(model_path, "rb")))
        model_data["name"] = model_name
        hyper_opt_results.append(model_data)

    logger.info("All models has been loaded")
    logger.info("Creating generic ensemble model")

    ensemble_model = EnsembleModel(
        models=[result["model"] for result in hyper_opt_results],
        combination_feature_lists=[result["features"] for result in hyper_opt_results],
        combination_names=[result["name"] for result in hyper_opt_results],
        scores=[result["score"] for result in hyper_opt_results],
    )

    logger.info("Loading data")
    train, test = function_dto["load_data_func"]()
    logger.info(f"Using {limit_data_percentage * 100}% data")
    train = train.head(int(len(train) * limit_data_percentage))
    logger.info("Engineering features")
    engineered_data = function_dto["engineer_features_func"](train).set_index("id")

    logger.info("Running optimization")
    final_ensemble_model, ensemble_result_df = optimize_ensemble(
        ensemble_model=ensemble_model,
        X=engineered_data,
        y=engineered_data["class"],
        n_cv=n_cv,
        processes=processes,
    )

    return final_ensemble_model, ensemble_result_df


def evaluate_combination(
    y_test: pd.Series,
    combination_names: List[str],
    scores: List[float],
    predictions: List[pd.Series],
) -> Tuple[str, float]:

    combination_names_string = "-".join(combination_names)

    temp_ensemble = EnsembleModel(
        models=[],
        combination_feature_lists=[],
        combination_names=combination_names,
        scores=scores,
        processing_pipelines=[],
        predictions=predictions,
    )

    y_pred = temp_ensemble._combine_classification_predictions()

    accuracy = (y_pred == y_test).sum() / len(y_test)

    del temp_ensemble
    del y_pred

    gc.collect()

    return combination_names_string, accuracy


def run_parralel_bitmap_processing(
    ensemble_model: EnsembleModel, y_test: pd.Series, processes: int
) -> List[Tuple[str, int, float]]:

    if ensemble_model.predictions is None:
        raise ValueError("The ensemble model has not been predicted yet")

    # Number of models in the ensemble
    num_models = len(ensemble_model.predictions)

    # Generate all possible combinations (2^num_models)
    num_combinations = range(1, 2**num_models)
    results_list: List[Tuple[str, int, float]] = []

    log_interval = 2 ** (num_models - 1) // 10

    for j in num_combinations:
        if j % log_interval == 0:
            logger.info(f"Processing combination {j}/{2**num_models}")

        y_pred = ensemble_model._combine_classification_predictions(j)

        combination_name = "-".join(
            [
                ensemble_model.combination_names[k]
                for k in range(num_models)
                if j & (1 << k)
            ]
        )
        y_accuracy = (y_pred == y_test).sum() / len(y_test)

        results_list.append((combination_name, j, y_accuracy))

    return results_list


def optimize_ensemble(
    ensemble_model: EnsembleModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_cv: int,
    processes: int,
) -> Tuple[EnsembleModel, pd.DataFrame]:
    logger.info("Optimizing ensemble model | " + "Optimizing with bitmap")

    processes = mp.cpu_count() if processes == -1 else processes
    kfold = KFold(n_splits=n_cv)
    results_list: List[Tuple[str, int, float]] = []

    for cnt, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        logger.info(f"Optimizing fold {cnt + 1}")

        log_system_usage("Before fitting ensemble model")

        ensemble_model.fit(X_train, y_train)
        if ensemble_model.processing_pipelines is None:
            raise ValueError("The ensemble model has not been fitted yet")

        ensemble_model.predict(X_test)
        if ensemble_model.predictions is None:
            raise ValueError("The ensemble model has not been predicted yet")

        logger.info(f"Evaluating combinations with {processes} processes")

        log_system_usage("Before multiprocessing starts")

        results_list.extend(
            run_parralel_bitmap_processing(ensemble_model, y_test, processes)
        )

        log_system_usage(f"Finished fold {cnt + 1}")

    log_system_usage("After optimization")
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
            ensemble_model.models[ensemble_model.combination_names.index(name)]
            for name in best_combination_names
        ],
        combination_feature_lists=[
            ensemble_model.combination_feature_lists[
                ensemble_model.combination_names.index(name)
            ]
            for name in best_combination_names
        ],
        combination_names=best_combination_names,
        scores=[
            ensemble_model.scores[ensemble_model.combination_names.index(name)]
            for name in best_combination_names
        ],
    )

    return best_ensemble, results_df.to_frame()
