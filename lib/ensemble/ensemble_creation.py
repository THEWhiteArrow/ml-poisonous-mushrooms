import gc
from multiprocessing.managers import ListProxy
from pathlib import Path
import pickle
from typing import Callable, List, Tuple, TypedDict, cast
import multiprocessing as mp


import numpy as np
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


def extract_names(mp_names_np: np.ndarray) -> List[str]:
    names = []
    current_name = bytearray()  # Use bytearray to build each name

    for byte in mp_names_np:
        if byte == 0:  # Check for null terminator
            if current_name:  # If there's a name collected
                names.append(current_name.decode())  # Decode and add to the list
                current_name = bytearray()  # Reset for the next name
        else:
            current_name.append(byte)  # Add byte to the current name

    return names


def evaluate_combination(
    bitmap: int,
    predictions: np.ndarray,
    scores: np.ndarray,
    names: ListProxy,
    y_test: np.ndarray,
) -> Tuple[str, float]:
    try:

        combination_names: List[str] = [
            names[i] for i in range(len(names)) if bitmap & (1 << i)
        ]

        combination_predictions = predictions[
            (bitmap & (1 << np.arange(predictions.shape[0]))).astype(bool)
        ]

        combination_scores = scores[
            (bitmap & (1 << np.arange(scores.shape[0]))).astype(bool)
        ]

        combination_y_pred = EnsembleModel._combine_classification_predictions(
            predictions=[pd.Series(pred) for pred in combination_predictions],
            scores=[score for score in combination_scores],
            combination_names=combination_names,
        )

        combination_accuracy = (combination_y_pred == y_test).sum() / len(y_test)

        del combination_y_pred

        gc.collect()

        return "-".join(combination_names), combination_accuracy
    except Exception as e:
        logger.error(f"Error in evaluate_combination: {e}")
        raise e


def run_parralel_bitmap_processing(
    ensemble_model: EnsembleModel, y_test: pd.Series, fold: int, processes: int
) -> List[Tuple[str, int, float]]:

    results_list: List[Tuple[str, int, float]] = []
    if ensemble_model.predictions is None:
        raise ValueError("The ensemble model has not been predicted yet")

    # Number of models in the ensemble
    num_models = len(ensemble_model.predictions)
    n_samples = len(y_test)

    # --- Multiprocessing setup ---
    mp_predictions = mp.Array(
        "q", num_models * n_samples
    )  # Double type for predictions
    mp_predictions_np = np.frombuffer(mp_predictions.get_obj()).reshape(
        num_models, n_samples
    )
    # Ensure the shared array is filled with predictions
    for i in range(num_models):
        mp_predictions_np[i] = ensemble_model.predictions[i].to_numpy()

    # Setup for targets (use int64 type)
    mp_y_true = mp.Array("q", n_samples)  # Use 'q' for int64
    mp_y_true_np = np.frombuffer(mp_y_true.get_obj())
    mp_y_true_np[:] = y_test.to_numpy()  # Assign values to the shared array

    # Setup for scores
    mp_scores = mp.Array("d", len(ensemble_model.scores))
    mp_scores_np = np.frombuffer(mp_scores.get_obj())
    mp_scores_np[:] = ensemble_model.scores  # Assign values to the shared array

    # Setup for names
    mp_names_np = mp.Manager().list(ensemble_model.combination_names)

    # --- Multiprocessing ---
    with mp.Pool(processes=processes) as pool:
        tasks = [
            pool.apply_async(
                evaluate_combination,
                args=(
                    i,
                    mp_predictions_np,
                    mp_scores_np,
                    mp_names_np,
                    mp_y_true_np,
                ),
            )
            for i in range(1, 2**num_models)
        ]

    log_interval = max(1, 2 ** (num_models - 1) // 10)
    for i, task in enumerate(tasks):
        if i % log_interval == 0:
            logger.info(f"Processing combination {i} out of {len(tasks)}")

        combination_names, accuracy = task.get()

        results_list.append((combination_names, fold, accuracy))

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
            run_parralel_bitmap_processing(ensemble_model, y_test, cnt, processes)
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
