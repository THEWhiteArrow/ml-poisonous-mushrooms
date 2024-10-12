import gc
from multiprocessing.managers import ListProxy
import multiprocessing as mp
from typing import List, Tuple

import psutil
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

from lib.models.EnsembleModel2 import EnsembleModel2
from lib.logger import setup_logger

logger = setup_logger(__name__)


def log_system_usage(message=""):
    # Get memory info
    mem = psutil.virtual_memory()
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)

    logger.info(f"{message} | Memory used: {mem.percent}% | CPU used: {cpu_percent}%")


def evaluate_combination(
    bitmap: int,
    predictions: np.ndarray,
    weights: np.ndarray,
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

        combination_weights = weights[
            (bitmap & (1 << np.arange(weights.shape[0]))).astype(bool)
        ]

        combination_y_pred = EnsembleModel2._combine_weighted_voting(
            predictions=[pd.Series(pred) for pred in combination_predictions],
            weights=[weight for weight in combination_weights],
        )

        combination_accuracy = (combination_y_pred == y_test).sum() / len(y_test)

        del combination_y_pred

        gc.collect()

        return "-".join(combination_names), combination_accuracy
    except Exception as e:
        logger.error(f"Error in evaluate_combination: {e}")
        raise e


def run_parralel_bitmap_processing(
    ensemble_model: EnsembleModel2, y_test: pd.Series, fold: int, processes: int
) -> List[Tuple[str, int, float]]:

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
    mp_weights = mp.Array("d", len(ensemble_model.weights))
    mp_weights_np = np.frombuffer(mp_weights.get_obj())
    mp_weights_np[:] = ensemble_model.weights  # Assign values to the shared array

    # Setup for names
    mp_names_np = mp.Manager().list(ensemble_model.combination_names)

    logger.info("Starting multiprocessing")
    results: List[Tuple[str, int, float]] = []
    # --- Multiprocessing ---
    with mp.Pool(processes=processes) as pool:
        mp_results = pool.starmap(
            evaluate_combination,
            [
                (
                    i,
                    mp_predictions_np,
                    mp_weights_np,
                    mp_names_np,
                    mp_y_true_np,
                )
                for i in range(1, 2**num_models)
            ],
        )

    results = [
        (combination_names, fold, accuracy)
        for combination_names, accuracy in mp_results
    ]
    return results


def optimize_ensemble(
    ensemble_model: EnsembleModel2,
    X: pd.DataFrame,
    y: pd.Series,
    n_cv: int,
    processes: int,
) -> Tuple[List[str], pd.DataFrame]:
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
        ensemble_model.predict(X_test)

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
    best_combination = str(results_df.idxmax())
    best_combination_names = best_combination.split("-")

    return best_combination_names, results_df.to_frame()


def test_ensemble(
    ensemble_model: EnsembleModel2,
    X: pd.DataFrame,
    y: pd.Series,
    n_cv: int,
) -> pd.DataFrame:
    logger.info("Testing ensemble model")
    kfold = KFold(n_splits=n_cv)
    results_list: List[Tuple[str, int, float]] = []

    for cnt, (train_index, test_index) in enumerate(kfold.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        logger.info(f"Testing fold {cnt + 1}")

        ensemble_model.fit(X_train, y_train)

        ensemble_model.predict(X_test)

        for i in range(len(ensemble_model.estimators)):
            y_pred = ensemble_model.predictions[i]
            accuracy = (y_pred == y_test).sum() / len(y_test)
            results_list.append((ensemble_model.combination_names[i], cnt, accuracy))  # type: ignore

        combination_name = "-".join(ensemble_model.combination_names)
        y_pred = ensemble_model.predict(X_test)
        accuracy = (y_pred == y_test).sum() / len(y_test)
        results_list.append((combination_name, cnt, accuracy))

    results_df = pd.DataFrame(
        results_list,
        columns=["combination", "fold", "score"],
    )
    results_df = results_df.groupby("combination")["score"].mean()

    return results_df.to_frame()
