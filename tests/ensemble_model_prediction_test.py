from copy import deepcopy
import json
from pathlib import Path
import pickle
from typing import List, cast
import pandas as pd
import pytest


from lib.ensemble.EnsembleConfigDict import EnsembleConfigDict
from lib.models.EnsembleModel import EnsembleModel
from lib.pipelines.ProcessingPipelineWrapper import ProcessingPipelineWrapper
from ml_poisonous_mushrooms.data_load.data_load import load_data, load_ensemble_config
from ml_poisonous_mushrooms.engineering.engineering_features import engineer_features
from ml_poisonous_mushrooms.utils.PathManager import PathManager
from ml_poisonous_mushrooms.utils.PrefixManager import PrefixManager


@pytest.fixture
def ensemble_model() -> EnsembleModel:
    config = load_ensemble_config()
    ensemble_model_path = Path(
        f"{PathManager.OUTPUT_DIR_PATH.value}"
    ) / f"{PrefixManager.ENSEMBLE_PREFIX.value}{config['model_run']}" / "ensemble_model.pkl"

    return cast(EnsembleModel, pickle.load(open(ensemble_model_path, "rb")))


@pytest.fixture
def ensemble_config() -> EnsembleConfigDict:
    return load_ensemble_config()


def test_ensemble_model_prediction():
    """
    A function that checks if the ensemble model is using its models the same was as the invividual models are used.

    The idea is that if we use models in an ensemble model seperately, we should get some predicitons.
    If we use ensemble model to predict, we should get the same predictions.

    This is a unit test to verify that the ensemble model is using its models correctly.
    """

    # --- SETUP ---
    config = load_ensemble_config()
    ensemble_model_path = Path(
        f"{PathManager.OUTPUT_DIR_PATH.value}"
    ) / f"{PrefixManager.ENSEMBLE_PREFIX.value}{config['model_run']}" / "ensemble_model.pkl"

    ensemble_model = cast(EnsembleModel, pickle.load(
        open(ensemble_model_path, "rb")))

    # --- DATA ---
    train, test = load_data()
    train = train.head(int(len(train) * 0.12))
    engineered_data = engineer_features(train).set_index("id")

    X_train = engineered_data.head(int(len(engineered_data) * 0.5))
    X_test = engineered_data.tail(int(len(engineered_data) * 0.5))
    y_train = X_train["class"].head(int(len(engineered_data) * 0.5))

    # --- ACT ---
    e1 = deepcopy(ensemble_model)
    e1.fit(X_train, y_train).predict(X_test)  # type

    e2 = deepcopy(ensemble_model)
    singular_model_predictions: List[pd.Series] = []
    for i, model in enumerate(e2.models):
        pipeline = ProcessingPipelineWrapper().create_pipeline(model)
        pipeline = pipeline.fit(
            X_train[ensemble_model.combination_feature_lists[i]], y_train)

        prediction = pipeline.predict(
            X_test[ensemble_model.combination_feature_lists[i]])

        prediction = pd.Series(
            prediction, index=X_test.index, name=ensemble_model.combination_names[i])

        singular_model_predictions.append(prediction)

    # --- ASSERT ---
    if e1.predictions is None:
        raise ValueError("Ensemble model did not predict anything")

    for i in range(len(e1.models)):
        assert e1.predictions[i].name == singular_model_predictions[i].name
        assert e1.predictions[i].equals(singular_model_predictions[i])


def test_ensemble_models_params(ensemble_model: EnsembleModel, ensemble_config: EnsembleConfigDict):
    """
    This is a test to check if the ensemble's models are using correct parameters and if those models
    by themselves are using the correct parameters.
    """

    # --- SETUP ---
    results = pd.read_csv(
        f"{PathManager.OUTPUT_DIR_PATH.value}/{PrefixManager.HYPER_OPT_PREFIX.value}{ensemble_config['model_run']}/results_{ensemble_config['model_run']}.csv")

    # --- ASSERT ---
    for i, model in enumerate(ensemble_model.models):
        combination_name = ensemble_model.combination_names[i]
        result_params = json.loads(
            results.loc[results["name"].eq(combination_name), "params"].iloc[0].replace("'", "\""))

        model_params = model.get_params()

        for key in result_params.keys():
            assert key in model_params
            assert model_params[key] == result_params[key]


def test_ridge_models_have_same_predictions(ensemble_model: EnsembleModel):

    # --- DATA ---
    train, test = load_data()
    train = train.head(int(len(train) * 0.05))
    engineered_data = engineer_features(train).set_index("id")

    X_train = engineered_data.head(int(len(engineered_data) * 0.5))
    X_test = engineered_data.tail(int(len(engineered_data) * 0.5))
    y_train = X_train["class"].head(int(len(engineered_data) * 0.5))

    # --- ACT ---

    # --- Ensemble model ---
    e1 = deepcopy(ensemble_model)
    e1.fit(X_train, y_train).predict(X_test)

    if e1.predictions is None:
        raise ValueError("Ensemble model did not predict anything")

    e_ridge_prediction = [
        prediction for prediction in e1.predictions if prediction.name is not None and "ridge" in str(prediction.name).lower()][0]

    # --- Ridge model ---
    e2 = deepcopy(ensemble_model)

    ridge = [
        model for model in e2.models if "ridge" in model.__class__.__name__.lower()][0]

    pipeline = ProcessingPipelineWrapper().create_pipeline(model=ridge)
    features_in = [
        features for (i, features) in enumerate(
            ensemble_model.combination_feature_lists) if "ridge" in ensemble_model.combination_names[i].lower()][0]

    pipeline = pipeline.fit(X_train[features_in], y_train)
    ridge_prediction = pd.Series(pipeline.predict(
        X_test[features_in]), index=X_test.index, name="ridge")

    # --- ASSERT ---
    assert e_ridge_prediction.equals(ridge_prediction)
