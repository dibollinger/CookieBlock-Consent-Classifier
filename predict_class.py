# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Using a pretrained model, and given cookie data in JSON format, predict labels for each cookie.
Can choose between the three tree boosters.

Usage:
    predict_class <model_path> <json_data>

Options:
    -h --help              Show this help message.
"""

from docopt import docopt
import os
import sys
import json
import logging

import xgboost as xgb
import lightgbm as lgbm
import catboost as catb

import numpy as np
from scipy.sparse import csr_matrix

from typing import Union, Dict, Optional
from feature_extraction.processor import CookieFeatureProcessor

from classifiers.utils import log_accuracy_and_confusion_matrix

logger = logging.getLogger("classifier")


def get_equal_loss_weights():
    """ Replicates the argmax probability decision. """
    return np.array([[0., 1., 1., 1.],
                     [1., 0, 1., 1.],
                     [1., 1., 0, 1.],
                     [1., 1., 1., 0]])


def bayesian_decision(prob_vectors: np.ndarray, loss_weights: np.ndarray):
    """
    Compute class predictions using Bayesian Decision Theory.
    :param prob_vectors: Probability vectors returns by the multiclass classification.
    :param loss_weights: nclass x nclass matrix, loss per classification choice
    :return: Numpy array of discrete label predictions.
    """
    num_instances, num_classes = prob_vectors.shape
    assert loss_weights.shape == (num_classes, num_classes), f"Loss weight matrix shape does not match number of actual classes: {loss_weights.shape} vs. {num_classes} classes"
    b = np.repeat(prob_vectors[:, :, np.newaxis], num_classes, axis=2)
    return np.argmin(np.sum(b * loss_weights, axis=1), axis=1)


class ModelWrapper:

    def __init__(self, model_path: str) -> None:
        self.model: Optional[Union[xgb.Booster, lgbm.Booster, catb.CatBoost]] = None
        if model_path.endswith(".xgb"):
            self.model = xgb.Booster(model_file=model_path)
        elif model_path.endswith(".lgbm"):
            self.model = lgbm.Booster(model_file=model_path)
        elif model_path.endswith(".catb"):
            self.model = catb.CatBoostClassifier()
            self.model.load_model(fname=model_path)
        else:
            error_msg: str = f"Unrecognized model type for '{model_path}'"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def predict_with_bayes_decision(self, data: csr_matrix, loss_function: np.ndarray) -> np.ndarray:
        if type(self.model) is xgb.Booster:
            dmat = xgb.DMatrix(data)
            predicted_probabilities = self.model.predict(dmat, training=False)
        elif type(self.model) is lgbm.Booster:
            dmat = lgbm.Dataset(data)
            predicted_probabilities = self.model.predict(dmat)
        elif type(self.model) is catb.CatBoost:
            predicted_probabilities = self.model.predict(data, prediction_type='Probability', thread_count=-1)
        else:
            error_msg: str = f"Unrecognized model type loaded!"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return bayesian_decision(predicted_probabilities, loss_function)


def setup_logger() -> None:
    """Log to standard output"""
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


def main() -> int:
    """
    Run the prediction.
    :return: exit code
    """
    argv = None
    cargs = docopt(__doc__, argv=argv)

    setup_logger()

    test_data_json: Dict[str, Dict]
    data_path = cargs["<json_data>"]
    if os.path.exists(data_path):
        try:
            with open(data_path) as fd:
                test_data_json = json.load(fd)
        except json.JSONDecodeError:
            logger.error("File is not a valid JSON object.")
            return 2
    else:
        logger.error(f"File does not exist: '{data_path}'")
        return 1

    model_path = cargs["<model_path>"]
    if not os.path.exists(model_path):
        logger.error(f"Model does not exist: '{model_path}'")
        return 3

    # Set up feature processor and model
    cfp = CookieFeatureProcessor("./feature_extraction/features.json", skip_cmp_cookies=False)
    model = ModelWrapper(model_path)

    logger.info("Extracting Features...")
    cfp.extract_features_with_labels(test_data_json)
    sparse = cfp.retrieve_sparse_matrix()
    true_labels = np.array(cfp.retrieve_labels())

    logger.info("Predicting Labels...")
    loss_function = np.array([[0, 1.0, 1.0, 1.0], [1.0, 0, 1.0, 1.0], [1.0, 1.0, 0, 1.0], [1.0, 1.0, 1.0, 0]])
    discrete_predictions = model.predict_with_bayes_decision(sparse, loss_function)

    assert len(discrete_predictions) == len(test_data_json)

    log_accuracy_and_confusion_matrix(discrete_predictions, true_labels, ["Necessary", "Functional", "Analytics", "Advertising"])

    pred_json: Dict[str, int] = dict()
    i: int = 0
    for k in test_data_json.keys():
        pred_json[k] = int(discrete_predictions[i])
        i += 1

    with open("predictions.json", 'w') as fd:
        json.dump(pred_json, fd)

    # Check accuracy on Consent Cookies
    consentcookie_predictions = {"OptanonConsent": [0,0,0,0], "CookieConsent": [0,0,0,0]}
    i: int = 0
    for k in test_data_json.keys():
        if k.startswith("OptanonConsent;"):
            consentcookie_predictions["OptanonConsent"][discrete_predictions[i]] += 1
        elif k.startswith("CookieConsent;"):
            consentcookie_predictions["CookieConsent"][discrete_predictions[i]] += 1
        i += 1

    logger.info(f"Predictions on Consent Cookies: {consentcookie_predictions}")
    cfp.reset_processor()

    return 0


if __name__ == "__main__":
    exit(main())
