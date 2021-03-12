# Author: Dino Bollinger
# LICENSE: MIT
"""
Recompute the validation stats using the validation dataset that was split away during training.
To be used with XGBoost, LightGBM or Catboost.
Can be used to optimize the decision rule used for predicting labels using the soft probabilities.

Usage:
    validation_stats_xgb <model> <valid_set> [--matrix_only] [--feat_contribs]

Options:
    -f --feat_contribs   Compute the feature contributions prediction (takes some time)
    -m --matrix_only     Only produce confusion matrix output.
    -h                   Show this help message.
"""

import numpy as np
import xgboost as xgb
import catboost as catb
import lightgbm as lgbm
import pickle
import os
from docopt import docopt
from datetime import datetime

from classifiers.xgboost.train_xgb import analyse_feature_contribs
from classifiers.utils import (log_accuracy_and_confusion_matrix, log_validation_statistics, setupLogger,
                               get_optimized_loss_weights, bayesian_decision)

import logging

logger = logging.getLogger("classifier")


def main() -> int:
    """  """
    argv = None
    argv = ["../performance_reports/xgboost_best_18_01_21/models/xgbmodel_20210119_004712.xgb",
            "../performance_reports/xgboost_best_18_01_21/xgb_predict_stats/validation_matrix_20210119_004617.sparse",
            "--feat_contribs"]
    class_names = ["necessary", "functional", "analytics", "advertising"]

    cargs = docopt(__doc__, argv=argv)
    setupLogger(f"./val_stats{''}.log")

    # check for errors in input parameters
    model_path: str = cargs["<model>"]
    if not os.path.exists(model_path):
        logger.error("Specified model path does not exist.")
        return 2

    validset_path: str = cargs["<valid_set>"]
    if not os.path.exists(validset_path):
        logger.error("Specified validation file does not exist.")
        return 3

    true_labels: np.ndarray
    predicted_probabilities: np.ndarray
    if model_path.endswith(".xgb"):
        model = xgb.Booster(model_file=model_path)
        if validset_path.endswith(".sparse"):
            with open(validset_path, 'rb') as fd:
                data = pickle.load(fd)
            with open(validset_path + ".labels", 'rb') as fd:
                tlabels = pickle.load(fd)
            dtest = xgb.DMatrix(data, label=tlabels)
        else:
            dtest = xgb.DMatrix(validset_path)
        predicted_probabilities = model.predict(dtest, training=False)
        true_labels: np.ndarray = dtest.get_label().astype(int)
    elif model_path.endswith(".lgbm"):
        with open(validset_path, 'rb') as fd:
            data = pickle.load(fd)
        model = lgbm.Booster(model_file=model_path)
        dmat = lgbm.Dataset(data)
        predicted_probabilities = model.predict(dmat)
        labels_fn = validset_path + ".labels"
        with open(labels_fn, 'rb') as fd:
            labels = pickle.load(fd)
        true_labels: np.ndarray = np.array(labels).astype(int)
    elif model_path.endswith(".catb"):
        with open(validset_path, 'rb') as fd:
            data = pickle.load(fd)
        model = catb.CatBoostClassifier()
        model.load_model(fname=model_path)
        predicted_probabilities = model.predict(data, prediction_type='Probability', thread_count=-1)
        labels_fn = validset_path + ".labels"
        with open(labels_fn, 'rb') as fd:
            labels = pickle.load(fd)
        true_labels: np.ndarray = np.array(labels).astype(int)
    else:
        raise ValueError("Unrecognized extension.")

    eval_path = f"./valid_stats{''}/"
    timestamp_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(eval_path, exist_ok=True)

    assert len(predicted_probabilities) == len(true_labels), f"Number of labels {len(true_labels)} did not match number of predictions {len(predicted_probabilities)}"

    if cargs["--feat_contribs"]:
        assert model_path.endswith(".xgb"), "Mode only supported for XGBoost"
        # Feature contribution matrix has dimensions nsamples x nclass x nfeats,
        # Serves to analyze the contribution each feature had for each prediction.
        pred_contribs_mat = model.predict(dtest, pred_contribs=True, training=False)
        analyse_feature_contribs(predicted_probabilities, pred_contribs_mat, true_labels, class_names, eval_path)
    elif cargs["--matrix_only"]:
        logger.info("Argmax Baseline:")
        argmax_weights = np.array([[0, 1.0, 1.0, 1.0], [1.0, 0, 1.0, 1.0], [1.0, 1.0, 0, 1.0], [1.0, 1.0, 1.0, 0]])
        disc_preds_bayes = bayesian_decision(predicted_probabilities, argmax_weights)
        log_accuracy_and_confusion_matrix(disc_preds_bayes, true_labels, class_names)

        logger.info(".............................")
        logger.info("Custom Loss Function:")
        loss_weights = get_optimized_loss_weights()
        disc_preds_bayes = bayesian_decision(predicted_probabilities, loss_weights)
        log_accuracy_and_confusion_matrix(disc_preds_bayes, true_labels, class_names)
        logger.info(f"Loss used: \n{loss_weights}")
    else:
        # output statistics such as confidence in predictions, confusion matrix and accuracy
        log_validation_statistics(predicted_probabilities, true_labels, class_names, eval_path, timestamp_str)


if __name__ == "__main__":
    exit(main())
