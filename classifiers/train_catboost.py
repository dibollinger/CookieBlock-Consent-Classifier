# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Train a Catboost model.
Supported data input formats are "sparse" and "LibSVM", identified by file extension.

The recommended input format is pickled sparse matrix, as this is the quickest way to load data.
LibSVM text works too but comes with some overhead in setting up.

<mode> is one of {"train", "split", "cross_validate", "grid_search", "random_search"}.

Usage:
    train_catboost.py <tr_data> <mode>

Options:
    -h --help        Show this help message.
"""

import logging
import os
import catboost as catb
import pandas as pd
import numpy as np
import pickle

from docopt import docopt
from datetime import datetime

from collections import Counter

from utils import load_data, log_validation_statistics, setupLogger, save_validation
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from typing import Union, Optional, List, Dict

logger = logging.getLogger("classifier.catboost")


def get_fixed_params() -> Dict[str, Union[None, float, int, str]]:
    """
    Retrieve dictionary of a single fixed set of parameters, for simple CV/split training.
    """
    return {
        'loss_function': 'MultiClass',
        'custom_metric': ['Accuracy', 'Precision', 'Recall', 'TotalF1'],
        'eval_metric': "Accuracy",
        'num_boost_round': 2000,
        "early_stopping_rounds": 30,
        'od_type': "IncToDec",  # Iter # Overfitting detector
        'od_pval': 0,  # For best results, it is recommended to set a value in the range [10^-10, 10^-2].
        'reg_lambda': 2.0,  # L2 regularizer (higher is more conservative)
        'bootstrap_type': 'Bayesian',  # method for sampling weights
        'bagging_temperature': 1,  # exponential dist. at 1
        'random_strength': 1,
        'max_depth': 12,  # up to max of 16
        'grow_policy': "Depthwise",  # ,"SymmetricTree" Lossguide
        'min_child_samples': 1,  # minimum number of training samples
        'leaf_estimation_method': "Newton",  # Gradient, Exact
        'auto_class_weights': "SqrtBalanced",  # "Balanced", #
        'langevin': False,  # SGD boosting mode
        'logging_level': "Verbose",

        # 'boosting_type': 'Ordered', #determined automatically
        # 'max_leaves': 32, #only used for lossguide
        # 'classes_count': 4, # automatically determined
        #    'learning_rate': 0.25,  # disabled as it is determined automatically
        #    'subsample': 1  # automatically determined
        # 'roc_file': "" # only for cross validation

        ## use for text type feature columns
        # 'tokenizer': ,
        # 'dictionaries': ,
        # 'feature_calcers': ,
    }


def get_search_params() -> Dict[str, List[Union[None, float, int, str]]]:
    """
    Used for parameter search. Each entry is a list of values.
    """
    return {
        # 'loss_function': ['MultiClass'],
        # 'custom_metric': [['Accuracy', 'Precision', 'Recall', 'TotalF1']],
        # 'eval_metric': ['MultiClass'],
        'num_boost_round': [2500, 3000, 4500],
        "early_stopping_rounds": [30],
        'od_type': ["IncToDec"],  # Iter # Overfitting detector
        'od_pval': [0],  # For best results, it is recommended to set a value in the range [10^-10, 10^-2].
        # 'classes_count': 4, # automatically determined
        #    'learning_rate':[0.25],  # disabled as it is determined automatically
        #    'subsample': [1]  # automatically determined
        'reg_lambda': [3.0],  # L2 regularizer (higher is more conservative)
        'bootstrap_type': ['Bayesian'],  # method for sampling weights
        'bagging_temperature': [1],  # exponential dist. at 1
        'random_strength': [1],
        'max_depth': [10, 11, 12],  # up to max of 16
        'grow_policy': ["Depthwise"],  # , Lossguide
        'min_child_samples': [1],  # minimum number of training samples
        # 'max_leaves': [32], #only used for lossguide
        'leaf_estimation_method': ["Newton"],  # Gradient, Exact
        # Tree parameters
        'auto_class_weights': ["SqrtBalanced"],
        # 'boosting_type': ['Ordered'], #determined automatically
        'langevin': [False],  # SGD boosting mode
        'logging_level': ["Verbose"]
    }


def save_model(bst: catb.CatBoostClassifier) -> None:
    """
    Save the given booster model.
    :param bst: Computed booster.
    """
    model_path: str = f"./models{''}/"
    os.makedirs(model_path, exist_ok=True)
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_path = os.path.join(model_path, f"catbmodel_{now_str}.xgb")
    bst.save_model(file_path)
    logger.info(f"Model dumped to '{file_path}'")


def output_validation_statistics(bst: catb.CatBoostClassifier, ntree: int, dtest: csr_matrix, y_test: List[int], class_names: List[str]) -> None:
    """
    Output a large number of statistics resulting from predictions on the validation set to a subfolder.
    :param bst: Trained Booster
    :param ntree: Maximum number of trees to use for prediction
    :param dtest: DMatrix of the validation data
    :param y_test:  Labels for the validation data (same ordering)
    :param class_names: names corresponding to the classes
    """
    eval_path = "./catboost_predict_stats/"
    timestamp_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(eval_path, exist_ok=True)

    predicted_probabilities: np.ndarray = bst.predict(dtest, prediction_type="Probability", ntree_end=ntree)
    labels_as_nparray: np.ndarray = np.array(y_test).astype(int)
    assert len(predicted_probabilities) == len(labels_as_nparray), "Number of labels did not match number of predictions"

    # output statistics such as confidence in predictions, confusion matrix and accuracy
    log_validation_statistics(predicted_probabilities, labels_as_nparray, class_names, eval_path, timestamp_str)

    # Save the validation data
    save_validation(dtest, y_test, eval_path, timestamp_str)


def simple_train(X: csr_matrix, y: List[int], weights: Optional[List[float]]) -> None:
    """
    Train on the full provided dataset and labels, with weights, and output the resulting model file to disk.
    No validation data used. Early Stopping is done on the training data.
    :param X: Dataset to train on.
    :param y: Labels for each data point.
    :param weights: Weights for each data point.
    """
    class_names = ["necessary", "functional", "analytics", "advertising"]
    num_classes = len(class_names)
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    params = get_fixed_params()

    # train on entire dataset, save that model
    class_weights = {i: weights[y.index(i)] for i in range(num_classes)}
    bst = catb.CatBoostClassifier()  # class_weights=class_weights)
    bst.set_params(**params)
    bst.fit(X, y, verbose_eval=True)

    try:
        logger.info(f"Best Score: {bst.best_score}")
        logger.info(f"Best Iteration: {bst.best_iteration}")
        logger.info(f"Best NTree Limit: {bst.best_ntree_limit}")
    except AttributeError:
        pass

    save_model(bst)


def split_train(X: csr_matrix, y: List[int], weights: Optional[List[float]], train_fraction: float = 0.8) -> None:
    """
    Perform a training set / validation set split and train the CatBoost model.
    Outputs the resulting model, trained on the specified fraction of data.
    The remaining fraction is used for validation.

    This function is intended to give more information about the resulting prediction in a separate validation set.
    Most importantly, it outputs a confusion matrix that can be used to analyse which categories still cause issues.
    :param X: Training features, as a sparse matrix.
    :param y: Training Labels, as a list of integers.
    :param weights: Weights for each instance.
    :param train_fraction: Fraction of data to use for training.
    """
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    class_names = ["necessary", "functional", "analytics", "advertising"]
    num_classes = len(class_names)
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    # Split the data into train and test set
    (X_train, X_test,
    y_train, y_test,
    w_train, w_test) = train_test_split(X, y, weights, train_size=train_fraction, shuffle=True)

    # Retrieve Parameters
    params = get_fixed_params()
    logger.info(f"Parameters: {params}")

    # Validation data must appear last to be used by early stopping
    evalset = (X_test, y_test)

    bst: catb.CatBoostClassifier = catb.CatBoostClassifier()
    bst.set_params(**params)
    bst.fit(X_train, y_train, use_best_model=True, eval_set=evalset, verbose_eval=True)

    # Output the information collected by early stopping
    limit: int = 0
    try:
        logger.info(f"Best Score: {bst.best_score}")
        logger.info(f"Best Iteration: {bst.best_iteration}")
        logger.info(f"Best NTree Limit: {bst.best_ntree_limit}")
        limit = bst.best_ntree_limit
    except AttributeError:
        pass
    logger.info(bst.get_params())

    # Produce statistics on validation set (confusion matrix, accuracy, etc.)
    output_validation_statistics(bst, limit, X_test, y_test, class_names)

    # Save the model to the computed models subfolder
    save_model(bst)


def crossvalidate_train(X: csr_matrix, y: Optional[List[int]],
                        weights: Optional[List[float]], random_seed: Optional[int] = None):
    """
    Perform cross-validation to measure the mean effectiveness of the classifier, then train on the entire dataset.
    Early stopping set to an interval of 3. Outputs a model trained on the entire dataset.
    :param X: Training data
    :param y: Training labels
    :param weights: Label weights.
    :param random_seed: Random seed for Cross-Validation kfold split.
    """
    num_classes = 4
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"


    params = get_fixed_params()
    logger.info(f"Parameters: {params}")

    cv_results = catb.cv(catb.Pool(X, y), params, num_boost_round=2000, nfold=5, stratified=True,
                         early_stopping_rounds=5, verbose_eval=True, seed=random_seed, shuffle=True)

    logger.info(cv_results)

    eval_path = "./crossvalidate_results/"
    os.makedirs(eval_path, exist_ok=True)
    timestamp_now = datetime.now()
    cv_results_path = os.path.join(eval_path, f"cv_results_{timestamp_now}.pkl")
    with open(cv_results_path, 'wb') as fd:
        pickle.dump(cv_results, fd)
    logger.info(cv_results_path)


def paramsearch_train(X: csr_matrix, y: List[int], weights: Optional[List[float]],
                      search_type: str, random_seed: Optional[int] = None) -> None:
    """
    Perform either grid search or random search on parameters, verifying performance with 5-fold cross-validation.
    Outputs the best model to disk. Produces a search statistics CSV file to analyze.
    :param X: Training data
    :param y: Training labels
    :param weights: Label weights.
    :param search_type: Either "grid" for gridsearch or "random" for random search.
    :param random_seed: Seed to use for initialization
    """
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    num_classes = 4
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    param_lists = get_search_params()
    logger.info("Parameter Lists:")
    logger.info(param_lists)

    # Need to use SKLearn API for this
    # Objective is either "multi:softmax" or "multi:softprob"
    model: catb.CatBoostClassifier = catb.CatBoostClassifier(loss_function="MultiClass", random_seed=random_seed,
                                                             eval_metric="TotalF1", thread_count=2)

    # Use either grid search or random search
    num_combinations = 64
    if search_type == "grid":
        clf = GridSearchCV(model, param_lists, n_jobs=5,
                           cv=StratifiedKFold(n_splits=5, shuffle=True),
                           scoring=['balanced_accuracy', 'neg_log_loss'],
                           verbose=3, refit='balanced_accuracy')
    elif search_type == "random":
        clf = RandomizedSearchCV(model, param_lists, n_iter=num_combinations, n_jobs=5,
                                 cv=StratifiedKFold(n_splits=5, shuffle=True),
                                 scoring=['balanced_accuracy', 'neg_log_loss'],
                                 verbose=3, refit='balanced_accuracy',
                                 random_state=random_seed)
    else:
        raise ValueError(f"Invalid Type: {search_type}")

    # Finally start the search with the provided sample weights
    clf.fit(X, y, sample_weight=weights, eval_metric=['merror', 'mlogloss'])

    # Write the hyperparameter search results to a subfolder
    psearch_path: str = f"param_search{''}/"
    os.makedirs(psearch_path, exist_ok=True)
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    psearch_stats_path = os.path.join(psearch_path, f"search_stats_{now_str}.csv")

    logger.info(f"Output hyperparameter search stats to '{psearch_stats_path}'")
    pd.DataFrame(clf.cv_results_).to_csv(psearch_stats_path, index=False)

    # dump best estimator as booster
    bst: catb.CatBoostClassifier = clf.best_estimator_.get_booster()
    save_model(bst)

    logger.info(f"Best Score:")
    logger.info(clf.best_score_)

    logger.info("Best Hyperparameters:")
    logger.info(clf.best_params_)


def main() -> int:
    """ Perform training of an catboost model """
    argv = None
    cargs = docopt(__doc__, argv=argv)
    setupLogger(f"./train_catboost{''}.log")

    # check for errors in input parameters
    tr_dat_path: str = cargs["<tr_data>"]
    if not os.path.exists(tr_dat_path):
        logger.error("Specified training data file does not exist.")
        return 2

    # load features, labels and weights
    X, y, W = load_data(tr_dat_path)

    if y is None:
        logger.error("Could not load labels -- labels required for training.")
        return 3

    # Mode of training
    mode: str = cargs["<mode>"]

    # Switch on training mode
    if mode == "train":
        simple_train(X, y, W)
    elif mode == "split":
        logger.info("Training using train-test split.")
        split_train(X, y, W)
    elif mode == "cross_validate":
        logger.info("Training using crossvalidation.")
        crossvalidate_train(X, y, W, random_seed=0)
    elif mode == "grid_search":
        logger.info("Training using  split.")
        paramsearch_train(X, y, W, search_type="grid", random_seed=0)
    elif mode == "random_search":
        paramsearch_train(X, y, W, search_type="random", random_seed=0)
    else:
        logger.error('Unrecognized mode. Available modes are: {"train", "split", "cross_validate", "grid_search", "random_search"}')
        return 100


if __name__ == "__main__":
    exit(main())
