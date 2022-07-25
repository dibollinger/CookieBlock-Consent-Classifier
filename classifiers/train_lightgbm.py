# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Train a LightGBM model.
The expected training data format is libSVM or pickled sparse matrix.

<mode> is one of {"split", "cross_validate", "grid_search", "random_search"}.

Usage:
    train_lightgbm.py <tr_data> <mode>

Options:
    -h --help       Show this help message.
"""

import logging
import os
import lightgbm as lgbm
import pandas as pd
import numpy as np

from docopt import docopt
from datetime import datetime

from collections import Counter

from utils import load_data, log_validation_statistics, setupLogger, save_validation
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from typing import Union, Optional, List, Dict


logger = logging.getLogger("classifier.lightgbm")


def get_fixed_params() -> Dict[str, Union[None, float, int, str]]:
    """
    Retrieve dictionary of a single fixed set of parameters, for simple CV/split training.
    """
    return {
        # objective params,
        "objective": "multiclass",  # multiclassova, cross_entropy, cross_entropy_lambda
        "num_class": 4,
        "is_unbalance": True,
        "sigmoid": 1.0,
        "boost_from_average": True,  # adjusts initial score to the mean of labels for faster convergence

        # basic params
        #"boosting": "goss",  # dart, goss, rf
        "num_rounds": 2000,
        "early_stopping_round": 30,  # 0 is disable
        "learning_rate": 0.3,
        "max_leaves": 128,
        "tree_learner": "serial",  # serial, feature, data, voting
        "num_threads": int(cpu_count() / 2),
        'device_type': 'cpu',  # "gpu"
        'verbosity': 1,

        # enable for consistent results
        #'seed': 0,
        #'deterministic': False,

        # learning control
        "force_col_wise": False,
        "force_row_wise": False,
        "max_depth": 12,  # maximum tree depth
        "min_data_in_leaf": 20,  # minimal number of data in one leaf. Higher is more conservative
        "min_sum_hessian_in_leaf": 1e-3,  # minimal sum hessian in one leaf. Higher is more conservative
        "subsample": 1.0,  # ratio of data to select
        "subsample_freq": 0,  # every kth iteration, perform subsampling
        "bagging_seed": 3,
        "extra_trees": False,  # use extremely randomized trees if true
        "max_delta_step": 0.0,  # used to limit the max output of tree leaves if > 0. 0 is no constraint.
        "lambda_l1": 0.0,  # L1 regularization
        "lambda_l2": 1.0,  # L2 regularization
        "path_smooth": 0.0,  # larger values give stronger regularization
        "interaction_constraints": "",

        # metric parameters
        'metric': ['multi_logloss', 'multi_error'],
        "is_provide_training_metric": True,
        "first_metric_only": False,

        # DART parameters
        "drop_rate": 0.1,  # dropout rate (fraction of trees to drop)
        "max_drop": 50,  # maximum number of trees to drop
        "skip_drop": 0.5,  # probability of skipping dropout
        "uniform_drop": False,

        # GOSS parameters
        "top_rate": 0.2,  # retain ratio of large gradient data
        "other_rate": 0.1,  # retain ratio of small gradient data

        # Categorical Feature Parameters
        "max_cat_threshold": 32,  # limit number of split points considered for categorical features
        "cat_l2": 10.0,  # cat feature L2 regularizer
        "cat_smooth": 10.0,  # reduce the effect of noises in cat features
        "max_cat_to_onehot": 4,  # ???
    }

def get_search_params() -> Dict[str, List[Union[None, float, int, str]]]:
    """
    Used for parameter search. Each entry is a list of values.
    """
    return {
        # objective params
        "objective": ["multiclass"],  # multiclassova, cross_entropy, cross_entropy_lambda
        "num_class": [4],
        "is_unbalance": [True],
        "sigmoid": [1.0],
        "boost_from_average": [True],  # adjusts initial score to the mean of labels for faster convergence

        # basic params
        "boosting": ["gbdt", "dart", "goss", "rf"],  # dart, goss, rf
        "num_rounds": [100],
        "learning_rate": [0.1],
        "max_leaves": [128],
        "tree_learner": ["serial", "feature", "data", "voting"],
        'device_type': ['cpu'],  # "gpu"
        'verbosity': [1],

        # enable for consistent results
        # 'seed': 0,
        # 'deterministic': False,

        # learning control
        "force_col_wise": [False],
        "force_row_wise": [False],
        "max_depth": [0],  # maximum tree depth
        "min_data_in_leaf": [20],  # minimal number of data in one leaf. Higher is more conservative
        "min_sum_hessian_in_leaf": [1e-3],  # minimal sum hessian in one leaf. Higher is more conservative
        "subsample": [1.0],  # ratio of data to select
        "subsample_freq": [0],  # every kth iteration, perform subsampling
        "bagging_seed": [3],
        "extra_trees": [False],  # use extremely randomized trees if true
        "max_delta_step": [0.0],  # used to limit the max output of tree leaves if > 0. 0 is no constraint.
        "lambda_l1": [0.0],  # L1 regularization
        "lambda_l2": [1.0],  # L2 regularization
        "path_smooth": [0.0],  # larger values give stronger regularization

        # metric parameters
        'metric': [['multi_logloss', 'multi_error']],
        "is_provide_training_metric": [True],
        "first_metric_only": [False],

        # DART parameters
        "drop_rate": [0.1],  # dropout rate (fraction of trees to drop)
        "max_drop": [50],  # maximum number of trees to drop
        "skip_drop": [0.5],  # probability of skipping dropout
        "uniform_drop": [False],

        # GOSS parameters
        "top_rate": [0.2],  # retain ratio of large gradient data
        "other_rate": [0.1],  # retain ratio of small gradient data

        # Categorical Feature Parameters
        "max_cat_threshold": [32],  # limit number of split points considered for categorical features
        "cat_l2": [10.0],  # cat feature L2 regularizer
        "cat_smooth": [10.0],  # reduce the effect of noises in cat features
        "max_cat_to_onehot": [4]  # ???
    }




def save_model(bst: lgbm.Booster) -> None:
    """
    Save the given booster model.
    :param bst: Computed booster.
    """
    model_path: str = "./models/"
    os.makedirs(model_path, exist_ok=True)
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_path = os.path.join(model_path, f"lgbmodel_{now_str}.lgbm")
    bst.save_model(file_path)
    logger.info(f"Model dumped to {file_path}")



def output_validation_statistics(bst: lgbm.Booster, dtest: csr_matrix, y_test: List[int],
                                 ntree: int, class_names: List[str]) -> None:
    """
    Output validation statistics such as the confusion matrix, precision, recall and more.
    :param bst: Trained LightGBM Booster
    :param dtest: LightGBM Dataset of the validation data
    :param y_test:  Labels for the validation data (same ordering)
    :param ntree: Maximum number of trees to use for prediction
    :param num_classes: Number of classes to use
    :param class_names: names corresponding to the classes
    """
    # Further evaluation data
    eval_path = "./xgb_predict_stats/"
    os.makedirs(eval_path, exist_ok=True)
    timestamp_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get actual predictions for the validation data
    predicted_probabilities: np.ndarray = bst.predict(dtest, num_iteration=ntree)
    dtest_labels: np.ndarray = np.array(y_test)
    assert len(predicted_probabilities) == len(dtest_labels), "Number of labels did not match number of predictions"

    # output statistics such as confidence in predictions, confusion matrix and accuracy
    log_validation_statistics(predicted_probabilities, dtest_labels, class_names, eval_path, timestamp_str)

    # Save the validation data
    save_validation(dtest, y_test, eval_path, timestamp_str)


def simple_train(X: csr_matrix, y: List[int], weights: Optional[List[float]]) -> None:
    """
    Train a model on the full training data with no validation.
    :param X: Full training data in sparse matrix format.
    :param y: Labels as integers.
    :param weights: List of instance weights
    """
    dtrain: lgbm.Dataset = lgbm.Dataset(data=X, label=y, weight=weights)

    params = get_fixed_params()
    logger.info(f"Parameters: {params}")

    evals_result = dict()
    bst: lgbm.Booster = lgbm.train(params, train_set=dtrain, verbose_eval=True, valid_sets=[dtrain], evals_result=evals_result)

    save_model(bst)


def split_train(X: csr_matrix, y: List[int], weights: Optional[List[float]],
                train_fraction: float = 0.8, apply_weights_to_validation: bool = True) -> None:
    """
    Perform a training set / validation set split and train the LightGBM model.
    Outputs the resulting model, trained on the specified fraction of data.
    The remaining fraction is used for the validation score.

    This function is intended to give more information about the resulting prediction in a separate validation set.
    Most importantly, it outputs a confusion matrix that can be used to analyse which categories still cause issues.
    :param X: Training features.
    :param y: Training Labels.
    :param weights: Weights for each label.
    :param train_fraction: Fraction of data to use for training.
    :param apply_weights_to_validation: If false, will not include instance weights for the validation set.
    """
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    num_classes = 4
    class_names = ["necessary", "functional", "analytics", "advertising"]
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    # Split the data into train and test set
    (X_train, X_test,
     y_train, y_test,
     w_train, w_test) = train_test_split(X, y, weights, train_size=train_fraction, shuffle=True)

    dtrain: lgbm.Dataset = lgbm.Dataset(data=X_train, label=y_train, weight=w_train)

    # Evaluation Dataset
    dtest: lgbm.Dataset
    if apply_weights_to_validation:
        dtest = lgbm.Dataset(data=X_test, label=y_test, weight=w_test)
    else:
        dtest = lgbm.Dataset(data=X_test, label=y_test)

    # Retrieve Parameters
    params = get_fixed_params()
    logger.info("Parameters:")
    logger.info(params)

    # Validation data must appear last to be used by early stopping
    evallist = [dtrain, dtest]
    evals_result = dict()

    bst: lgbm.Booster = lgbm.train(params, train_set=dtrain, verbose_eval=True, valid_sets=evallist, evals_result=evals_result)

    # Output the information collected by early stopping
    limit: int = 0
    try:
        logger.info(f"Best Score: {bst.best_score}")
        logger.info(f"Best Iteration: {bst.best_iteration}")
        logger.info(f"Best NTree Limit: {bst.best_ntree_limit}")
        limit = bst.best_ntree_limit
    except AttributeError:
        pass
    logger.info(evals_result)

    # Produce statistics on validation set (confusion matrix, accuracy, etc.)
    output_validation_statistics(bst, X_test, y_test, limit, class_names)

    # Save the model to the computed models subfolder
    save_model(bst)


def crossvalidate_train(X: csr_matrix, y: List[int], weights: Optional[List[float]],
                        random_seed: Optional[int] = None):
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

    dtrain: lgbm.Dataset = lgbm.Dataset(data=X, label=y, weight=weights)

    params = get_fixed_params()
    logger.info(f"Parameters: {params}")

    res = lgbm.cv(params, dtrain, nfold=5, stratified=True,
                  show_stdv=True, verbose_eval=True,
                  seed=random_seed, shuffle=True, eval_train_metric=True)

    logger.info(res)
    with open("./cvresults.pkl", 'wb') as fd:
        fd.write(res)



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

    model = lgbm.LGBMClassifier(objective='multiclass',
                                boosting_type="gbdt",  # {'gbdt', 'dart', 'goss', 'rf'}
                                num_class=4, importance_type='split',
                                verbosity=1, n_jobs=1, random_state=random_seed)

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
    clf.fit(X, y, sample_weight=weights)

    # Write the hyperparameter search results to a subfolder
    psearch_path: str = "./param_search/"
    os.makedirs(psearch_path, exist_ok=True)
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    psearch_stats_path = os.path.join(psearch_path, f"search_stats_{now_str}.csv")
    logger.info(f"Output hyperparameter search stats to '{psearch_stats_path}'")
    pd.DataFrame(clf.cv_results_).to_csv(psearch_stats_path, index=False)

    # dump best estimator as booster
    bst: lgbm.Booster = clf.best_estimator_.booster_
    save_model(bst)

    logger.info(f"Best Score:")
    logger.info(clf.best_score_)

    logger.info("Best Hyperparameters:")
    logger.info(clf.best_params_)



def main() -> int:
    """ Perform training of a LightGBM model """
    argv = None
    cargs = docopt(__doc__, argv=argv)
    setupLogger(f"./train_lightgbm{''}.log")

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
        logger.info("Simple training without validation on full dataset.")
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
        logger.error("Unrecognized mode. Available modes are: {split, cross_validate, grid_search, random_search}")
        return 100


if __name__ == "__main__":
    exit(main())
