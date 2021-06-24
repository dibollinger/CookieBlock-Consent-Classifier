# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Train an XGBoost model.
Supported data input formats are "sparse", "LibSVM" and "XGB", identified by extension.

The recommended input format is pickled sparse matrix, as this is the quickest way to load data.
LibSVM text works too but comes with some overhead in setting up.

<mode> is one of {"train", "split", "cross_validate", "grid_search", "random_search"}.

Usage:
    train_xgb <tr_data> <mode> [<class_weights>]

Options
    -h --help   Show this help message.
"""

import logging
import os
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle

from docopt import docopt
from datetime import datetime

from collections import Counter

from utils import load_data, log_validation_statistics, setupLogger, save_validation
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

from typing import Union, Optional, List, Dict, Tuple

logger = logging.getLogger("classifier.xgboost")


def get_search_params() -> Dict[str, List[Union[None, float, int, str]]]:
    """
    Used for parameter search. Each entry is a list of values.
    """
    return {
        'booster': ['gbtree'],  # alternative: 'dart'
        'tree_method': ['hist'],
        'n_estimators': [100],  # aka. number of rounds/trees
        'verbosity': [1],
        'seed': [0],

        # Tree parameters
        'learning_rate': [0.5],           # [0.2, 0.3, 0.4],
        'gamma': [1],                     # [0, 1, 3],  # minimum loss reduction required to further partition
        'max_depth': [7, 9],              # [5, 6, 7, 8],  # maximum tree depth
        'min_child_weight': [1, 10, 30],  # minimum sum of instance weight needed in a child
        'max_delta_step': [0],            # 0 is no constraint, may be useful for class imbalance
        'subsample': [1.0],               # ratio of training data to use in each step
        'colsample_bytree': [0.5, 1],     # number of features to use, per each tree

        'sampling_method': ['uniform'],
        'reg_lambda': [2],                # L2 regularizer
        'reg_alpha': [0],                 # L1 regularizer
        'grow_policy': ['depthwise'],     # alternative: 'lossguide' -- only supported for 'hist'
        'max_leaves': [0],                # max number of leaf nodes, only relevant with 'lossguide'
        'max_bin': [256, 512, 768],       # only used with 'hist' -- larger -> improve optimality of splits but higher comp. time
        'predictor': ['auto'],            # 'cpu_predictor' or 'gpu_predictor'
        'base_score': [0.5],              # default is 0.5
        'interaction_constraints': [None],

        # Following only used with DART
        "rate_drop": [0.1],
        "one_drop": [1]
    }


def get_best_params() -> Dict[str, Union[None, float, int, str]]:
    """
    Best training parameters found so far, for simple CV/split training.
    """
    return {'booster': "gbtree",
            'verbosity': 1,
            'nthread': int(cpu_count() - 1),

            # Tree parameters
            'learning_rate': 0.25,
            'gamma': 1,
            'max_depth': 32,
            'min_child_weight': 3,
            'max_delta_step': 0,
            'subsample': 1,
            'sampling_method': 'uniform',
            'lambda': 1,
            'alpha': 2,
            'grow_policy': 'depthwise',
            'max_leaves': 0,
            'max_bin': 256,
            'predictor': 'auto',

            # Learning Task Parameters
            'objective': 'multi:softprob',
            'eval_metric': ['merror', 'mlogloss'],
            'num_class': 4,
            'base_score': 0.2,
            }


def save_model(bst) -> None:
    """
    Save the given booster model.
    :param bst: Computed booster.
    """
    dummy = ""
    model_path: str = f"./models{dummy}/"
    os.makedirs(model_path, exist_ok=True)
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    file_path = os.path.join(model_path, f"xgbmodel_{now_str}.xgb")
    bst.save_model(file_path)
    logger.info(f"Model dumped to {file_path}")


def custom_metrics(lweights: np.ndarray, data) -> List[Tuple[str, float]]:
    """
    Custom evaluation metrics:
        Currently computes the negated sum of the f1score based on the argmax probability prediction.
        Can also however be altered to output the sum of the precision, recall, or per-class outputs.
    :param lweights: Leaf weights.
    :param data: Features + Labels
    :return: A list of tuples, custom evaluation metrics name + value. Last one in the list is the one used for early stopping.
    """
    y = data.get_label()
    num_instances, num_classes = lweights.shape

    ## NOTE: This is how XGBoost computes the class weights from the leaf weight sums
    #softprob = np.exp(lweights) / np.sum(np.exp(lweights), axis=1, keepdims=True)
    #logger.info(y[0])
    #logger.info(softprob[0])

    # predt is untransformed leaf weight -- argmax prediction is however equivalent
    disc_preds = np.argmax(lweights, axis=1)

    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        z = np.bincount(disc_preds[y == i])
        confusion_matrix[i][:] = np.pad(z, (0, num_classes - len(z)), mode='constant', constant_values=0)

    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)

    f1_score = 2 * ((precision * recall) / (precision + recall))
    # return [('m_anti_recall', - np.sum(recall - 1)), ('m_anti_precision', - np.sum(precision - 1)), ('mf1score', -np.sum(f1_score))]
    return [('mf1score', -np.sum(f1_score))]


def analyse_feature_contribs(pred_probabilities: np.ndarray, feature_contributions: np.ndarray,
                             test_labels: np.ndarray, class_names: List[str], eval_path: str) -> None:
    """
    Outputs the average contribution of each feature in the dataset for the cases
    where the predicted label (based on a simple majority probability rule) matches
    the actual label in the dataset. These contributions are separated by class.

    The result of this should give us an idea about which features are most important
    for correctly predicting that a specific cookie belongs to a specific class.
    :param pred_probabilities: Predicted probabilities for each class.
    :param feature_contributions: Compute feature contributions for each prediction.
    :param test_labels: Actual labels of the test data for which we predicted labels.
    :param class_names: Class names. (also required because of length)
    :param eval_path: Output folder for the computed feature contributions.
    """
    num_classes = len(class_names)

    # Total feature importance sum for each class
    sums_feat_imp = [np.zeros(feature_contributions.shape[2]) for l in range(num_classes)]

    # number of instances for each class
    count_feat_imp = [0 for l in range(num_classes)]

    # compute the predicted label based on a simple majority probability
    for l in range(len(test_labels)):
        actual_label = int(test_labels[l])
        predicted_label = np.argmax(pred_probabilities[l])
        # Only sum feature contribution wherever the prediction was correct.
        if actual_label == predicted_label:
            sums_feat_imp[actual_label] += feature_contributions[l, actual_label]
            count_feat_imp[actual_label] += 1

    # Compute average contribution separated by class, and output the contributiosn separately for each class
    for i in range(num_classes):
        avg_feat_cont = sums_feat_imp[i] / count_feat_imp[i]
        avg_feat_path = os.path.join(eval_path, f"avg_cont_{class_names[i]}.txt")
        with open(avg_feat_path, 'w') as fd:
            for cont in avg_feat_cont:
                fd.write(f"{cont}\n")
        logger.info(f"Average Feature Contributions for {class_names[i]} written to {avg_feat_path}")


def output_validation_statistics(bst, ntree: int, compute_feat_contribs: bool,
                                 X_test: csr_matrix, y_test: List[int], class_names: List[str]) -> None:
    """
    Output a large number of statistics resulting from predictions on the validation set to a subfolder.
    :param bst: Trained Booster
    :param ntree: Maximum number of trees to use for prediction
    :param compute_feat_contribs: If true, compute feature contributions (may take a long time)
    :param X_test: DMatrix of the validation data
    :param y_test:  Labels for the validation data (same ordering)
    :param class_names: names corresponding to the classes
    """
    eval_path = "./xgb_predict_stats/"
    timestamp_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(eval_path, exist_ok=True)

    dtest = xgb.DMatrix(data=X_test, label=y_test)

    predicted_probabilities: np.ndarray = bst.predict(dtest, ntree_limit=ntree, training=False)
    labels_as_nparray: np.ndarray = np.array(y_test).astype(int)
    assert len(predicted_probabilities) == len(labels_as_nparray), "Number of labels did not match number of predictions"

    # output statistics such as confidence in predictions, confusion matrix and accuracy
    log_validation_statistics(predicted_probabilities, labels_as_nparray, class_names, eval_path, timestamp_str)

    if compute_feat_contribs:
        # Feature contribution matrix has dimensions nsamples x nclass x nfeats,
        # Serves to analyze the contribution each feature had for each prediction.
        pred_contribs_mat = bst.predict(dtest, ntree_limit=ntree, pred_contribs=True, training=False)
        analyse_feature_contribs(predicted_probabilities, pred_contribs_mat, labels_as_nparray, class_names, eval_path)

    # dump validation data as xgb DMatrix
    #dtest_name = os.path.join(eval_path, f"validation_matrix_{timestamp_str}.buffer")
    #dtest.save_binary(dtest_name)
    #logger.info(f"Dumped Validation DMatrix to: {dtest_name}")

    # Save the validation data
    save_validation(X_test, y_test, eval_path, timestamp_str)


def simple_train(X: csr_matrix, y: List[int], weights: Optional[List[float]]) -> None:
    """
    Train on the full provided dataset and labels, with weights, and output the resulting model file to disk.
    No validation data used. Early Stopping is done on the training data.
    :param X: Dataset to train on.
    :param y: Labels for each data point.
    :param weights: Weights for each data point.
    """

    dtrain: xgb.DMatrix = xgb.DMatrix(data=X, label=y, weight=weights)
    params = get_best_params()

    # train on entire dataset, save that model
    evallist = [(dtrain, 'train')]
    bst: xgb.Booster = xgb.train(params, dtrain,
                                 num_boost_round=20, early_stopping_rounds=3,
                                 verbose_eval=True, evals=evallist)

    try:
        logger.info(f"Best Score: {bst.best_score}")
        logger.info(f"Best Iteration: {bst.best_iteration}")
        logger.info(f"Best NTree Limit: {bst.best_ntree_limit}")
    except AttributeError:
        pass

    save_model(bst)


def split_train(X: csr_matrix, y: List[int], weights: Optional[List[float]],
                train_fraction: float = 0.8, apply_weights_to_validation: bool = True) -> None:
    """
    Perform a training set / validation set split and train the XGBoost model.
    Outputs the resulting model, trained on the specified fraction of data.
    The remaining fraction is used for validation.

    This function is intended to give more information about the resulting prediction in a separate validation set.
    Most importantly, it outputs a confusion matrix that can be used to analyse which categories still cause issues.
    :param X: Training features, as a sparse matrix.
    :param y: Training Labels, as a list of integers.
    :param weights: Weights for each label.
    :param train_fraction: Fraction of data to use for training.
    :param apply_weights_to_validation: If false, will not include instance weights for the validation set.
    """
    assert type(X) is not xgb.DMatrix, "Can't perform train-test split on XGB DMatrix input"
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    class_names = ["necessary", "functional", "analytics", "advertising"]
    num_classes = len(class_names)
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    # Split the data into train and test set
    (X_train, X_test,
     y_train, y_test,
     w_train, w_test) = train_test_split(X, y, weights, train_size=train_fraction, shuffle=True)

    dtrain: xgb.DMatrix = xgb.DMatrix(data=X_train, label=y_train, weight=w_train)

    # Validate either with or without the weights. May give significantly different performance.
    # If weights are not applied, will not be comparable with cross-validation or hyperparam search.
    dtest: xgb.DMatrix
    if apply_weights_to_validation:
        dtest = xgb.DMatrix(data=X_test, label=y_test, weight=w_test)
    else:
        dtest = xgb.DMatrix(data=X_test, label=y_test)

    # Retrieve Parameters
    params = get_best_params()
    logger.info("Parameters:")
    logger.info(params)

    # Validation data must appear last to be used by early stopping
    evallist = [(dtrain, 'train'), (dtest, 'eval')]

    evals_result = dict()
    bst: xgb.Booster = xgb.train(params, dtrain, num_boost_round=20, early_stopping_rounds=10,
                                 verbose_eval=True, evals=evallist, evals_result=evals_result)

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
    output_validation_statistics(bst, limit, False, X_test, y_test, class_names)

    # Save the model to the computed models subfolder
    save_model(bst)


def split_train_crossvalidate(X: csr_matrix, y: List[int], weights: Optional[List[float]],
                train_fraction: float = 0.8, apply_weights_to_validation: bool = True) -> None:
    """
    Apply the split training as if it were cross-validation.
    """
    assert type(X) is not xgb.DMatrix, "Can't perform train-test split on XGB DMatrix input"
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    class_names = ["necessary", "functional", "analytics", "advertising"]
    num_classes = len(class_names)
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    y_np = np.array(y)
    w_np = np.array(weights)
    kf = KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(X, y_np):
        # Split the data into train and test set
        #    (X_train, X_test,
        #     y_train, y_test,
        #     w_train, w_test) = train_test_split(X, y, weights, train_size=train_fraction, shuffle=True)
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y_np[train_indices]
        y_test = y_np[test_indices]
        w_train = w_np[train_indices]
        w_test = w_np[test_indices]

        dtrain: xgb.DMatrix = xgb.DMatrix(data=X_train, label=y_train, weight=w_train)

        # Validate either with or without the weights. May give significantly different performance.
        # If weights are not applied, will not be comparable with cross-validation or hyperparam search.
        dtest: xgb.DMatrix
        if apply_weights_to_validation:
            dtest = xgb.DMatrix(data=X_test, label=y_test, weight=w_test)
        else:
            dtest = xgb.DMatrix(data=X_test, label=y_test)

        # Retrieve Parameters
        params = get_best_params()
        logger.info("Parameters:")
        logger.info(params)

        # Validation data must appear last to be used by early stopping
        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        evals_result = dict()
        bst: xgb.Booster = xgb.train(params, dtrain, num_boost_round=30, early_stopping_rounds=10,
                                     feval=custom_metrics, verbose_eval=True, evals=evallist, evals_result=evals_result)

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
        output_validation_statistics(bst, limit, False, X_test, y_test, class_names)


def crossvalidate_train(X, y: Optional[List[int]],
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

    dtrain: xgb.DMatrix
    if type(X) is xgb.DMatrix:
        dtrain = X
    else:
        dtrain: xgb.DMatrix = xgb.DMatrix(data=X, label=y, weight=weights)

    params = get_best_params()
    logger.info("Parameters:")
    logger.info(params)

    cv_results = xgb.cv(params, dtrain, num_boost_round=30, nfold=5, stratified=True, metrics=["merror", "mlogloss"],
                        early_stopping_rounds=10, show_stdv=True, verbose_eval=True, seed=random_seed, shuffle=True)

    logger.info(cv_results)

    eval_path = "./crossvalidate_results/"
    os.makedirs(eval_path, exist_ok=True)
    timestamp_now = datetime.now()
    cv_results_path = os.path.join(eval_path, f"cv_results_{timestamp_now}.pkl")
    with open(cv_results_path, 'wb') as fd:
        pickle.dump(cv_results, fd)
    logger.info(cv_results_path)


def paramsearch_train(X, y: List[int], weights: Optional[List[float]],
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
    assert type(X) is not xgb.DMatrix, "Can't perform gridsearch on XGB DMatrix input"
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    num_classes = 4
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    param_lists = get_search_params()
    logger.info("Parameter Lists:")
    logger.info(param_lists)

    # Need to use SKLearn API for this
    # Objective is either "multi:softmax" or "multi:softprob"
    model = xgb.XGBClassifier(use_label_encoder=False, objective='multi:softmax',
                              num_class=4, importance_type='gain',
                              verbosity=1, n_jobs=2, random_state=random_seed)

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
    dummy = ""
    psearch_path: str = f"param_search{dummy}/"
    os.makedirs(psearch_path, exist_ok=True)
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    psearch_stats_path = os.path.join(psearch_path, f"search_stats_{now_str}.csv")

    logger.info(f"Output hyperparameter search stats to '{psearch_stats_path}'")
    pd.DataFrame(clf.cv_results_).to_csv(psearch_stats_path, index=False)

    # dump best estimator as booster
    bst: xgb.Booster = clf.best_estimator_.get_booster()
    save_model(bst)

    logger.info(f"Best Score:")
    logger.info(clf.best_score_)

    logger.info("Best Hyperparameters:")
    logger.info(clf.best_params_)


def main() -> int:
    """ Perform training of an xgboost_other model """
    argv = None
    cargs = docopt(__doc__, argv=argv)
    setupLogger(f"./train_xgb{''}.log")

    # check for errors in input parameters
    tr_dat_path: str = cargs["<tr_data>"]
    if not os.path.exists(tr_dat_path):
        logger.error("Specified training data file does not exist.")
        return 2

    # load features, labels and weights
    X, y, W = load_data(tr_dat_path)

    # if X is an XGBMatrix, assume that the labels are integrated
    if y is None and type(X) is not xgb.DMatrix:
        logger.error("Could not load labels -- labels required for training.")
        return 3

    # set up instance weights based on class weights, if specified
    if cargs["<class_weights>"]:
        class_weights = []
        with open(cargs["<class_weights>"], 'r') as fd:
            for line in fd:
                class_weights.append(float(line.strip()))
        W = [class_weights[int(l)] for l in y]


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
        split_train_crossvalidate(X,y,W)
        #crossvalidate_train(X, y, W, random_seed=0)
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
