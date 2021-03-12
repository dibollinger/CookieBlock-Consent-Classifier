# Author: Dino Bollinger
# License: MIT
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import pickle
import os
import xgboost as xgb
from typing import Union, Optional, List, Callable

from statistics import mean, stdev

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger("classifier")


def save_validation(dtest: csr_matrix, y_test: List[int], eval_path, timestamp_str) -> None:
    """
    Save validation data.
    :param dtest: Validation split.
    :param y_test: Corresponding labels.
    :param eval_path: Path to store the files in
    :param timestamp_str: Timestamp for the filename
    """
    dtest_name = os.path.join(eval_path, f"validation_matrix_{timestamp_str}.sparse")
    with open(dtest_name, 'wb') as fd:
        pickle.dump(dtest, fd, pickle.HIGHEST_PROTOCOL)
    logger.info(f"Dumped Validation DMatrix to: {dtest_name}")

    labels_fn = dtest_name + ".labels"
    with open(labels_fn, 'wb') as fd:
        pickle.dump(y_test, fd, pickle.HIGHEST_PROTOCOL)


def load_data(dpath: str):
    """ Multi purpose data loading function.
    Supports loading from xgb binary, libsvm and pickled sparse matrix.
    The filetype is determined from the extension in the given path string.
    :param dpath: Path for the data to be loaded.
    :return a tuple of (features, labels, weights), where 'features' is a sparse matrix and the others are lists.
    """
    features: Union[xgb.DMatrix, csr_matrix]
    labels: Optional[List[float]] = None
    weights: Optional[List[float]] = None

    if dpath.endswith(".buffer"):
        logger.info("Loading DMatrix data...")
        features = xgb.DMatrix(dpath)

    elif dpath.endswith(".libsvm"):
        logger.info("Loading LibSVM data...")
        features, labels = load_svmlight_file(dpath)
        weights_fn = dpath + ".weights"
        if os.path.exists(weights_fn):
            with open(weights_fn, 'rb') as fd:
                weights = pickle.load(fd)

    elif dpath.endswith(".sparse"):
        logger.info("Loading sparse data...")
        with open(dpath, 'rb') as fd:
            features = pickle.load(fd)

        labels_fn = dpath + ".labels"
        if os.path.exists(labels_fn):
            with open(labels_fn, 'rb') as fd:
                labels = pickle.load(fd)

        weights_fn = dpath + ".weights"
        if os.path.exists(weights_fn):
            with open(weights_fn, 'rb') as fd:
                weights = pickle.load(fd)
    else:
        logger.error("Unknown data input format.")
        return None

    logger.info("Loading complete.")

    return features, labels, weights


def get_optimized_loss_weights():
    """
     Returns a 4 x 4 matrix of weights, indicating how severely a cookie misclassification
     is graded, using the categories: {0:Necessary, 1:Functional, 2:Analytics, 3:Advertising}
     The row index corresponds to the true class, while the columns correspond to predictions.
    -----------------------------------------------------------------------------------
     The general idea is that necessary and functional cookies are similar in nature,
     as are analytical and advertising cookies. Necessary and Functional cookies both
     serve to enable essential or non-essential functionality on the website, while
     analytics and advertising cookies serve to track the users, the former on the same
     domain, while the latter tracks across multiple domains.
    -----------------------------------------------------------------------------------
     In that vein, a classification of a necessary cookie as "functional" is less severe
     of a mistake than classifying it as advertising, and vice-versa.
     :return A 4x4 numpy array of weights.
    """
    return np.array([[0.0, 0.25, 0.75, 1.0],
                     [0.25, 0.0, 0.5, 0.75],
                     [0.75, 0.5, 0.0, 0.5],
                     [1.0, 0.75, 0.5, 0.0]])


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


def log_confidence_per_label(probs_with_label: pd.DataFrame, class_names: List[str],
                             use_true_label: bool, comp: Callable, logstring: str) -> None:
    """
    Prints the confidence for each label to the log.
    This is based on the predicted class probabilities by the classifier.
    Predicted label is computed via a simple argmax probability.
    :param probs_with_label: Pandas dataframe, first column is true labels, rest is probabilities per class.
    :param class_names: Names for each of the classes (and number of classes implicitly)
    :param use_true_label: If true, output confidence for the true label. False, output confidence for the predicted label.
    :param comp: Callable function to decide when to count the confidence (e.g. if true label matches predicted label)
    :param logstring: String prefix to use in the log message.
    """
    num_classes: int = len(class_names)

    # Output mean confidence in proper label to log
    confidence_per_label = [list() for i in range(num_classes)]
    for index, row in probs_with_label.iterrows():
        true_label = int(row[0])
        maxprob_label = np.argmax(row[1:])
        if comp(true_label, maxprob_label):
            cur_label = int(row[0]) if use_true_label else maxprob_label
            confidence_per_label[cur_label].append(row[cur_label + 1])

    for i in range(num_classes):
        if len(confidence_per_label[i]) > 1:
            logger.info(logstring + f"'{class_names[i]}': {mean(confidence_per_label[i]) * 100.0:.3f}%"
                                    f"+{stdev(confidence_per_label[i]) * 100.0:.3f}%")


def log_accuracy_and_confusion_matrix(discrete_predictions: np.ndarray, true_labels: np.ndarray, class_names: List[str]) -> None:
    """
    Log the confusion matrix, and overall accuracy rates for each class inside that confusion matrix.
    Also log the total accuracy over all classes.
    :param discrete_predictions: Numpy vector of discrete predictions (i.e. labels)
    :param true_labels: The true label vector, to compute the accuracy.
    :param class_names: Names for the classes.
    """
    num_classes: int = len(class_names)
    num_instances: int = len(discrete_predictions)
    assert max(discrete_predictions) <= (num_classes - 1), "Number of classes in predictions exceeds expected maximum."

    # Compute confusion matrix and output as CSV via pandas
    confusion_matrix: np.ndarray = np.zeros((num_classes, num_classes), dtype=int)

    # Note: this expects labels to be contiguous starting from 0, ending at num_classes, with no gaps in numbering
    for i in range(num_instances):
        confusion_matrix[int(true_labels[i]), discrete_predictions[i]] += 1

    # Output the confusion matrix to the log
    logger.info(f"Confusion Matrix:\n{confusion_matrix}")

    # DISABLED: Output the individual error rates per class
    # for i in range(num_classes):
    #    logger.info(f"Precision errors by true class '{class_names[i]}': {confusion_matrix[:, i] / np.sum(confusion_matrix[:, i])}")
    #    logger.info(f"Recall errors by class '{class_names[i]}': {confusion_matrix[i, :] / np.sum(confusion_matrix[i, :])}")

    # Output Precision + Recall
    precision_vector = np.zeros(num_classes)
    recall_vector = np.zeros(num_classes)
    f1_score_vector = np.zeros(num_classes)
    for i in range(num_classes):
        precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        precision_vector[i] = precision
        recall_vector[i] = recall
        f1_score_vector[i] = 2 * ((precision * recall) / (precision + recall))

    logger.info(f"Total Accuracy: {np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix) * 100.0:.3f}%")
    logger.info(f"Precision: {precision_vector}")
    logger.info(f"Recall: {recall_vector}")
    logger.info(f"F1 Scores: {f1_score_vector}")


def log_validation_statistics(predicted_probs: np.ndarray, true_labels: np.ndarray,
                              class_names: List[str], eval_path: str, timestamp: str) -> None:
    """
    Write the probability predictions to disk and log confidence and confusion matrix.
    :param predicted_probs: N x M matrix, N instances, M classes with probabilities
    :param true_labels: True labels to base accuracy metric off of.
    :param class_names: Names for the classes
    :param eval_path: Path to output files to.
    :param timestamp: timestamp for the filenames.
    """
    # Output the softprob predictions with true labels to a csv first.
    preds_df = pd.DataFrame(predicted_probs, columns=class_names)
    preds_df.insert(0, "labels", true_labels)

    softprob_path = os.path.join(eval_path, f"softprob_predictions_{timestamp}.csv")
    preds_df.to_csv(softprob_path, index=False)
    logger.info(f"Dumped softprob prediction matrix csv to: {softprob_path}")

    # Compute the confidence for 3 different cases:

    # Output confidence for each true label
    log_confidence_per_label(preds_df, class_names, True, (lambda t, m: True),
                             "Mean/Stddev confidence in true label overall: ")

    # Output confidence where the prediction was correct
    log_confidence_per_label(preds_df, class_names, False, (lambda t, m: t == m),
                             "Mean/Stddev confidence in predicted label, where prediction is correct: ")

    # Ouput confidence where the prediction was wrong
    log_confidence_per_label(preds_df, class_names, False, (lambda t, m: t != m),
                             "Mean/Stddev confidence in predicted label, where prediction is incorrect: ")

    logger.info("....................................................................")
    logger.info("Predicted labels & accuracy when using ARGMAX as a prediction rule")
    disc_preds_argmax = np.argmax(predicted_probs, axis=1)
    log_accuracy_and_confusion_matrix(disc_preds_argmax, true_labels, class_names)

    logger.info("....................................................................")
    logger.info("Predicted labels & accuracy when using Bayesian Decision Theory with Loss Weights")
    loss_weights = get_optimized_loss_weights()
    disc_preds_bayes = bayesian_decision(predicted_probs, loss_weights)
    log_accuracy_and_confusion_matrix(disc_preds_bayes, true_labels, class_names)
    logger.info(f"Loss Weights used: \n {loss_weights}")


def setupLogger(filename: Optional[str]) -> None:
    """
    Set up the logger instance, which will write its output to stderr.
    :param loglevel: Log level at which to record.
    """
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt="%Y-%m-%d-%H:%M:%S")
    ch = logging.StreamHandler()

    if filename:
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Separator to tell if new log started
    logger.info("=========================================")
