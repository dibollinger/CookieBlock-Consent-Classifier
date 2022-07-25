# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
TEST SCRIPT:
This script parses a dumped xgboost model and attempts to reimplement the predictor based on said dump.

Usage:
    xgbpredictor_dump.py <dump_json> (<validset_path>)
"""

import numpy as np

import logging
import os
import json
import pickle
from tqdm import tqdm

from time import perf_counter_ns
from datetime import datetime

from utils import setupLogger, get_equal_loss_weights, bayesian_decision, log_validation_statistics
from xgbpredictor_native import get_sample_input01, get_sample_input02
from docopt import docopt
from typing import Tuple, Union, Dict, List

logger = logging.getLogger("classifier.parsedump")

# Recursive class for the binary decision tree
class BinaryTreeNode:
    # Example Dump Entry: {"nodeid": 0, "depth": 0, "split": 470, "split_condition": 0, "yes": 1, "no": 2, "missing": 1, "children": []}

    def __init__(self, dict_node: Dict[str, Union[int, float, List[Dict]]]):

        # Apparently, the model contains trees that have absolutely no decisions, only a single leaf weight.
        if "leaf" in dict_node:
            self.direct_value = dict_node["leaf"]
            return
        else:
            self.direct_value = None

        self.feature_idx: int = dict_node["split"]
        self.split_cond: float = dict_node["split_condition"]

        children: List = dict_node["children"]

        # Four apparent constants within the model:
        # There are always two children to every non-leaf node.
        # The yes path is always on the left, the no path is always on the right, and for undefined data, always the left path is chosen.
        # The following verifies these constants
        assert len(children) == 2, "Given tree was not binary!"
        assert dict_node["yes"] < dict_node["no"], f"Yes path didn't go down true tree: yes -- {dict_node['yes']}, no -- {dict_node['no']}"
        assert dict_node["missing"] == dict_node["yes"], "Missing path did not choose the yes path"

        self.left_child: Union[float, BinaryTreeNode]
        if "leaf" in children[0]:
            self.left_child = children[0]["leaf"]
        else:
            self.left_child = BinaryTreeNode(children[0])

        self.right_child: Union[float, BinaryTreeNode]
        if "leaf" in children[1]:
            self.right_child = children[1]["leaf"]
        else:
            self.right_child = BinaryTreeNode(children[1])

    def __call__(self, features) -> float:
        """
        Retrieve the leaf weight for the given feature array.
        :param features: An array-like data structure of features.
        :return: Leaf weight gained for this feature vector.
        """
        if self.direct_value is not None:
            return self.direct_value

        self.left_child: Union[float, BinaryTreeNode]
        self.right_child: Union[float, BinaryTreeNode]
        # Missing data
        if features[self.feature_idx] == 0.0:
            if isinstance(self.left_child, BinaryTreeNode):
                return self.left_child(features)
            else:
                return self.left_child
        # True case
        elif features[self.feature_idx] < self.split_cond:
            if isinstance(self.left_child, BinaryTreeNode):
                return self.left_child(features)
            else:
                return self.left_child
        # False Case
        else:
            if isinstance(self.right_child, BinaryTreeNode):
                return self.right_child(features)
            else:
                return self.right_child


def load_samples():
    """ Load two different samples for testing """
    true_label01, feature_vector01 = get_sample_input01()
    true_label02, feature_vector02 = get_sample_input02()

    true_labels = np.array([true_label01, true_label02])
    feature_vectors = np.vstack((feature_vector01, feature_vector02))
    return true_labels, feature_vectors


def load_validation(validset_path:str ):
    """ Load the entire validation set used for this booster. """
    with open(validset_path, 'rb') as fd:
        data = pickle.load(fd)
    with open(validset_path + ".labels", 'rb') as fd:
        tlabels = pickle.load(fd)
    return tlabels, data.toarray()


def load_decision_forests(model_dump: List[Dict[str, Union[float, List[Dict]]]], num_classes: int) -> List[List[BinaryTreeNode]]:
    """
    Initialize the decision forests using the model dump for the given number of classes.
    Assumes that the number of trees is divisible by the number of classes.
    :param model_dump: Model dump in the form of a list of dictionaries, loaded from JSON.
    :param num_classes: Number of classes to assume were used for this model.
    :return: A list of forests, where each forest is a list of root tree nodes.
    """
    assert len(model_dump) % num_classes == 0, f"Total number of trees {len(model_dump)} is not divisible by the number of classes {num_classes}."

    trees_per_class = [list() for i in range(num_classes)]
    for i in range(len(model_dump)):
        trees_per_class[i % num_classes].append(BinaryTreeNode(model_dump[i]))
    return trees_per_class


def forest_predict(class_forests: List[List[BinaryTreeNode]], features: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """
    Applies predictions to the input feature vectors.
    :param class_forests: List of decision forests, each forest corresponding to a class label in sequence.
    :param features: 2 dimensional numpy array, rows are instances, columns are features.
    :return: Predicted probabilities, as well as the discrete prediction, using an argmax approach.
    """
    assert len(features.shape) == 2, "input features must be a 2d numpy array"
    all_weights = []

    # Inefficient as it is done sequentially -- this isn't intended for bulk application anyways so it is not important
    for i in tqdm(range(features.shape[0])):
        all_weights.append([np.sum([tree(features[i]) for tree in forest]) for forest in class_forests])

    # This approach to computing the probabilities appears to be the one employed by XGBoost, but the predictions don't match exactly.
    # Unfortunately, I have no way of determining whether this is due to the dumped model, or because the algorithm is wrong.
    # Either way, the validation accuracy differs by at most .3%, so it is likely not a huge deal.
    exp_lweights = np.exp(np.vstack(all_weights))
    predicted_probabilities = exp_lweights / np.sum(exp_lweights, axis=1, keepdims=True)
    discrete_prediction = bayesian_decision(predicted_probabilities, get_equal_loss_weights())
    return predicted_probabilities, discrete_prediction


def main():
    """ Run the prediction using the decision tree reconstructed from the model dump. """
    argv = None
    cargs = docopt(__doc__, argv=argv)

    validation_stats: bool = False
    setupLogger("./modeldump_validation.log" if validation_stats else None)

    # Load the model dump
    model_dump: List[Dict[str, Union[float, List[Dict]]]]
    dump_path = cargs["<dump_json>"]
    if dump_path.endswith(".json"):
        if os.path.exists(dump_path):
            with open(dump_path, 'r') as fd:
                model_dump = json.load(fd)
        else:
            logger.error("Specified dump file does not exist.")
            return 2
    elif cargs["<dump_json>"].endswith(".txt"):
        logger.error("Text format dump is not supported.")
        return 3
    else:
        logger.error("Unrecognized extension. Only .json dump format is supported.")
        return 4

    # Set up the decision tree
    start = perf_counter_ns()
    decision_forests = load_decision_forests(model_dump, num_classes=4)
    logger.info(f"One-time setup time: {(perf_counter_ns() - start) / 1000000:.3f}ms")

    # Load the data to predict labels for
    true_labels, features = load_validation(cargs["<validset_path>"]) if validation_stats else load_samples()

    # Perform the prediction
    start = perf_counter_ns()
    probs, disc_pred = forest_predict(decision_forests, features)
    logger.info(f"Prediction time: {(perf_counter_ns() - start) / 1000000:.3f}ms")

    # Output results
    if validation_stats:
        # output statistics such as confidence in predictions, confusion matrix and accuracy
        class_names = ["Necessary", "Functional", "Analytics", "Advertising"]
        eval_path = f"./dump_predict_stats{''}/"
        os.makedirs(eval_path, exist_ok=True)
        timestamp_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_validation_statistics(probs, true_labels, class_names, eval_path, timestamp_str)
    else:
        np.set_printoptions(suppress=True)
        logger.info(f"Predicted Probabilities:\n{probs}")
        logger.info(f"True Labels: {true_labels}")
        logger.info(f"Predicted Labels: {disc_pred}")

    return 0


if __name__ == "__main__":
    exit(main())
