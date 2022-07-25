# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
TEST SCRIPT:
Parses the new minimal format tree dumps and uses it to construct a predictor.
Unlike the other script, this one does not utilize arrays, and instead expects dictionaries as feature input.
This is done to prepare for the Javascript implementation, and check the accuracy of the extracted model.

Usage:
    xgbpredictor_sparsedump.py <dump_files>...
"""

import logging
import os
import json

from math import exp
from time import perf_counter_ns

from utils import setupLogger
from docopt import docopt
from typing import Tuple, Union, Dict, List

logger = logging.getLogger("classifier.newpredictor")


def get_sample_features01():
    return 0, {"356": 0.833333333, "357": -1.0, "358": 4.0, "359": -1.0, "360": 24.0, "362": 32.0, "364": 3.31977384,
               "365": 0.0948489746, "367": 1.0, "368": 1.0, "369": -1.0, "370": -1.0, "371": -1.0, "372": -1.0,
               "373": 1.0, "374": -1.0, "375": 31535999.0, "376": -1.0, "377": -1.0, "378": -1.0, "379": -1.0,
               "380": -1.0, "381": -1.0, "382": 1.0, "383": -1.0, "384": 24.0, "385": 32.0, "386": -8.0,
               "387": 3.25270548, "388": -1.0, "389": -1.0, "390": -1.0, "391": -1.0, "392": -1.0, "393": -1.0,
               "394": -1.0, "445": -1.0, "446": -1.0, "447": -1.0, "448": -1.0, "449": -1.0, "450": -1.0, "451": -1.0,
               "452": -1.0, "453": -1.0, "454": -1.0, "455": -1.0, "456": -1.0, "457": -1.0, "458": -1.0, "459": -1.0,
               "460": -1.0, "461": -1.0, "462": -1.0, "463": -1.0, "464": 1.0, "465": -1.0, "466": -1.0, "467": -1.0,
               "468": -1.0, "469": -1.0, "470": 1.0, "471": -1.0, "472": -1.0, "473": -1.0, "474": -1.0, "475": -1.0,
               "476": -1.0, "477": -1.0}


def get_sample_features02():
    return 2, {"0": 1.0, "356": 1.0, "360": 26.0, "362": 34.0, "364": 3.5073801, "369": -1.0, "370": -1.0,
               "371": -1.0, "372": 1.0, "373": -1.0, "374": -1.0, "375": 63071999.0, "376": -1.0, "377": -1.0,
               "378": -1.0, "379": -1.0, "380": -1.0, "381": -1.0, "382": -1.0, "383": 1.0, "384": 26.0,
               "385": 34.0, "386": -8.0, "387": 3.5073801, "388": -1.0, "389": -1.0, "390": -1.0, "391": -1.0,
               "392": -1.0, "393": -1.0, "394": -1.0, "445": -1.0, "446": -1.0, "447": -1.0, "448": -1.0,
               "449": -1.0, "450": -1.0, "451": -1.0, "452": -1.0, "453": -1.0, "454": -1.0, "455": -1.0,
               "456": -1.0, "457": -1.0, "458": -1.0, "459": -1.0, "460": -1.0, "461": -1.0, "462": -1.0,
               "463": -1.0, "464": 1.0, "465": -1.0, "466": -1.0, "467": -1.0, "468": -1.0, "469": 1.0, "470": -1.0,
               "471": -1.0, "472": -1.0, "473": -1.0, "474": -1.0, "475": -1.0, "476": -1.0, "477": -1.0}


def get_score(tree_node: Dict, features: Dict) -> float:
    """
    Given a tree node and corresponding features, retrieve the weight.
    Recursive function. Depth is limited to the maximum tree depth in the forest.
    :param tree_node: Node or Leaf of the tree.
    :param features: Features to base decisions on, in the form of a sparse dict.
    :return: The score resulting from the input features.
    """
    # Example Node: {"f": 470, "c": 0, "u": "l", "l": {}, "r": {}}
    # Example Leaf: {"v": 32.0}
    if "v" in tree_node:
        return tree_node["v"]
    else:
        fidx = str(tree_node["f"])
        if fidx not in features:
            return get_score(tree_node[tree_node["u"]], features)
        elif features[fidx] < tree_node["c"]:
            return get_score(tree_node["l"], features)
        else:
            return get_score(tree_node["r"], features)


def forest_predict(class_forests: List[Dict], features: Dict) -> Tuple[List[float], int]:
    """
    Use the forest dictionary to predict the label for the given input features.
    This variant of the predictor only accepts a single instance at a time.
    :param class_forests: Forest in the form of a list of dictionaries.
    :param features: Sparse feature dictionary.
    :return: Class probabilities and discrete prediction
    """
    class_scores = [sum([get_score(root, features) for root in forest]) for forest in class_forests]
    exp_class_scores = [exp(s) for s in class_scores]
    total_score = sum(exp_class_scores)
    predicted_probabilities = [s / total_score for s in exp_class_scores]
    discrete_prediction = predicted_probabilities.index(max(predicted_probabilities))
    return predicted_probabilities, discrete_prediction


def main():
    """ Run the prediction using the decision tree reconstructed from the model dump. """
    argv = None
    cargs = docopt(__doc__, argv=argv)

    setupLogger(None)

    # Load the model dump
    start = perf_counter_ns()
    model_dumps: List[Dict[str, Union[float, List[Dict]]]] = list()
    dump_paths = cargs["<dump_files>"]
    for d in dump_paths:
        if d.endswith(".json"):
            if os.path.exists(d):
                with open(d, 'r') as fd:
                    model_dumps.append(json.load(fd))
            else:
                logger.error(f"Specified dump file {d} does not exist.")
                return 2
        elif d.endswith(".txt"):
            logger.error(f"Text format dump {d} is not supported.")
            return 3
        else:
            logger.error(f"Unrecognized file extension for {d}. Only .json dump format is supported.")
            return 4
    logger.info(f"One-time setup: {(perf_counter_ns() - start) / 1000000:.3f}ms")

    # Load the data to predict labels for
    # true_label, features = get_sample_features01()
    true_label, features = get_sample_features02()

    # Perform the prediction
    start = perf_counter_ns()
    probs, disc_pred = forest_predict(model_dumps, features)
    logger.info(f"Prediction time: {(perf_counter_ns() - start) / 1000000:.3f}ms")

    # Output results
    logger.info(f"Predicted Probabilities: {probs}")
    logger.info(f"True Labels: {true_label}")
    logger.info(f"Predicted Labels: {disc_pred}")

    return 0


if __name__ == "__main__":
    exit(main())
