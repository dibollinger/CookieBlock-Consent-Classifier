# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Script to transform cookies to feature vectors.
The cookie transformation is described in the file "feature_extraction/features.json".
Output format can be either XGB matrix, sparse matrix or as libsvm text.

Usage:
    prepare_training_data.py <tr_data>... [--format <FORMAT>] [--out <OFPATH>]

Options:
    -f --format <FORMAT>   Output format. Options: {libsvm, sparse, debug. xgb} [default: sparse]
    -o --out <OFPATH>      Filename for the output. If not specified, will reuse input filename.
    -h --help              Show this help message.
"""

import logging
import os
import json
import sys
import traceback

from typing import Dict, Any, List
from docopt import docopt
from datetime import datetime

from feature_extraction.processor import CookieFeatureProcessor

logger = logging.getLogger("feature-extract")

featuremap_path = "feature_extraction/features.json"

def setup_logger() -> None:
    """ Instruct the logger to output data to standard output and a log file """
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)

    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    fh = logging.FileHandler(f"./prepare_training_data{''}.log", mode='w', encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def main() -> int:
    """
    Extract feature vectors from the provided json cookie data.
    :return: exit code (0 success, 1 failure)
    """
    argv = None

    # Test arguments
#    argv = ["training_data/example_crawl_20210213_153228.json",
#            "--format", "sparse"]
#    argv = ["training_data/test_case.json",
#            "--format", "debug",
#            "--out", "debug_test"]

    cargs = docopt(__doc__, argv=argv)
    setup_logger()

    # load the inputs and check for errors
    input_data: Dict[str, Dict[str, Any]] = dict()
    for t in cargs["<tr_data>"]:
        try:
            if os.path.exists(t):
                with open(t, 'r') as fd:
                    t_data = json.load(fd)
            else:
                logger.error(f"Invalid data path specified: {t}")
                return 1
        except json.JSONDecodeError:
            logger.error(f"Input data is not a valid JSON file: {t}")
            logger.error(traceback.format_exc())
            return 2

        logger.info(f"Successfully retrieved json data from '{t}'")
        input_data = {**input_data, **t_data}

    logger.info(f"Number of cookies loaded: {len(input_data)}")

    if cargs["--format"] == "xgb":
        logger.info("Selected XGBoost matrix output format")
    elif cargs["--format"] == "sparse":
        logger.info("Output as sparse feature matrix.")
    elif cargs["--format"] == "libsvm":
        logger.info("Selected libsvm sparse text format")
    elif cargs["--format"] == "debug":
        logger.info("Selected debug format for checking the correctness of the feature extraction")
    else:
        logger.error("Unsupported output format")
        return 3

    # Initialize the feature processor
    feature_processor = CookieFeatureProcessor(featuremap_path, skip_cmp_cookies=True)

    # Print the number of features that will be extracted from each training data entry
    feature_processor.print_feature_info()

    logger.info("Begin feature transformation...")
    feature_processor.extract_features_with_labels(input_data)

    # output the results in the desired format
    dump_path: str = "./processed_features/"
    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(dump_path, exist_ok=True)

    # Output filename
    if cargs["--out"]:
        basefn = cargs["--out"]
    else:
        basefn = f"processed_{now_str}"

    # Output format
    if cargs["--format"] == "xgb":
        out_path = os.path.join(dump_path, basefn + ".buffer")
        dtrain = feature_processor.retrieve_xgb_matrix(include_labels=True, include_weights=True)
        dtrain.save_binary(out_path)
    elif cargs["--format"] == "sparse":
        out_path = os.path.join(dump_path, basefn + ".sparse")
        feature_processor.dump_sparse_matrix(out_path, dump_weights=True)
    elif cargs["--format"] == "libsvm":
        out_path = os.path.join(dump_path, basefn + ".libsvm")
        feature_processor.dump_libsvm(out_path, dump_weights=True)
    # Only intended for small data samples
    elif cargs["--format"] == "debug":
        out_path = os.path.join(dump_path, basefn + ".debug.txt")
        debug_out: List[Dict[str, float]] = feature_processor.retrieve_debug_output()
        with open(out_path, 'w') as fd:
            fd.write(json.dumps(debug_out, indent=4))
    else:
        raise ValueError("Unsupported output format -- this is not supposed to be reached.")

    if not cargs["--format"] == "debug":
        # This is not required to produce predictions, but it is required for making sense of the classifier
        feat_map_file = os.path.join(dump_path, f"feature_map_{now_str}.txt")
        feature_processor.dump_feature_map(feat_map_file)
        feature_processor.reset_processor()
    return 0


if __name__ == "__main__":
    exit(main())
