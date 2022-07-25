# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Takes as input a feature matrix and outputs which features are most used.

Usage:
    feature_matrix_statistics.py <train_data> <fmap>

Options:
    -h   Help
"""

from docopt import docopt
import logging
import os
import re
import numpy as np
import pickle

logger = logging.getLogger("matrix_statistics")

def setupLogger(logdir: str, loglevel: str) -> None:
    """
    Set up the logger instance, which will write its output to stderr.
    :param loglevel: Log level at which to record.
    """
    loglevel = logging.getLevelName(loglevel)
    logger.setLevel(loglevel)

    """ Enables logging to stderr """
    formatter = logging.Formatter('%(asctime)s :: %(name)s :: %(levelname)s :: %(message)s', datefmt="%Y-%m-%d-%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main() -> int:
    argv = None
    args = docopt(__doc__, argv=argv)

    setupLogger(".", "INFO")

    if args["<train_data>"] is not None:
        train_data = args["<train_data>"]
        if not os.path.exists(train_data):
            logger.error("Training data file does not exist.")
            return 1

        with open(args["<train_data>"], 'rb') as fd:
            sparse_mat = pickle.load(fd)

        feature_map = dict()
        with open(args["<fmap>"], 'r') as fd:
            for l in fd:
                mobj = re.search("([0-9]*) (.*) ", l)
                if mobj:
                    feature_map[int(mobj.group(1))] = mobj.group(2)

        all_counts = list()
        dtrain = sparse_mat.todense()
        print(f"Number of cookies total: {dtrain.shape[0]}")
        print(f"Number of features total: {dtrain.shape[1]}")
        for i in range(dtrain.shape[1]):
            count: int = np.count_nonzero(dtrain[:, i])
            all_counts.append((feature_map[i], count))


        for feat, c in sorted(all_counts, key=lambda x: x[1], reverse=True):
            print(f"{feat}, {c}")


if __name__ == "__main__":
    exit(main())