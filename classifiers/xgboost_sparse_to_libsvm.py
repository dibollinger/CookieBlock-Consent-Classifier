# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Given a sparse matrix, transform it to libsvm format.

Usage:
    xgboost_sparse_to_libsvm.py <sparse>
"""

import pickle
from sklearn.datasets import dump_svmlight_file
import sys

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Need argument")
        exit(1)

    fn = sys.argv[1]

    with open(fn, 'rb') as fd:
        csr_matrix = pickle.load(fd)
    with open(fn + ".labels", 'rb') as fd:
        labels = pickle.load(fd)

    dump_svmlight_file(csr_matrix, labels, fn[:-7] + ".libsvm")
