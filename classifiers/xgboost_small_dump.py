# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Given an XGBoost model, generates the JSON dump, and transforms it to reduce redundancy and lower the filesize.
This serves to be used inside the browser extension as a basis to generate the decision tree.

Usage:
    xgboost_small_dump.py <model>
"""


import xgboost as xgb
import os
import logging
import json

from docopt import docopt
from typing import List, Dict, Union

from utils import setupLogger

logger = logging.getLogger("classifier.minimal_dump")


def transform_node_dict(node:Dict):
    transf_dict = dict()
    if "leaf" in node:
        transf_dict["v"] = node["leaf"]
        return transf_dict
    else:
        transf_dict["f"] = node["split"]
        transf_dict["c"] = node["split_condition"]
        transf_dict["u"] = "l" if node["missing"] == node["yes"] else "r"
        assert len(node["children"]) == 2, f"Node did not have exactly 2 children. Actual amount: {len(node['children'])}"
        if node["children"][0]["nodeid"] == node["yes"]:
            transf_dict["l"] = transform_node_dict(node["children"][0])
            transf_dict["r"] = transform_node_dict(node["children"][1])
        else:
            transf_dict["l"] = transform_node_dict(node["children"][1])
            transf_dict["r"] = transform_node_dict(node["children"][0])
        return transf_dict


def main() -> int:
    argv = None
    cargs = docopt(__doc__, argv=argv)
    setupLogger(None)

    model_fp = cargs["<model>"]
    if not os.path.exists(model_fp):
        logger.error(f"Model does not exist: {model_fp}")
        return 2

    bst = xgb.Booster(model_file=model_fp)

    tmp_path = "./model_dmp.tmp"
    bst.dump_model(tmp_path, with_stats=False, dump_format="json")
    with open(tmp_path, 'r') as fd:
        mdl_dump_json: List[Dict[str, Union[float, List[Dict]]]] = json.load(fd)
    os.unlink(tmp_path)

    dump_path = f"minimal_dump{''}/"
    os.makedirs(dump_path, exist_ok=True)

    num_classes = 4
    for i in range(num_classes):
        new_forest = list()
        class_forest = [mdl_dump_json[j] for j in range(len(mdl_dump_json)) if j % num_classes == i]
        for tree in class_forest:
            new_forest.append(transform_node_dict(tree))

        r_path = os.path.join(dump_path, f"forest_class{i}.json")
        with open(r_path, 'w') as fw:
            json.dump(new_forest, fw)

        logger.info(f"Minified model dump output to {r_path}")

    return 0


if __name__ == "__main__":
    exit(main())
