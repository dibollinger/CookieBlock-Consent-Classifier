# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Using the given trained XGB booster model, and the provided feature map, output additional
information on the booster, such as feature importance and graphs describing individual trees.

If no options specified, will produce default outputs

Usage:
    xgboost_stats.py <model> <feature_map> [--split_hist <fname>] [--plot <limit>] [--plot_tree <tree>] [--feat_cont <fvals>] [--predict <vdat>]

Options:
    -s --split_hist <fname>   Produce a split histogram for the specified feature name.
    -i --plot <limit>         Produce importance plots with specified limit on features.
    -t --plot_tree <tree>     Produce a graphviz plot for the specified tree, shows it immediately.
    -f --feat_cont <fvals>    Given the feature contributions file, will match with feature map to produce human-readable statistics. (sorted)
    -p --predict <vdat>       Predict for a specified dataset, then output statistics on this data.
    -h --help                 Show this help message.
"""

import xgboost as xgb
import matplotlib.pyplot as plt
import os
import sys
import logging

from datetime import datetime
from docopt import docopt
from utils import setupLogger

from typing import Dict, Union

logger = logging.getLogger("classifier.xgbstats")


def dump_importance(importances: Dict[str, Union[int, float]], filename: str) -> None:
    """
    Output feature importances to a csv file sorted descending.
    :param importances: Dictionary where the key is the feature name, and value is the importance.
    :param filename: Output filename
    """
    with open(filename, 'w') as fd:
        for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
            fd.write(f"{feat},{imp}\n")
    logger.info(f"Feature importance by weight dumped at: {filename}")


def plot_importance(filename: str, booster: xgb.Booster, fm_fp: str,
                    feat_limit: int, msm_type: str = "weight",
                    title: str = "Feature Importance by Weight",
                    xlabel: str = "Number of Appearances over all Trees") -> None:
    """
    Plot the feature importance as a pyplot bar chart, and write to disk.
    :param filename: Filename to output the graph to.
    :param booster: Booster to retrieve the importances from.
    :param fm_fp: Feature map, containing the names of the features.
    :param feat_limit: Number of features to display on the chart.
    :param msm_type: "weight", "gain", "cover", "total_gain" or "total_cover"
    :param title: Title for the chart
    :param xlabel: x-axis label
    """
    xgb.plot_importance(booster, grid=False, fmap=fm_fp, importance_type=msm_type,
                        max_num_features=feat_limit, title=title, xlabel=xlabel)

    plt.gcf().subplots_adjust(left=0.3)
    plt.savefig(filename, dpi=180)
    plt.cla()
    logger.info(f"Saved importance plot figure '{filename}' with type {msm_type}")


def plot_tree_and_show(booster: xgb.Booster, fm_fp: str, tree_index: int) -> None:
    """
    Plot a single tree of the booster using graphviz, and immediately displays it.
    :param booster: Booster to derive the tree from.
    :param fm_fp:  Feature names.
    :param tree_index: Identifies which tree to plot.
    """
    xgb.plot_tree(booster, fmap=fm_fp, num_trees=tree_index)
    plt.show()


def main() -> int:
    """
    Output additional statistics, such as feature importance and model dump.
    :return: exit code
    """
    argv = None
    cargs = docopt(__doc__, argv=argv)

    model_fp = cargs["<model>"]
    if not os.path.exists(model_fp):
        print(f"Model does not exist: {model_fp}", file=sys.stderr)
        return 2

    fm_fp = cargs["<feature_map>"]
    if not os.path.exists(fm_fp):
        print(f"Feature map does not exist: {fm_fp}", file=sys.stderr)
        return 2

    setupLogger(f"stats_xgb{''}.log")
    bst = xgb.Booster(model_file=model_fp)

    stats_path = f"xgb_booster_stats{''}/"
    os.makedirs(stats_path, exist_ok=True)

    # timestamp to prevent overwriting existing files
    timestamp: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    if cargs["--split_hist"]:
        # Show the split value histogram -- currently non-functional
        hist_array = bst.get_split_value_histogram(cargs["--split_hist"], fm_fp, as_pandas=True)
        logger.info(hist_array)
    elif cargs["--plot"]:
        # Plot the importance of weights, gain and cover
        f_limit = int(cargs["--plot"])
        plot_importance(os.path.join(stats_path, "importance_weights.png"), bst, fm_fp, f_limit)
        plot_importance(os.path.join(stats_path, "importance_gain.png"), bst, fm_fp, f_limit,
                        msm_type="gain", title="Feature Importance by Gain",
                        xlabel="Average Gain of Splits that use the Feature")
        plot_importance(os.path.join(stats_path, "importance_cover.png"), bst, fm_fp, f_limit,
                        msm_type="cover", title="Feature Importance by Cover",
                        xlabel="Average Number of Samples affected by Splits that use the Feature")

    elif cargs["--plot_tree"]:
        # Plot the tree and immediately show it
        plot_tree_and_show(bst, fm_fp, int(cargs["--plot_tree"]))
    elif cargs["--feat_cont"]:
        # This improves the feature contribution output of the training in order to match it with the feature names
        feature_cont_file = cargs["--feat_cont"]

        fmap_lines = [line.strip() for line in open(fm_fp, 'r').readlines()]
        feat_lines = [float(cont) for cont in open(feature_cont_file, 'r').readlines()[:-1]]
        assert len(fmap_lines) == len(feat_lines)

        matched = zip(fmap_lines, feat_lines)

        fcontrib_cleaned = os.path.join(stats_path, f"feature_contribs_{timestamp}.csv")
        with open(fcontrib_cleaned, 'w') as fw:
            for m in sorted(matched, key=lambda x: x[1], reverse=True):
                fw.write(m[0].split(sep=" ")[1] + f",{m[1]:.8f}\n")
        logger.info(f"Feature contribution mapping output at: '{fcontrib_cleaned}'")
    else:
        # 1. Output model dump
        path_to_model_dump = os.path.join(stats_path, f"model_dump_{timestamp}.json")
        bst.dump_model(path_to_model_dump, with_stats=False, dump_format="json")
        logger.info(f"Model dumped at: {path_to_model_dump}")

        # 2. Dump feature importance in text format
        dump_importance(bst.get_score(fmap=fm_fp, importance_type='weight'),
                        os.path.join(stats_path, f"importance_weight_{timestamp}.csv"))
        dump_importance(bst.get_score(fmap=fm_fp, importance_type='gain'),
                        os.path.join(stats_path, f"importance_gain_{timestamp}.csv"))
        dump_importance(bst.get_score(fmap=fm_fp, importance_type='cover'),
                        os.path.join(stats_path, f"importance_cover_{timestamp}.csv"))
        dump_importance(bst.get_score(fmap=fm_fp, importance_type='total_gain'),
                        os.path.join(stats_path, f"importance_totalgain_{timestamp}.csv"))
        dump_importance(bst.get_score(fmap=fm_fp, importance_type='total_cover'),
                        os.path.join(stats_path, f"importance_totalcover_{timestamp}.csv"))

        all_feats = set()
        with open(fm_fp, 'r') as fd:
            for l in fd:
                all_feats.add(l.strip().split()[1])

        imp = bst.get_score(fmap=fm_fp, importance_type='gain')
        for k in imp.keys():
            all_feats.remove(k)

        for f in all_feats:
            print(f)


    return 0


if __name__ == "__main__":
    exit(main())
