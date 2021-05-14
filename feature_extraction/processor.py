# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
This file contains the class responsible for transforming cookie json data into feature vectors.
The concept is as follows:
   * Each individual feature with number of slots taken up by it, is defined inside the file 'features.json'.
   * Attributes "function", "vector_size", "enabled" and "args" are required.
      - "function" defines the corresponding function name inside the CookieFeatureProcessor.
      - "vector_size" defines the size of the sub-vector taken up by this feature inside the full array.
        For instance, if the feature creates a one-hot vector of size 100, this value needs to be 100.
      - "enabled" toggles whether the feature is used in the transformation.
      - "args" defines additional keyword arguments to be used by the feature extraction function.
   * In addition, two additional arguments may be defined: "setup" and "source"
      - "setup" defines a secondary setup function that is executed once when the processor is initialized.
        This is intended for situations where, for example, a lookup table needs to be initialized.
      - "source" defines the path for an external data resource, used by the setup function.
      - Any additional arguments to the setup functions need to be defined inside "args".
   * The order in the JSON is ultimately the order in which the features appear in the vector.
     This should not have an effect on training the classifier, but it is important to know for the feature map.
   * The features are split up into 3 arrays:
      - per-cookie features: Extracted once for each cookie. These features utilize values that remain constant
                             throughout the cookie's lifetime. These are transformed first.
      - per-update features: Extracted once for each cookie update. Involve values that may change in each update.
                             These are transformed second, and we have as many as defined by the cookie updates parameter.
      - per-diff features: Extracted once for each sequential pair of cookie updates, ordered by timestamp.
                           Utilize the values of two cookies, specifically the difference between them.
                           Transformed third. As many as (num_updates - 1) features are extracted for these.
   * Finally there is also an attribute "num_updates" which define how many updates at most are considered for the
     extraction of per-update features.
"""

# Essential imports required to extract features
import base64
import csv
import json

import re
from statistics import mean, stdev
import urllib.parse
import zlib
from collections import Counter
from math import log

import scipy.sparse
from sklearn.datasets import dump_svmlight_file
import xgboost as xgb
import pycountry
import difflib
from Levenshtein import distance as lev_distance

from utils import (load_lookup_from_csv, url_to_uniform_domain, split_delimiter_separated,
                   check_flag_changed, try_decode_base64, try_split_json, delim_sep_check)

# Non-essential
import logging
import time
import pickle

from tqdm import tqdm
from typing import Tuple, List, Dict, Any, Optional, Set

logger = logging.getLogger("feature-extract")


class CookieFeatureProcessor:

    def __init__(self, feature_parameter_source: str, skip_cmp_cookies: bool) -> None:
        """
        The feature processor tracks extracted features using sparse vector representation. This is chosen because
        one-hot vectors are utilized, in addition to a fixed-size update list, where missing updates are represented by 0.

        The features to be extracted are defined inside "features.json". The setup loads these features and  initializes
        the internal data structures necessary to track the addition of sparse features.
        :param feature_parameter_source: Feature definition
        :param skip_cmp_cookies: Whether to skip CMP-specific cookies. These may bias training. (Cookiebot, OneTrust)
        """

        # Initialize features from json file
        with open(feature_parameter_source) as fmmap:
            self.feature_mapping: Dict[str, Any] = json.load(fmmap)

        # Initialize a single global csv sniffer, in order to find the correct delimiter for CSV data
        self.csv_sniffer: csv.Sniffer = csv.Sniffer()

        # cookie names specific to one of the consent management platforms we crawled
        # these are over-represented as they are present on nearly every crawled website of the training data
        self._skipped_names: Optional[re.Pattern] = re.compile("OptanonConsent|CookieConsent") if skip_cmp_cookies else None

        # compute the expected number of features based on the feature mapping
        self.num_cookie_features: int = 0
        funcs = 0
        for f in self.feature_mapping["per_cookie_features"]:
            funcs += 1
            if f["enabled"]:
                self.num_cookie_features += f["vector_size"]
        logger.info(f"Number of per-cookie functions: {funcs}")

        self.num_update_features: int = 0
        funcs = 0
        for f in self.feature_mapping["per_update_features"]:
            funcs += 1
            if f["enabled"]:
                self.num_update_features += f["vector_size"] * self.feature_mapping["num_updates"]
        logger.info(f"Number of per-update functions: {funcs}")

        self.num_diff_features: int = 0
        funcs = 0
        for f in self.feature_mapping["per_diff_features"]:
            funcs += 1
            if f["enabled"]:
                self.num_diff_features += f["vector_size"] * (self.feature_mapping["num_updates"] - 1)
        logger.info(f"Number of per-diff functions: {funcs}")

        self.num_features: int = (self.num_cookie_features + self.num_update_features + self.num_diff_features)

        # tracks the current features in sparse representation
        self._row_indices: List[int] = list()
        self._col_indices: List[int] = list()
        self._data_entries: List[float] = list()
        self._labels: List[int] = list()

        # cursor for sparse features
        self._current_row: int = 0
        self._current_col: int = 0

        # Lookup table: Name -> Rank
        self._top_name_lookup: Optional[Dict[str, int]] = None
        # Lookup table: Domain -> Rank
        self._top_domain_lookup: Optional[Dict[str, int]] = None
        # List: Tuple of cookie name pattern and corresponding rank
        self._pattern_names: Optional[List[Tuple[re.Pattern, int]]] = None
        # List: Tuple of cookie name features and corresponding rank
        self._name_features: Optional[List[Tuple[re.Pattern, int]]] = None
        # List: Tuple of content term, and corresponding rank
        self._content_terms: Optional[List[Tuple[re.Pattern, int]]] = None
        # Set: IAB Vendor domains
        self._iab_europe_vendors: Optional[Set[str]] = None

        # This set is required to limit false positives. These are all the separators recognized as valid
        self.valid_csv_delimiters: str = ",|#:;&_"

        # Strings that identify boolean values.
        self.truth_values: re.Pattern = re.compile(r"\b(true|false|yes|no|0|1|on|off)\b", re.IGNORECASE)

        # One large set that contains strings that relate to country, currency or language
        self.locale_lookup: Set = set()
        self.locale_lookup.update({c.name for c in pycountry.countries})
        self.locale_lookup.update({c.alpha_2 for c in pycountry.countries})
        self.locale_lookup.update({c.alpha_3 for c in pycountry.countries})
        self.locale_lookup.update({c.name for c in pycountry.currencies})
        self.locale_lookup.update({c.alpha_3 for c in pycountry.currencies})
        self.locale_lookup.update({c.name for c in pycountry.languages})
        self.locale_lookup.update({c.alpha_3 for c in pycountry.languages})

        # Horrible Date Regexes
        self.pattern_year_month_day: re.Pattern = re.compile("(19[7-9][0-9]|20[0-3][0-9]|[0-9][0-9])-[01][0-9]-[0-3][0-9]")
        self.pattern_day_month_year: re.Pattern = re.compile("[0-3][0-9]-[01][0-9]-(19[7-9][0-9]|20[0-3][0-9]|[0-9][0-9])")
        self.pattern_month_day_year: re.Pattern = re.compile("[01][0-9]-[0-3][0-9]-(19[7-9][0-9]|20[0-3][0-9])")

        # Day and Month identifiers
        self.pattern_alpha3_days_eng: re.Pattern = re.compile("(Mon|Tue|Wed|Thu|Fri|Sat|Sun)", re.IGNORECASE)
        self.pattern_alpha3_months_eng: re.Pattern = re.compile("(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.IGNORECASE)
        self.pattern_full_days_eng: re.Pattern = re.compile("(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", re.IGNORECASE)
        self.pattern_full_months_eng: re.Pattern = re.compile("(January|February|March|April|May|June|July|August|September|October|November|December)", re.IGNORECASE)

        # Other patterns
        self.pattern_id_string: re.Pattern = re.compile("(id|ident)", re.IGNORECASE)
        self.pattern_timestamp: re.Pattern = re.compile("16[0-9]{8}([0-9]{3})?")
        self.pattern_canon_uuid: re.Pattern = re.compile("[0-9a-f]{8}-[0-9a-f]{4}-([0-9a-f])[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
        self.pattern_noncanon_uuid: re.Pattern = re.compile("[0-9a-f]+-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+-[0-9a-f]+", re.IGNORECASE)
        self.pattern_http: re.Pattern = re.compile("http(s)?://.*\.")
        self.pattern_www: re.Pattern = re.compile("www(2-9)?\..*\.")
        self.pattern_hexstring: re.Pattern = re.compile("^[0-9a-f]+$", re.IGNORECASE) # want exact match
        self.js_pat: re.Pattern = re.compile("^\{.*}$")


        # Setup external resources through CSV and TXT file data
        logger.debug("Setting up lookup data...")
        all_features_list: List[Dict[str, Any]] = (self.feature_mapping["per_cookie_features"]
                                                   + self.feature_mapping["per_update_features"]
                                                   + self.feature_mapping["per_diff_features"])
        for feature in all_features_list:
            if feature["enabled"] and "setup" in feature:
                assert hasattr(self, feature["setup"]), f"Setup function not found: {feature['setup']}"
                logger.debug(f"Running setup method: {feature['setup']}")
                function = getattr(self, feature["setup"])
                function(source=feature["source"], vector_size=feature["vector_size"], **feature["args"])

        logger.debug("Lookup setup complete.")

    #
    ## Internal Data Handling
    ## Methods used to construct the sparse matrix representation
    #

    def _reset_col(self) -> None:
        """ Reset column position and verify feature vector size. """
        assert self.num_features == self._current_col, f"Inconsistent Feature Count {self.num_features} and {self._current_col}"
        self._current_col = 0

    def _increment_row(self, amount: int = 1) -> int:
        """ Each row of the matrix stores features for a single cookie instance (including all updates).
            :param amount: By how much to shift the cursor
        """
        self._current_row += amount
        return self._current_row

    def _increment_col(self, amount: int = 1) -> int:
        """ Increment the internal column counter, i.e. change feature index.
            :param amount: By how much to shift the cursor
        """
        self._current_col += amount
        return self._current_col

    def _insert_label(self, label: int) -> None:
        """ Append label to the internal listing.
        :param label: Label to append, as integer.
        """
        self._labels.append(label)

    def _multi_insert_sparse_entries(self, data: List[float], col_offset: int = 0) -> None:
        """
        Insert multiple sparse entries -- required in certain cases
        :param data: Floating point entries to insert into the sparse representation.
        :param col_offset: By how many entries to offset the insertion from the current cursor.
        """
        c = 0
        for d in data:
            self._row_indices.append(self._current_row)
            self._col_indices.append(self._current_col + col_offset + c)
            self._data_entries.append(d)
            c += 1

    def _insert_sparse_entry(self, data: float, col_offset: int = 0) -> None:
        """
            Updates sparse representation arrays with the provided data.
            :param data: Data entry to insert into the sparse matrix.
            :param col_offset: Used when position of one-hot vector is shifted from current cursor.
        """
        self._row_indices.append(self._current_row)
        self._col_indices.append(self._current_col + col_offset)
        self._data_entries.append(data)

    ##
    ## Outwards-facing methods:
    ##

    def reset_processor(self) -> None:
        """ Reset all data storage -- to be used once a matrix is fully constructed, and another needs to be generated. """
        self._row_indices.clear()
        self._col_indices.clear()
        self._data_entries.clear()
        self._labels.clear()
        self._current_col = 0
        self._current_row = 0

    def retrieve_labels(self) -> List[int]:
        """ Get a copy of the current label list. """
        return self._labels.copy()


    def retrieve_label_weights(self, num_labels: int) -> List[float]:
        """
        Compute weights from the label array in order to counter class imbalance.
        Assumption: Labels start from 0, up to num_labels.
        :param num_labels: Maximum label index. Final index ranges from 0 to num_labels.
        :return: Inverse frequency of each label.
        """
        num_total = len(self._labels)
        inverse_ratio = [num_total / self._labels.count(i) for i in range(num_labels)]
        logger.info(f"Computed Weights: {inverse_ratio}")
        return [inverse_ratio[lab] for lab in self._labels]


    def retrieve_feature_names_as_list(self) -> List[str]:
        """Retrieve the list of feature names in a sequential list"""
        feat_list = []
        feat_cnt = 0
        for feature in self.feature_mapping["per_cookie_features"]:
            if feature["enabled"]:
                for i in range(feature["vector_size"]):
                    feat_list.append(str(feat_cnt + i) + " " + feature["name"] + f"-{i} i")
                feat_cnt += feature["vector_size"]
        for feature in self.feature_mapping["per_update_features"]:
            if feature["enabled"]:
                for u in range(self.feature_mapping["num_updates"]):
                    for i in range(feature["vector_size"]):
                        feat_list.append(str(feat_cnt + i) + f" update_{u}_" + feature["name"] + f"-{i} i")
                    feat_cnt += feature["vector_size"]
        for feature in self.feature_mapping["per_diff_features"]:
            if feature["enabled"]:
                for u in range(self.feature_mapping["num_updates"] - 1):
                    for i in range(feature["vector_size"]):
                        feat_list.append((str(feat_cnt + i) + f" diff_{u}_" + feature["name"] + f"-{i} i"))
                    feat_cnt += feature["vector_size"]

        return feat_list


    def retrieve_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """ From the collected data, construct a CSR format sparse matrix using scipy. """
        assert len(self._data_entries) > 0, "No features stored by processor!"
        return scipy.sparse.csr_matrix((self._data_entries, (self._row_indices, self._col_indices)))

    def retrieve_xgb_matrix(self, include_labels: bool, include_weights: bool) -> xgb.DMatrix:
        """
        From the collected data, construct a xgb binary-format matrix.
        :param include_labels: If true, will include labels inside the binary.
        :param include_weights: If true, will include weights for each label inside the binary.
        :return: XGB DMatrix
        """
        assert len(self._data_entries) > 0, "No features stored by processor!"
        assert (not include_labels and not include_weights) or len(self._labels) > 0, "No labels stored by processor!"
        sparse_mat: scipy.sparse.csr_matrix = self.retrieve_sparse_matrix()
        labels: Optional[List[int]] = self.retrieve_labels() if include_labels else None
        weights: Optional[List[float]] = self.retrieve_label_weights(num_labels=4) if include_weights else None
        return xgb.DMatrix(sparse_mat, label=labels, weight=weights, feature_names=self.retrieve_feature_names_as_list())

    def dump_sparse_matrix(self, out_path: str, dump_weights: bool = True) -> None:
        """
        Dump the sparse matrix of features extracted from the cookies.
        :param out_path: filename for the pickled sparse matrix
        :param dump_weights: if true, will also dump the instance weights
        """
        dtrain = self.retrieve_sparse_matrix()
        with open(out_path, 'wb') as fd:
            pickle.dump(dtrain, fd)

        feature_names = self.retrieve_feature_names_as_list()
        with open(out_path + ".feature_names", 'wb') as fd:
            pickle.dump(feature_names, fd)

        labels = self.retrieve_labels()
        with open(out_path + ".labels", 'wb') as fd:
            pickle.dump(labels, fd)

        if dump_weights and len(Counter(labels).keys()) == 4:
            weights = self.retrieve_label_weights(num_labels=4)
            with open(out_path + ".weights", 'wb') as fd:
                pickle.dump(weights, fd)

    def dump_libsvm(self, path: str, dump_weights: bool = True) -> None:
        """ Dump the collected data to the specified path as a libsvm file """
        sparse = self.retrieve_sparse_matrix()
        labels = self.retrieve_labels()
        dump_svmlight_file(sparse, labels, path)

        feature_names = self.retrieve_feature_names_as_list()
        with open(path + ".feature_names", 'wb') as fd:
            pickle.dump(feature_names, fd)

        if dump_weights and len(Counter(labels).keys()) == 4:
            weights = self.retrieve_label_weights(num_labels=4)
            with open(path + ".weights", 'wb') as fd:
                pickle.dump(weights, fd)

    def retrieve_debug_output(self) -> List[Dict[str, float]]:
        """
        Retrieve JSON pretty printed data to verify that the features are transformed correctly.
        """
        feature_names: List[str] = self.retrieve_feature_names_as_list()
        csr_mat = self.retrieve_sparse_matrix()
        matrix = csr_mat.todense()
        assert matrix.shape[1] <= len(feature_names), f"Number of columns exceeds number of features: Matrix: {matrix.shape[1]} -- Features: {len(feature_names)}"

        numerical_dict_features: List[Dict[str, float]] = list()
        for i in range(matrix.shape[0]):
            numerical_dict_features.append(dict())
            for j in range(matrix.shape[1]):
                numerical_dict_features[i][feature_names[j]] = matrix[i, j]

        return numerical_dict_features

    def print_feature_info(self) -> None:
        """Output information on the features """
        logger.info(f"Number of Per-Cookie Features: {self.num_cookie_features}")
        logger.info(f"Number of Per-Update Features: {self.num_update_features}")
        logger.info(f"Number of Per-Diff Features: {self.num_diff_features}")
        logger.info(f"Number of Features Total: {self.num_features}")

    def dump_feature_map(self, filename: str) -> None:
        """
        Produces a named feature map for use with XGBoost.
        :param filename: feature map filename
        """
        with open(filename, 'w') as fd:
            flist = self.retrieve_feature_names_as_list()
            for f in flist:
                fd.write(f + "\n")
        logger.info(f"Extracted xgboost feature map to {filename}")

    def extract_features(self, input_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Extract cookie data from given input dictionary and apply featue extraction methods.
        Does not skip CMP-specific names, and does not expect any training labels in the dataset.
        Furthermore, it skips debug timing and other similar tasks. Intended for transforming test data for classification.
        :param input_data: Cookie data to transform
        """
        for entry_name, entry_values in input_data.items():
            # Extract features from cookie data that is consistent across all updates.
            # This includes name, domain, path and first-party domain
            for feature in self.feature_mapping["per_cookie_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Defined per-cookie function not found: {feature['function']}"
                    getattr(self, feature["function"])(entry_values, **feature["args"])
                    self._increment_col(feature["vector_size"])

            # Extract features from cookie data that may change with each update.
            # This includes flags such as host_only, http_only, and content.
            for feature in self.feature_mapping["per_update_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Per-update function not found: {feature['function']}"
                    function = getattr(self, feature["function"])
                    v_iter = iter(entry_values["variable_data"])

                    # Iterate until maximum update number reached, or out of updates
                    update_count = 0
                    try:
                        while update_count < self.feature_mapping["num_updates"]:
                            var_data = next(v_iter)
                            function(var_data, **feature["args"])
                            self._increment_col(feature["vector_size"])
                            update_count += 1
                    except StopIteration:
                        # if out of updates, need to increment column counter so size is uniform
                        empty_update_slots = feature["vector_size"] * (self.feature_mapping["num_updates"] - update_count)
                        self._increment_col(empty_update_slots)

            # Extract features from the difference between cookie updates.
            # This includes differences in expiry time and content, for instance.
            for feature in self.feature_mapping["per_diff_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Difference function not found: {feature['function']}"
                    v_iter = iter(entry_values["variable_data"])
                    function = getattr(self, feature["function"])

                    # Iterate until maximum number of updates is reached, or until we are out of updates
                    update_count = 0
                    try:
                        prev_update = next(v_iter)
                        while update_count < self.feature_mapping["num_updates"] - 1:
                            curr_update = next(v_iter)
                            function(prev_update, curr_update, **feature["args"])
                            self._increment_col(feature["vector_size"])
                            prev_update = curr_update
                            update_count += 1
                    except StopIteration:
                        # if out of updates, need to increment column counter so size is uniform
                        empty_update_slots = feature["vector_size"] * (self.feature_mapping["num_updates"] - update_count - 1)
                        self._increment_col(empty_update_slots)

            # before moving to the next cookie entry, reset the column index and move to the next row
            self._reset_col()
            self._increment_row()

    def extract_features_with_labels(self, input_data: Dict[str, Dict[str, Any]]) -> None:
        """
        Intended for the training data feature extraction. Expects labels in the input dictionary.
        Filters unwanted labels and unwanted names too.
        Performs timing measurements to analyze the feature extraction performance.
        :param input_data: Cookie training data to transform, with labels.
        """
        timings_per_function: Dict[str, List] = dict()

        ctr_label_skipped: int = 0
        ctr_cmp_cookie_skipped: int = 0

        logger.info("Begin feature extraction process...")
        start = time.perf_counter()
        for entry_name, entry_values in tqdm(input_data.items()):
            # retrieve the label and skip ones we don't want
            category_label = int(entry_values["label"])

            # Make sure we only consider desired labels
            if not (0 <= category_label <= 3):
                ctr_label_skipped += 1
                continue

            # filter out specific cookie names
            if self._skipped_names and self._skipped_names.match(entry_values["name"]):
                ctr_cmp_cookie_skipped += 1
                continue

            # append the label to the list
            self._insert_label(category_label)

            # Extract features from cookie data that is consistent across all updates.
            # This includes name, domain, path and first-party domain
            for feature in self.feature_mapping["per_cookie_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Defined per-cookie function not found: {feature['function']}"

                    if feature["function"] not in timings_per_function:
                        timings_per_function[feature["function"]] = list()

                    function = getattr(self, feature["function"])

                    t_start = time.perf_counter_ns()
                    function(entry_values, **feature["args"])
                    timings_per_function[feature["function"]].append(time.perf_counter_ns() - t_start)

                    self._increment_col(feature["vector_size"])


            # Extract features from cookie data that may change with each update.
            # This includes flags such as host_only, http_only, and content.
            for feature in self.feature_mapping["per_update_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Per-update function not found: {feature['function']}"

                    if feature["function"] not in timings_per_function:
                        timings_per_function[feature["function"]] = list()

                    function = getattr(self, feature["function"])
                    v_iter = iter(entry_values["variable_data"])

                    # Iterate until maximum update number reached, or out of updates
                    update_count: int = 0
                    try:
                        while update_count < self.feature_mapping["num_updates"]:
                            var_data = next(v_iter)
                            t_start = time.perf_counter_ns()
                            function(var_data, **feature["args"])
                            timings_per_function[feature["function"]].append(time.perf_counter_ns() - t_start)
                            self._increment_col(feature["vector_size"])
                            update_count += 1
                    except StopIteration:
                        # if out of updates, need to increment column counter so size is uniform
                        empty_update_slots = feature["vector_size"] * (self.feature_mapping["num_updates"] - update_count)
                        self._increment_col(empty_update_slots)

            # Extract features from the difference between cookie updates.
            # This includes differences in expiry time and content, for instance.
            for feature in self.feature_mapping["per_diff_features"]:
                if feature["enabled"]:
                    assert hasattr(self, feature["function"]), f"Difference function not found: {feature['function']}"
                    v_iter = iter(entry_values["variable_data"])
                    function = getattr(self, feature["function"])

                    if feature["function"] not in timings_per_function:
                        timings_per_function[feature["function"]] = list()

                    # Iterate until maximum number of updates is reached, or until we are out of updates
                    update_count = 0
                    try:
                        prev_update = next(v_iter)
                        while update_count < self.feature_mapping["num_updates"] - 1:
                            curr_update = next(v_iter)
                            t_start = time.perf_counter_ns()
                            function(prev_update, curr_update, **feature["args"])
                            timings_per_function[feature["function"]].append(time.perf_counter_ns() - t_start)
                            self._increment_col(feature["vector_size"])
                            prev_update = curr_update
                            update_count += 1
                    except StopIteration:
                        # if out of updates, need to increment column counter so size is uniform
                        empty_update_slots = feature["vector_size"] * (self.feature_mapping["num_updates"] - update_count - 1)
                        self._increment_col(empty_update_slots)

            # before moving to the next cookie entry, reset the column index and move to the next row
            self._reset_col()
            self._increment_row()

        end = time.perf_counter()
        total_time_taken: float = end - start
        logger.info(f"Feature extraction completed. Final row position: {self._current_row}")

        logger.info("Timings per feature:")
        total_time_spent = 0
        for func, t_list in sorted(timings_per_function.items(), key=lambda x: sum(x[1]), reverse=True):
            if len(t_list) == 0:
                continue
            else:
                time_spent = sum(t_list)
                total_time_spent += time_spent
                logmsg = (f"total:{sum(t_list) / 1e9:.3f} s"
                          f"|{sum(t_list) / (1e7 * total_time_taken):.3f}%"
                          f"|mean: {mean(t_list):.2f} ns|max: {max(t_list)} ns")
                if len(t_list) >= 2:
                    logmsg += f"|stdev: {stdev(t_list):.2f} ns"
                logmsg += f"|{func}"
                logger.info(logmsg)
        logger.info(f"Total time spent in feature extraction: {total_time_spent / 1e9:.3f} seconds")
        logger.info(f"Time lost to overhead: {total_time_taken - (total_time_spent / 1e9):.3f} seconds")
        logger.info(f"Num social media category skipped: {ctr_label_skipped}")
        logger.info(f"Num CMP-specific cookies skipped: {ctr_cmp_cookie_skipped}")

    #
    ## Setup methods for external resources
    #

    def setup_top_names(self, source: str, vector_size: int) -> None:
        """
        Set up the lookup table to determine whether the cookie name is in
        the top k names from the provided source ranking.
        The source ranking is assumed to be sorted in advance.
        :param source: Path to source ranking
        :param vector_size: Top k names to include in the lookup table
        """
        self._top_name_lookup = load_lookup_from_csv(source, vector_size)

    def setup_top_domains(self, source: str, vector_size: int) -> None:
        """
        Set up the lookup table to determine whether the cookie domain is in
        the top k domains from the provided source ranking.
        The source ranking is assumed to be sorted in advance.
        :param source: Path to source ranking
        :param vector_size: Top k domains to include in the lookup table
        """
        self._top_domain_lookup = load_lookup_from_csv(source, vector_size)

    def setup_pattern_names(self, source: str, vector_size: int) -> None:
        """
        Some cookie names may be part of a series of pattern cookies, all belonging to the same group.
        This method loads the specified number of regex patterns from the source file.
        :param source: source for the regular expression patterns, sorted in advance
        :param vector_size: number of patterns to extract
        """
        self._pattern_names = list()
        rank: int = 0
        with open(source) as fd:
            line = next(fd)
            try:
                while rank < vector_size:
                    pattern = line.strip().split(',')[-1]
                    self._pattern_names.append((re.compile(pattern), rank))
                    rank += 1
                    line = next(fd)
            except StopIteration:
                raise RuntimeError(f"Not enough patterns in file. Expected at least {vector_size}, max is {rank}.")

    def setup_name_features(self, source: str, vector_size: int) -> None:
        """
        Loads a list of name features which may be present inside the cookie name.
        :param source: Path to the source file
        :param vector_size: Number of features to use.
        """
        self._name_features: List[Tuple[re.Pattern, int]] = list()
        rank = 0
        with open(source) as fd:
            line = next(fd)
            try:
                while rank < vector_size:
                    pattern = line.strip().split(',')[-1]
                    self._name_features.append((re.compile(pattern), rank))
                    rank += 1
                    line = next(fd)
            except StopIteration:
                raise RuntimeError(f"Not enough patterns in file. Expected at least {vector_size}, max is {rank}.")

    def setup_iabeurope_vendors(self, source: str, vector_size: int) -> None:
        """
        Load IAB vendor domains, store as a set for lookup purposes.
        :param source: path to the text file resource
        :param vector_size: Unused
        """
        self._iab_europe_vendors: Set[str] = set()
        with open(source) as fd:
            for line in fd:
                normalized_domain = url_to_uniform_domain(line.strip())
                self._iab_europe_vendors.add(normalized_domain)

    def setup_content_terms(self, source: str, vector_size: int) -> None:
        """
        Loads a list of terms which may be present inside the cookie content.
        :param source: Path to the source file
        :param vector_size: Number of features to use.
        """
        self._content_terms: List[Tuple[re.Pattern, int]] = list()
        rank = 0
        with open(source) as fd:
            line = next(fd)
            try:
                while rank < vector_size:
                    pattern = line.strip().split(',')[-1]
                    self._content_terms.append((re.compile(pattern), rank))
                    rank += 1
                    line = next(fd)
            except StopIteration:
                raise RuntimeError(f"Not enough patterns in file. Expected at least {vector_size}, max is {rank}.")

    #
    ## Per Cookie Features
    #

    def feature_top_names(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        This feature detects whether the cookie name is part of the top K cookie names
        from the external resource document, and constructs a K-sized one-hot vector feature.
        :param cookie_features: Dictionary containing key "name" of the cookie
        """
        assert self._top_name_lookup is not None, "Top N name lookup was not set up prior to feature extraction!"
        cookie_name: str = cookie_features["name"]
        if cookie_name in self._top_name_lookup:
            rank = self._top_name_lookup[cookie_name]
            self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_top_domains(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        This feature function detects whether the cookie name is part of the top K cookie domains
        from the external resource document, and constructs a K-sized one-hot vector feature.
        :param cookie_features: Dictionary containing key "domain" of the cookie
        """
        assert (self._top_domain_lookup is not None), "Top N domain lookup was not set up prior to feature extraction!"
        cookie_domain: str = url_to_uniform_domain(cookie_features["domain"])
        if cookie_domain in self._top_domain_lookup:
            rank = self._top_domain_lookup[cookie_domain]
            self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_pattern_names(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        This feature detects whether the given cookie is part of one of the K selected cookie name patterns.
        A K-sized one-hot vector, with the corresponding occurrence set, is inserted as a result.
        :param cookie_features: Dictionary containing key "name" of the cookie
        """
        assert (self._pattern_names is not None), "Top N pattern lookup was not set up prior to feature extraction!"
        cookie_name: str = cookie_features["name"]
        for pattern, rank in self._pattern_names:
            if pattern.match(cookie_name):
                self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_name_tokens(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        Using a pre-selected set of name features, inserts a one-hot vector indicating whether said feature is
        present inside the vector. Multiple entries may be set to 1.0 by this. Features assumed to be sorted with rank.
        :param cookie_features: Dictionary containing key "name" of the cookie
        """
        assert (self._name_features is not None), "Top N name features map was not set up prior to feature extraction!"
        for token, rank in self._name_features:
            if token.search(cookie_features["name"]):
                self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_iab_vendor(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        Binary check if the given cookie domain is part of the IAB Europe vendors list, which are mostly
        third-party advertisers that are involved with IAB's TCF. Cookiebot and Onetrust in particular are
        CMPs that use the TCF framework for exchanging cookie data with third party vendors.
        :param cookie_features: Dictionary containing key "domain" of the cookie
        """
        assert (self._iab_europe_vendors is not None), "IAB Europe Vendors Set not initialized!"
        sanit_domain = url_to_uniform_domain(cookie_features["domain"])
        if sanit_domain in self._iab_europe_vendors:
            self._insert_sparse_entry(1.0)

    def feature_is_third_party(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        Single feature entry, inserts 1.0 if the cookie is a third-party cookie.
        :param cookie_features: Dictionary containing keys "domain" and "first_party_domain"
        """
        cookie_domain = url_to_uniform_domain(cookie_features["domain"])
        website_domain = url_to_uniform_domain(cookie_features["first_party_domain"])
        if cookie_domain not in website_domain:
            self._insert_sparse_entry(1.0)

    def feature_non_root_path(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        Single feature entry, inserts 1.0 if the cookie has a path other than "/".
        :param cookie_features: Dictionary containing key "path"
        """
        if cookie_features["path"].strip() != "/":
            self._insert_sparse_entry(1.0)

    def feature_update_count(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        Inserts the total number of cookie updates as a feature. Minimum is 1.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        self._insert_sparse_entry(len(cookie_features["variable_data"]))

    def feature_http_only_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Inserts 1 if throughout all updates for the cookie, the http_only flag changed at least once.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if check_flag_changed(cookie_features["variable_data"], "http_only"):
            self._insert_sparse_entry(1.0)


    def feature_secure_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Inserts 1 if throughout all updates for the cookie, the secure flag changed at least once.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if check_flag_changed(cookie_features["variable_data"], "secure"):
            self._insert_sparse_entry(1.0)

    def feature_same_site_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Inserts 1 if throughout all updates for the cookie, the same_site flag changed at least once.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if check_flag_changed(cookie_features["variable_data"], "same_site"):
            self._insert_sparse_entry(1.0)

    def feature_is_session_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Inserts 1 if throughout all updates for the cookie, the session flag changed at least once.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if check_flag_changed(cookie_features["variable_data"], "session"):
            self._insert_sparse_entry(1.0)


    def feature_gestalt_mean_and_stddev(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Over all updates, compute the mean ratcliff-obershelp similarity between cookie updates,
        plus standard deviation.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        diffs: List[float] = list()
        v_iter = iter(cookie_features["variable_data"])
        previous_entry = next(v_iter)  # first entry always non-null
        try:
            while previous_entry is not None:
                curr_entry = next(v_iter)
                if curr_entry is not None:
                    matcher = difflib.SequenceMatcher(a=previous_entry["value"], b=curr_entry["value"])
                    diffs.append(matcher.ratio())
                previous_entry = curr_entry
        except StopIteration:
            pass

        # Append mean of all update diffs
        if len(diffs) > 0:
            self._insert_sparse_entry(mean(diffs), col_offset=0)
        else:
            self._insert_sparse_entry(-1.0, col_offset=0)

        # Append standard deviation
        if len(diffs) > 1:
            self._insert_sparse_entry(stdev(diffs), col_offset=1)
        else:
            self._insert_sparse_entry(-1.0, col_offset=1)


    def feature_levenshtein_mean_and_stddev(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie update
        Over all updates, compute the mean levenshtein between cookie update contents,
        plus standard deviation.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        diffs: List[float] = list()
        v_iter = iter(cookie_features["variable_data"])
        previous_entry = next(v_iter)  # first entry always non-null
        try:
            while previous_entry is not None:
                curr_entry = next(v_iter)
                if curr_entry is not None:
                    lev_metric = lev_distance(previous_entry["value"], curr_entry["value"])
                    diffs.append(lev_metric)
                previous_entry = curr_entry
        except StopIteration:
            pass

        # Append mean of all update diffs
        if len(diffs) > 0:
            self._insert_sparse_entry(mean(diffs), col_offset=0)
        else:
            self._insert_sparse_entry(-1.0, col_offset=0)

        # Append standard deviation
        if len(diffs) > 1:
            self._insert_sparse_entry(stdev(diffs), col_offset=1)
        else:
            self._insert_sparse_entry(-1.0, col_offset=1)


    def feature_content_length_mean_and_stddev(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie update
        Over all updates, compute the mean content length plus standard deviation.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        content_lengths: List[int] = list()
        for cookie_update in cookie_features["variable_data"]:
            c_size = len(bytes(cookie_update["value"].encode("utf-8")))
            content_lengths.append(c_size)

        # mean
        if len(content_lengths) > 0:
            self._insert_sparse_entry(mean(content_lengths), col_offset=0)
        else:
            self._insert_sparse_entry(-1.0, col_offset=0)

        # stddev
        if len(content_lengths) > 1:
            self._insert_sparse_entry(stdev(content_lengths), col_offset=1)
        else:
            self._insert_sparse_entry(-1.0, col_offset=1)


    def feature_compressed_length_mean_and_stddev(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie update
        Over all updates, compute the mean compressed length plus standard deviation.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        compressed_lengths: List[int] = list()
        for cookie_update in cookie_features["variable_data"]:
            unquoted_content = urllib.parse.unquote(cookie_update["value"])
            content_bytes = bytes(unquoted_content.encode("utf-8"))
            compressed_lengths.append(len(zlib.compress(content_bytes, level=9)))

        # mean
        if len(compressed_lengths) > 0:
            self._insert_sparse_entry(mean(compressed_lengths), col_offset=0)
        else:
            self._insert_sparse_entry(-1.0, col_offset=0)

        # stddev
        if len(compressed_lengths) > 1:
            self._insert_sparse_entry(stdev(compressed_lengths), col_offset=1)
        else:
            self._insert_sparse_entry(-1.0, col_offset=1)


    def feature_entropy_mean_and_stddev(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie update
        Over all updates, compute the mean shannon entropy plus standard deviation.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        entropies: List[float] = list()
        for cookie_update in cookie_features["variable_data"]:
            unquoted_content = urllib.parse.unquote(cookie_update["value"])

            content_char_counts = Counter([ch for ch in unquoted_content])
            total_string_size = len(unquoted_content)

            entropy: float = 0
            for ratio in [char_count / total_string_size for char_count in content_char_counts.values()]:
                entropy -= ratio * log(ratio, 2)

            entropies.append(entropy)

        # mean
        if len(entropies) > 0:
            self._insert_sparse_entry(mean(entropies), col_offset=0)
        else:
            self._insert_sparse_entry(-1.0, col_offset=0)

        # stddev
        if len(entropies) > 1:
            self._insert_sparse_entry(stdev(entropies), col_offset=1)
        else:
            self._insert_sparse_entry(-1.0, col_offset=1)

    def feature_expiry_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Inserts 1 if throughout all updates for the cookie, the expiry changed at least once (more than 1 day difference).
        :param cookie_features: Dictionary containing key "variable_data"
        """
        v_iter = iter(cookie_features["variable_data"])
        previous_entry = next(v_iter)  # first entry always non-null
        try:
            while previous_entry is not None:
                next_entry = next(v_iter)
                if next_entry is not None:
                    abs_diff = abs(previous_entry["expiry"] - next_entry["expiry"])
                    if abs_diff >= 3600 * 24:
                        self._insert_sparse_entry(1.0)
                        return
                previous_entry = next_entry
        except StopIteration:
            pass


    def feature_http_only_first_update(self, cookie_features: Dict[str, Any]) -> None:
        """ per cookie feature
        HTTP_ONLY flag of the first update only. First update is guaranteed to exist.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if cookie_features["variable_data"][0]["http_only"]:
            self._insert_sparse_entry(1.0)

    def feature_host_only_first_update(self, cookie_features: Dict[str, Any]) -> None:
        """ per cookie feature
        HOST_ONLY flag of the first update only. First update is guaranteed to exist.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if cookie_features["variable_data"][0]["host_only"]:
            self._insert_sparse_entry(1.0)

    def feature_secure_first_update(self, cookie_features: Dict[str, Any]) -> None:
        """ per cookie feature
        SECURE flag of the first update only. First update is guaranteed to exist.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if cookie_features["variable_data"][0]["secure"]:
            self._insert_sparse_entry(1.0)

    def feature_session_first_update(self, cookie_features: Dict[str, Any]) -> None:
        """ per cookie feature
        SESSION flag of the first update only. First update is guaranteed to exist.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if cookie_features["variable_data"][0]["session"]:
            self._insert_sparse_entry(1.0)

    def feature_same_site_first_update(self, cookie_features: Dict[str, Any]) -> None:
        """ per cookie feature
        SAME_SITE flag of the first update only. First update is guaranteed to exist.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        s_flag: str = cookie_features["variable_data"][0]["same_site"]
        if s_flag == "no_restriction":
            self._insert_sparse_entry(1.0, col_offset=0)
        elif s_flag == "lax":
            self._insert_sparse_entry(1.0, col_offset=1)
        elif s_flag == "strict":
            self._insert_sparse_entry(1.0, col_offset=2)
        else:
            logger.warning(f"Unrecognized same_site content: {s_flag}")


    def feature_expiry_first_update(self, cookie_features: Dict[str, Any]) -> None:
        """ per cookie feature
        Expiry value of the first update only. First update is guaranteed to exist.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        expiry: int = cookie_features["variable_data"][0]["expiry"]
        self._insert_sparse_entry(expiry)


    def feature_content_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        Inserts 1 if throughout all updates for the cookie, the content changed at least once.
        :param cookie_features: Dictionary containing key "variable_data" and "value"
        """
        if check_flag_changed(cookie_features["variable_data"], "value"):
            self._insert_sparse_entry(1.0)


    #
    ## Per Update Features
    #

    def feature_http_only(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Set 1.0 if HTTP_ONLY flag is ON for this update, else -1.0
        :param var_data: Dictionary containing key "http_only"
        """
        self._insert_sparse_entry(1.0 if var_data["http_only"] else -1.0)


    def feature_secure(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Set 1.0 if "secure" flag is ON for this update, else -1.0.
        :param var_data: Dictionary containing key "secure"
        """
        self._insert_sparse_entry(1.0 if var_data["secure"] else -1.0)

    def feature_session(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Set 1.0 if "session" flag is set for this update, else -1.0
        :param var_data: Dictionary containing key "session"
        """
        self._insert_sparse_entry(1.0 if var_data["session"] else -1.0)

    def feature_same_site(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        The same_site flag can take one of three formats. Inserts a one-hot vector of 3 entries.
        :param var_data: Dictionary containing key "same_site"
            """
        if var_data["same_site"] == "no_restriction":
            self._multi_insert_sparse_entries([1.0, -1.0, -1.0])
        elif var_data["same_site"] == "lax":
            self._multi_insert_sparse_entries([-1.0, 1.0, -1.0])
        elif var_data["same_site"] == "strict":
            self._multi_insert_sparse_entries([-1.0, -1.0, 1.0])
        else:
            logger.warning(f"Unrecognized same_site content: {var_data['same_site']}")
            self._multi_insert_sparse_entries([-1.0, -1.0, -1.0])

    def feature_expiry(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Insert the time until expiration in seconds. Assumes that the time is already precomputed.
        :param var_data: Dictionary containing key "expiry"
        """
        self._insert_sparse_entry(var_data["expiry"])

    def feature_expiry_extra(self, var_data: Dict[str, Any]):
        """ per-update feature:
        Explicit boolean checks on the expiration time.
        Decided on some arbitrary intervals. Feature evaluation will determine if this is useful.
        :param var_data: Dictionary containing key "expiry"
        """
        # less than 1 hour
        self._insert_sparse_entry(1.0 if var_data["expiry"] < 3600 else -1.0, col_offset=0)
        # between 1 and 12 hours
        self._insert_sparse_entry(1.0 if 3600 <= var_data["expiry"] <= 3600 * 12 else -1.0, col_offset=1)
        # between 12 to 24 hours
        self._insert_sparse_entry(1.0 if 3600 * 12 <= var_data["expiry"] <= 3600 * 24 else -1.0, col_offset=2)
        # between 1 to 7 days
        self._insert_sparse_entry(1.0 if 3600 * 24 <= var_data["expiry"] <= 3600 * 24 * 7 else -1.0, col_offset=3)
        # between 1 week to 1 month
        self._insert_sparse_entry(1.0 if 3600 * 24 * 7 <= var_data["expiry"] <= 3600 * 24 * 30 else -1.0, col_offset=4)
        # between 1 month to 6 months
        self._insert_sparse_entry(1.0 if 3600 * 24 * 30 <= var_data["expiry"] <= 3600 * 24 * 30 * 6 else -1.0, col_offset=5)
        # between 6 months to 18 months
        self._insert_sparse_entry(1.0 if 3600 * 24 * 30 * 6 <= var_data["expiry"] <= 3600 * 24 * 30 * 18 else -1.0, col_offset=6)
        # more than 18 months
        self._insert_sparse_entry(1.0 if 3600 * 24 * 30 * 18 <= var_data["expiry"] else -1.0, col_offset=7)

    def feature_content_length(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Number of bytes of the cookie content.
        :param var_data: Dictionary of per-update cookie data, containing key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        data_size = len(bytes(unquoted_content.encode("utf-8")))
        self._insert_sparse_entry(data_size)

    def feature_compressed_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Number of bytes of the compressed content using zlib, as well as size reduction.

        This serves as a heuristic to represent entropy. If entropy is high, then the compressed
        data will like have around the same size as the uncompressed data. High entropy data is
        likely to be a randomly generated string. Low entropy data will have a stronger reduction
        in size after compression.

        :param var_data: Dictionary of per-update cookie data, containing key "value".
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        content_bytes = bytes(unquoted_content.encode("utf-8"))
        compressed_size = len(zlib.compress(content_bytes, level=9))

        # Append compressed size
        self._insert_sparse_entry(compressed_size, col_offset=0)

        # Append reduction
        reduced = len(content_bytes) - compressed_size
        self._insert_sparse_entry(reduced, col_offset=1)

    def feature_shannon_entropy(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Compute the Shannon entropy based on the characters present in the content string.
        This approach to represent entropy may differ from the zlib compression.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])

        content_char_counts = Counter([ch for ch in unquoted_content])
        total_string_size = len(unquoted_content)

        entropy: float = 0
        for ratio in [char_count / total_string_size for char_count in content_char_counts.values()]:
            entropy -= ratio * log(ratio, 2)

        self._insert_sparse_entry(entropy)


    def feature_url_encoding(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Check whether the content of the cookie is url-encoded by decoding and comparing
        with the original string. If a difference exists, we have URL encoding and set the
        entry to 1.0. Else, we set the entry to -1.0.
        :param var_data: Dictionary of per-update cookie data, containing key "value".
        """
        cookie_content: str = var_data["value"]
        unquoted_content = urllib.parse.unquote(cookie_content)
        if cookie_content != unquoted_content:
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)


    def feature_delimiter_separated(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """ per-update feature:
        Feature that checks whether the cookie content contains a delimiter separated value.
        Will insert the length if this is the case. A specific set of possible delimiters is considered for this.

        Excluded are "." and "-" as they usually have specific uses.
        Note that this is a heuristic, as it may include false positives.

        :param var_data: Dictionary of per-update cookie data containing key "value".
        :param min_seps: Minimum number (-1) of instances of the delimiter inside the string.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        split_data: List[str] = split_delimiter_separated(unquoted_content, self.csv_sniffer,
                                                          delimiters=self.valid_csv_delimiters, min_seps=min_seps)
        if split_data:
            self._insert_sparse_entry(len(split_data))
        else:
            self._insert_sparse_entry(-1.0)


    def feature_period_separated(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """ per-update feature:
        Check if the cookie content contains a string separated by the "." character.
        Inserts length. Heuristic, may include false positives.
        :param var_data: Dictionary of per-update cookie data containing key "value".
        :param min_seps: Minimum number of instances of the delimiter.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        split_data: List[str] = split_delimiter_separated(unquoted_content, self.csv_sniffer,
                                                          delimiters=".", min_seps=min_seps)
        if split_data:
            self._insert_sparse_entry(len(split_data))
        else:
            self._insert_sparse_entry(-1.0)


    def feature_dash_separated(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """per-update feature:
        Check if the cookie content contains a string separated by the "-" character.
        Inserts length if csv.
        :param var_data: Dictionary of per-update cookie data containing key "value".
        :param min_seps: Minimum number of instances of the delimiter.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        split_data: List[str] = split_delimiter_separated(unquoted_content, self.csv_sniffer,
                                                          delimiters="-", min_seps=min_seps)
        if split_data:
            self._insert_sparse_entry(len(split_data))
        else:
            self._insert_sparse_entry(-1.0)

    def feature_maybe_base64_encoding(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Heuristic Feature. Try to decode the cookie content as base64.
        If it works, then maybe it's binary data, maybe it's coincidence that the format matched.
        If it works, set 1.0, else set -1.0
        :param var_data: Dictionary of per-update cookie data.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        try:
            base64.b64decode(unquoted_content)
        except (base64.binascii.Error, ValueError):
            self._insert_sparse_entry(-1.0)
        else:
            self._insert_sparse_entry(1.0)

    def feature_definite_base64_encoding(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Heuristic Feature. Try to decode the cookie content as base64.
        If this worked, check if the resulting bytes can be encoded in utf-8.
        If so, we most likely have a valid base64 encoding.
        :param var_data: Dictionary of per-update cookie data with key "value".
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if try_decode_base64(unquoted_content):
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_contains_javascript_object(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Insert 1.0 if the cookie update content includes a javascript/JSON object.
        Else -1.0. Will also check inside base64 encoded strings.
        :param var_data: Dictionary of per-update cookie data with key "value".
        """
        unquoted_content: str = urllib.parse.unquote(var_data["value"])
        if self.js_pat.search(unquoted_content):
            self._insert_sparse_entry(1.0)
        else:
            maybe_base64: Optional[str] = try_decode_base64(unquoted_content)
            if maybe_base64 and self.js_pat.search(maybe_base64):
                self._insert_sparse_entry(1.0)
            else:
                self._insert_sparse_entry(-1.0)

    def feature_english_terms_in_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        Vector of features, where each entry is a binary value indicating whether the
        corresponding term is inside the content.
        :param var_data: Dictionary of per-update cookie data with key "value".
        """
        assert (self._content_terms is not None), "Content terms were not set up prior to feature extraction!"
        unquoted_content: str = urllib.parse.unquote(var_data["value"])
        for token, rank in self._content_terms:
            if token.search(unquoted_content):
                self._insert_sparse_entry(1.0, col_offset=rank)

    def feature_csv_content(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """ per-update feature:
        Split up a CSV separated string (if possible) and check whether it contains:
            1. numerical entries
            2. hexadecimal entries
            3. alphabetical entries
            4. alphanumerical entries
        Add these binary values as features.
        A predefined set of delimiters is considered for separation. Will also check inside base64.
        :param var_data: Cookie update dictionary with the key "value"
        :param min_seps: Minimum number of separations to be considered a CSV
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        data_entries = split_delimiter_separated(unquoted_content, self.csv_sniffer,
                                                 delimiters=self.valid_csv_delimiters, min_seps=min_seps)

        contains_num: bool = False
        contains_hex: bool = False
        contains_alpha: bool = False
        contains_alnum: bool = False
        contains_bool: bool = False
        if data_entries:
            entry: str
            for entry in data_entries:
                if type(entry) is list:
                    logger.error(data_entries)
                    logger.error(entry)
                contains_alpha |= entry.isalpha()
                contains_num |= entry.isnumeric()
                contains_alnum |= entry.isalnum()
                contains_hex |= (self.pattern_hexstring.match(entry) is not None)
                contains_bool |= (self.truth_values.match(entry) is not None)

        self._insert_sparse_entry(1.0 if contains_num else -1.0, col_offset=0)
        self._insert_sparse_entry(1.0 if contains_hex else -1.0, col_offset=1)
        self._insert_sparse_entry(1.0 if contains_alpha else -1.0, col_offset=2)
        self._insert_sparse_entry(1.0 if contains_alnum else -1.0, col_offset=3)
        self._insert_sparse_entry(1.0 if contains_bool else -1.0, col_offset=3)

    def feature_js_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Split javascript or JSON objects into individual components, and detect various types of contents
        as features. If the cookie content is not JSON, inserts -1.0 in every feature entry.
        Also tries to decode base64 if initial string is not json.

        Features extracted are:
          - whether one of the object keys contains "id"
          - bool, num, string, subobject, list, locale content binary checks

        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        data_dict: Optional[Dict[str, Any]] = None

        if "{" in unquoted_content:
            data_dict = try_split_json(unquoted_content)
        else:
            maybe_decoded = try_decode_base64(unquoted_content)
            if maybe_decoded and "{" in maybe_decoded:
                data_dict = try_split_json(maybe_decoded)

        found_identifier: bool = False
        contains_bool: bool = False
        contains_num: bool = False
        contains_string: bool = False
        contains_alpha: bool = False
        contains_alphanum: bool = False
        contains_subobject: bool = False
        contains_list: bool = False
        contains_hex: bool = False
        contains_none: bool = False

        if data_dict is not None:
            if type(data_dict) is dict:
                for key in data_dict:
                    if type(key) is not str:
                        logger.error("Unexpected key type:" + str(type(key)))
                    elif self.pattern_id_string.search(key):
                        found_identifier = True

                for v in data_dict.values():
                    if type(v) in (int, float):
                        contains_num = True
                    elif type(v) is bool:
                        contains_bool = True
                    elif type(v) is str:
                        contains_string = True
                        if self.truth_values.match(v):
                            contains_bool = True
                        elif v.isnumeric():
                            contains_num = True
                        elif self.pattern_hexstring.match(v):
                            contains_hex = True

                        if v.isalpha():
                            contains_alpha = True
                        elif v.isalnum():
                            contains_alphanum = True
                    elif type(v) is dict:
                        contains_subobject = True
                    elif type(v) is list:
                        contains_list = True
                    elif v is None:
                        contains_none = True
                    else:
                        logger.error("Unexpected type of value inside dict:" + str(type(v)))
            elif type(data_dict) is list:
                pass
            elif type(data_dict) is str:
                pass
            else:
                logger.warning("Unexpected non-dict result:" + str(type(data_dict)))

        if data_dict is not None:
            self._insert_sparse_entry(len(data_dict), col_offset=0)
        else:
            self._insert_sparse_entry(-1.0, col_offset=0)

        self._insert_sparse_entry(1.0 if found_identifier else -1.0, col_offset=1)
        self._insert_sparse_entry(1.0 if contains_bool else -1.0, col_offset=2)
        self._insert_sparse_entry(1.0 if contains_num else -1.0, col_offset=3)
        self._insert_sparse_entry(1.0 if contains_string else -1.0, col_offset=4)
        self._insert_sparse_entry(1.0 if contains_alpha else -1.0, col_offset=5)
        self._insert_sparse_entry(1.0 if contains_alphanum else -1.0, col_offset=6)
        self._insert_sparse_entry(1.0 if contains_subobject else -1.0, col_offset=7)
        self._insert_sparse_entry(1.0 if contains_list else -1.0, col_offset=8)
        self._insert_sparse_entry(1.0 if contains_none else -1.0, col_offset=9)
        self._insert_sparse_entry(1.0 if contains_hex else -1.0, col_offset=10)

    def feature_numerical_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Check if cookie content consists entirely of numbers 0-9.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if unquoted_content.isnumeric():
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)


    def feature_hex_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Check if the cookie content is a hexadecimal string.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if not unquoted_content.isnumeric() and self.pattern_hexstring.match(unquoted_content):
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_alpha_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Check if cookie content consists of alphabetical characters.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if unquoted_content.isalpha():
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_is_identifier(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
         Check if cookie content is a valid identifier that consists of at least
         one number and one alphabetical character.
         :param var_data: Cookie update dictionary with the key "value"
         """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if (not (unquoted_content.isalpha() or unquoted_content.isnumeric())
                and unquoted_content.isidentifier()):
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_all_uppercase_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content consists of all uppercase alphabetical characters,
        if at least one exists, disregarding non-alphabetical characters.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if unquoted_content.isupper():
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_all_lowercase_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content consists of all lowercase alphabetical characters,
        if at least one exists, disregarding non-alphabetical characters.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if unquoted_content.islower():
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_empty_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content is entirely empty or spaces.
        :param var_data: Cookie update dictionary with the key "value"
        """
        if not var_data["value"] or var_data["value"].isspace():
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_boolean_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content represents one of multiple possible truth values.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if self.truth_values.search(unquoted_content):
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_locale_term(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content represents a country, currency or language identifier
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if unquoted_content in self.locale_lookup:
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_timestamp_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content contains a timestamp.
        Range supported: 09/13/2020 to 11/14/2023.
        Matches both full second and millisecond timestamps.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if self.pattern_timestamp.search(unquoted_content):
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_date_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie content contains a formatted date of some form.
        Not complete -- this only covers some date representations.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        # Check for numerical dates
        if (self.pattern_year_month_day.search(unquoted_content) or self.pattern_day_month_year.search(unquoted_content)
                or self.pattern_month_day_year.search(unquoted_content)):
            self._insert_sparse_entry(1.0)
        # Check for month + day names
        elif ((self.pattern_alpha3_days_eng.search(unquoted_content) or self.pattern_full_days_eng.search(unquoted_content))
              and (self.pattern_alpha3_months_eng.search(unquoted_content) or self.pattern_full_months_eng.search(unquoted_content))):
            self._insert_sparse_entry(1.0)
        # Else no date present
        else:
            self._insert_sparse_entry(-1.0)

    def feature_canonical_uuid(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the content contains a canonical UUID of the format {8}-{4}-{4}-{4}-{12}.
        The inserted feature is a one-hot vector indicating the UUID version.

        The UUID version determines what data the identifier was constructed from.
        Version 1: Current time and MAC address of computer or node generating the id.
        (can be traced back to the computer that generated it)
        Version 2: DCE Security UUID, RFC 4122. local domain + local id, meaningful to local host.
        Version 3: predefined namespace + name, MD5 hash
        Version 4: entirely randomly generated
        Version 5: predefined namespace + name, SHA-1 hash

        Assumes that if there are multiple UUIDs, they are likely to be the same version.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        version_arr = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        match_obj = self.pattern_canon_uuid.search(unquoted_content)
        if match_obj:
            version_char = match_obj.group(1)
            if version_char == "1": version_arr[0] = 1.0
            elif version_char == "2": version_arr[1] = 1.0
            elif version_char == "3": version_arr[2] = 1.0
            elif version_char == "4": version_arr[3] = 1.0
            elif version_char == "5": version_arr[4] = 1.0
            else: version_arr[5] = 1.0

        self._multi_insert_sparse_entries(version_arr)


    def feature_url_content(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature
        Checks if the cookie contains a url or domain in its content field.
        :param var_data: Cookie update dictionary with the key "value"
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        if self.pattern_http.search(unquoted_content) or self.pattern_www.search(unquoted_content):
            self._insert_sparse_entry(1.0)
        else:
            self._insert_sparse_entry(-1.0)

    #
    ## Difference between Updates Features
    #

    def feature_time_diff(self, prev_update: Dict[str, Any], curr_update: Dict[str, Any]) -> None:
        """ per-diff feature:
        Compare the expiration time between updates, enter the time difference as a feature.
        May be positive or negative.
        :param prev_update: Dictionary of the previous update, containing key "expiry"
        :param curr_update: Dictionary of the current update, containing key "expiry"
        """
        self._insert_sparse_entry(curr_update["expiry"] - prev_update["expiry"])

    def feature_gestalt_pattern_ratio(self, prev_update: Dict[str, Any], curr_update: Dict[str, Any]) -> None:
        """ per-diff feature:
        Compare the ratio of similarity between the content of two contiguous updates.
        This is done using the python difflib library, which uses an algorithm similar to Gestalt Pattern matching.
        :param prev_update: Dictionary of the previous update, containing key "value"
        :param curr_update: Dictionary of the current update, containing key "value"
        """
        matcher = difflib.SequenceMatcher(a=prev_update["value"], b=curr_update["value"])
        difflib_metric = matcher.ratio()
        self._insert_sparse_entry(difflib_metric)

    def feature_levenshtein_dist(self, prev_update: Dict[str, Any], curr_update: Dict[str, Any]) -> None:
        """ per-diff feature:
        Compute the similarity of the content of two contiguous updates using the Levenshtein Distance.
        :param prev_update: Dictionary of the previous update, containing key "value"
        :param curr_update: Dictionary of the current update, containing key "value"
        """
        lev_metric = lev_distance(prev_update["value"], curr_update["value"])
        self._insert_sparse_entry(lev_metric)

    ###
    ### Experimental
    ###

    def feature_delimiter_separated_new(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """ per-update feature
        Reimplementation of the delimiter separation feature using native code.
        May be slower, but with this it is easier to know what is actually happening.
        Also, this is easier to port to Javascript.

        :param var_data: Dictionary of per-update cookie data containing key "value".
        :param min_seps: Minimum number (+1) of instances of the delimiter inside the string.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        chosen_delimiter, count = delim_sep_check(unquoted_content, self.valid_csv_delimiters, min_seps)

        logger.debug(f"Chosen Delimiter with {count} occurrences: {chosen_delimiter}")
        if chosen_delimiter is not None:
            self._insert_sparse_entry(count + 1)
        else:
            self._insert_sparse_entry(-1)

    def feature_period_separated_new(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """ per-update feature:

        :param var_data: Dictionary of per-update cookie data containing key "value".
        :param min_seps: Minimum number +1 of instances of the delimiter.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        chosen_delimiter, count = delim_sep_check(unquoted_content, ".", min_seps)

        logger.debug(f"Chosen delimiter {chosen_delimiter} with {count} occurrences")
        if chosen_delimiter is not None:
            self._insert_sparse_entry(count + 1)
        else:
            self._insert_sparse_entry(-1.0)

    def feature_dash_separated_new(self, var_data: Dict[str, Any], min_seps: int) -> None:
        """ per-update feature:

        :param var_data: Dictionary of per-update cookie data containing key "value".
        :param min_seps: Minimum number +1 of instances of the delimiter.
        """
        unquoted_content = urllib.parse.unquote(var_data["value"])
        chosen_delimiter, count = delim_sep_check(unquoted_content, "-", min_seps)

        logger.debug(f"Chosen delimiter {chosen_delimiter} with {count} occurrences")
        if chosen_delimiter is not None:
            self._insert_sparse_entry(count + 1)
        else:
            self._insert_sparse_entry(-1.0)


    ###
    ### Deprecated Features
    ###

    def feature_domain_period(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature:
        DEPRECATED
        Single feature entry, detects whether host domain starts with a period.
        The leading period indicates that the cookie can be used in all subdomains,
        and is not necessarily present on all cookies.
        :param cookie_features: Dictionary containing key "domain" of the cookie
        """
        if cookie_features["domain"].startswith("."):
            self._insert_sparse_entry(1.0)

    def feature_host_only_changed(self, cookie_features: Dict[str, Any]) -> None:
        """ per-cookie feature
        DEPRECATED
        Inserts 1 if throughout all updates for the cookie, the host_only flag changed at least once.
        :param cookie_features: Dictionary containing key "variable_data"
        """
        if check_flag_changed(cookie_features["variable_data"], "host_only"):
            self._insert_sparse_entry(1.0)

    def feature_host_only(self, var_data: Dict[str, Any]) -> None:
        """ per-update feature:
        DEPRECATED
        Set 1.0 if HOST_ONLY flag is ON for this update, else -1.0
        :param var_data: Dictionary containing key "host_only"
        """
        self._insert_sparse_entry(1.0 if var_data["host_only"] else -1.0)
