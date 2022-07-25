# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License

""" Contains static utility functions for the feature transformation process"""

import re
import base64
import csv
import json
import js2py
from typing import Tuple, Dict, List, Union, Optional, Any

MIN_B64_LEN: int = 8

# Types accepted by the function below
VariableCookieData = List[Dict[str, Union[int, str]]]


def url_to_uniform_domain(url: str) -> str:
    """
    Takes a URL or a domain string and transforms it into a uniform format.
    Examples: {"www.example.com", "https://example.com/", ".example.com"} --> "example.com"
    :param url: URL to clean and bring into uniform format
    """
    new_url = url.strip()
    new_url = re.sub("^http(s)?://", "", new_url)
    new_url = re.sub("^www([0-9])?", "", new_url)
    new_url = re.sub("^\\.", "", new_url)
    new_url = re.sub("/$", "", new_url)
    return new_url


def load_lookup_from_csv(csv_source: str, count: int) -> Dict[str, int]:
    """
    Load a lookup dictionary, mapping string to rank, from a CSV file.
    Assumes the CSV to be pre-sorted.
    :param csv_source: Source data filepath
    :param count: number of strings to load
    :return: dictionary mapping strings to rank
    """
    lookup_dict: Dict[str, int] = dict()
    rank: int = 0
    with open(csv_source, 'r') as fd:
        line = next(fd)
        try:
            while rank < count:
                if line.startswith("#"):
                    line = next(fd)
                    continue
                lookup_dict[line.strip().split(',')[-1]] = rank
                rank += 1
                line = next(fd)
        except StopIteration:
            raise RuntimeError(f"Not enough entries in file. Expected at least {count}, max is {rank}.")

    return lookup_dict


def check_flag_changed(cookie_updates: VariableCookieData, flag: str) -> bool:
    """
    Checks if given flag (by key) differs between cookie updates.
    If so, return True, else False. May iterate over all updates.
    :param cookie_updates: Cookie update dictionary
    :param flag: Key for the flag to check between updates
    :return: True if differs between at least 1 update, False otherwise.
    """
    v_iter = iter(cookie_updates)
    previous_entry = next(v_iter)  # first entry always non-null
    try:
        while previous_entry is not None:
            next_entry = next(v_iter)
            if next_entry is not None:
                if previous_entry[flag] != next_entry[flag]:
                    return True
            previous_entry = next_entry
    except StopIteration:
        pass
    return False


def try_decode_base64(possible_encoding: str) -> Optional[str]:
    """
    Try to decode the given input object as base64.
    :param possible_encoding: string that is potentially an encoded value
    :return: If result is an UTF-8 string, return it. Else, return None
    """
    if type(possible_encoding) is not str or len(possible_encoding) < MIN_B64_LEN:
        return None
    else:
        try:
            b64decoded = base64.b64decode(possible_encoding)
            return b64decoded.decode("utf-8")
        except (base64.binascii.Error, UnicodeDecodeError, ValueError):
            return None


def try_split_json(possible_json: str) -> Optional[Dict[str, Any]]:
    """
    Try to split the javascript or json string, if possible.
    :param possible_json: string to split into a dictionary
    :return: dictionary containing json keys and attributes
    """
    try:
        return json.loads(possible_json)
    except json.JSONDecodeError:
        try:
            js_func = js2py.eval_js("function a() { return " + possible_json + " }")
            return js_func().to_dict()
        except (js2py.internals.simplex.JsException, NotImplementedError, AttributeError):
            pass
    return None


def split_delimiter_separated(possible_csv: str, csv_sniffer: csv.Sniffer,
                              delimiters: str, min_seps: int = 2):
    """
    If the given string is delimiter separated, split it and return the list of content strings.
    :param possible_csv: String to split.
    :param csv_sniffer: Sniffer instance to use.
    :param delimiters: String of valid delimiters
    :param min_seps: number of instances required for CSV to be recognized
    :return: None if cannot be separated. Else, a list of strings.
    """
    try:
        dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
        num_separators = possible_csv.count(dialect.delimiter)
        if num_separators > min_seps:
            return list(csv.reader((possible_csv,), dialect))[0], dialect.delimiter
    except csv.Error:
        # not csv formatted -- check if it's base64
        maybe_decoded = try_decode_base64(possible_csv)
        if maybe_decoded is not None:
            try:
                # debug
                # print("Successfully decoded b64 with csv")
                dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
                num_separators = possible_csv.count(dialect.delimiter)
                if num_separators > min_seps:
                    return list(csv.reader((possible_csv,), dialect))[0], dialect.delimiter
            except csv.Error:
                pass

    return None, None


def contains_delimiter_separated(possible_csv: str, csv_sniffer: csv.Sniffer,
                                 delimiters: str, min_seps: int = 2) -> bool:
    """
    Verify whether the given string is delimiter separated.
    :param possible_csv: String to verify.
    :param csv_sniffer: Sniffer instance to use.
    :param delimiters: String of valid delimiters
    :param min_seps: number of instances required for CSV to be recognized
    :return: True if delimiter separated, False if not
    """
    try:
        dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
        num_separators = possible_csv.count(dialect.delimiter)
        if num_separators > min_seps:
            return True
    except csv.Error:
        # not csv formatted -- check if it's base64
        maybe_decoded = try_decode_base64(possible_csv)
        if maybe_decoded is not None:
            try:
                # debug
                # print("Successfully decoded b64 with csv")
                dialect = csv_sniffer.sniff(possible_csv, delimiters=delimiters)
                num_separators = possible_csv.count(dialect.delimiter)
                if num_separators > min_seps:
                    return True
            except csv.Error:
                pass

    return False


def delim_sep_check(to_check: str, delims: str, min_seps: int) -> Tuple[Optional[str], int]:
    """
    Determine the best separator in a string, where a separator must appear at least min_seps times.
    Heuristic. Tries to determine if we have CSV separated data.
    :param to_check: String to check
    :param delims: List of delimiters as a string
    :param min_seps: minimum number of occurrences
    :return: Best separator and number of occurrences of this separator.
    """
    maxoccs = min_seps
    chosen_delimiter = None

    for d in delims:
        numoccs = to_check.count(d)
        if numoccs > maxoccs:
            chosen_delimiter = d
            maxoccs = numoccs

    return chosen_delimiter, maxoccs