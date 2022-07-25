# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Small script to detect terms inside the content of cookies.
This produces a list of terms that are checked for as part of the feature extraction.
"""

import re
import json
from tqdm import tqdm
import enchant

from typing import Dict


min_length: int = 3
terms: Dict[str, int] = dict()
d = enchant.Dict('en_US')
names = set()

## Path to the training data JSON
with open("example_input/example_crawl_20210213_153228.json") as fd:
    tr = json.load(fd)
    for t in tr.values():
        for u in t["variable_data"]:
            names.add(u["value"].strip())

# try to find terms inside the string via  a substring search and the pyenchant package to detect if a term is valid
try:
    for name in tqdm(names):
        new_name = name.lower()
        size = len(new_name)
        for i in range(size):
            for j in range(i + min_length, size + 1):
                substr = new_name[i:j]
                if re.search("[0-9.-]", substr):
                    continue
                if d.check(substr):
                    if substr in terms:
                        terms[substr] += 1
                    else:
                        terms[substr] = 1
except KeyboardInterrupt:
    pass

# All substrings filtered out
filtered = list()

# Sort the terms by occurrence, remove ones that are contained in more common strings
sorted_terms = sorted(terms.items(), key=lambda x: x[1])
num_terms = len(sorted_terms)
for i in range(num_terms):
    lc_name = sorted_terms[i][0]
    for j in range(i+1, num_terms):
        mc_name = sorted_terms[j][0]
        if lc_name in mc_name:
            break
    else:
        filtered.append(sorted_terms[i])

# output the content feature list, sorted by number of occurrences
with open("content_features.out", 'w') as fw:
    for s, c in sorted(filtered, key=lambda x: x[1], reverse=True):
        if (len(s) > 3 or c > 100) and c > 10:
            fw.write(f"{c},{s}\n")
