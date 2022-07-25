# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Small script to detect terms inside the name of cookies.
This produces a list of terms that are checked for as part of the feature extraction.
"""
from typing import Dict
import re
import enchant

min_length: int = 3
terms: Dict[str, int] = dict()
d = enchant.Dict('en_US')
names = list()

with open("resources/top_names.csv") as fd:
    for l in fd:
        names.append(l.strip().split(',')[1])

total_cnt = len(names)
processed = 0
for name in names:
    new_name = name.lower()
    size = len(new_name)
    for i in range(size):
        for j in range(i + min_length, size + 1):
            substr = new_name[i:j]
            if re.search("[0-9.-]",substr):
                continue
            if d.check(substr):
                if substr in terms:
                    terms[substr] += 1
                else:
                    terms[substr] = 1
    if processed % 200 == 0:
        print(f"Processed {processed}/{total_cnt} names",end="\r")
    processed += 1

filtered = list()

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


with open("name_features.out", 'w') as fw:
    for s, c in sorted(filtered, key=lambda x: x[1], reverse=True):
        if (len(s) > 3 or c > 100) and c > 10:
            fw.write(f"{c},{s}\n")
