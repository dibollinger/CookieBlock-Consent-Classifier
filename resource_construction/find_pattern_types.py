# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Iterates though an alphabetical list of unique cookie names, picks out those names that share a great
degree of similarity in a contiguous sequence. This is used to determine cookie names that follow a
certain pattern. After extraction, manual work is needed to determine which names out of the filtered
list are truly pattern names, and additional work needs to be done to define a regex pattern for them.

These pattern names are then used for the feature extraction.

Requires name extraction script to be executed first.
"""
import re
from difflib import SequenceMatcher

minrepeat = 12
minlength = 2

names = list()
with open("./resources/top_names.csv", 'r') as fd:
    for l in fd:
        names.append(l.split(',')[1].strip())

sorted_names = sorted(names)
total = len(sorted_names)

patterns = list()

try:
    crepeat = 0
    matcher = SequenceMatcher(isjunk=None, autojunk=False)
    s_iter = iter(sorted_names)
    p_name = next(s_iter)
    while True:
        c_name = next(s_iter)
        matcher.set_seqs(p_name,c_name)
        sstringlength = matcher.find_longest_match(alo=0, ahi=len(p_name), blo=0, bhi=len(c_name)).size
        if sstringlength > minlength and re.search("[0-9]", c_name):
            crepeat += 1
        else:
            if crepeat >= minrepeat:
                patterns.append((crepeat,p_name))
            crepeat = 0
        p_name = c_name
except StopIteration:
    print("finished")

with open("possible_pattern_cookies.txt", 'w') as fw:
    cumsum = 0
    fw.write(f"#count,ratio,cumsum,last_name\n")
    fw.write(f"#{total},100%,100%,total\n")
    for c, p in sorted(patterns,key=lambda x:x[0], reverse=True):
        cumsum += c
        fw.write(f"{c},{c/total*100:.3f}%,{cumsum/total*100:.3f}%,{p}\n")
