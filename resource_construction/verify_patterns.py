# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Requires a list of regexes that have been manually defined by a user.
Verifies how common pattern names are and sorts them in descending order.

Requires "patterns.txt" in base folder, and list of names in "resources/top_names.csv"

This produces the pattern name ranking used for the feature extraction.
"""


import re
import os

def main() -> int:

    #bp = "./pattern_check/"
    #os.makedirs(bp, exist_ok=True)

    names = []
    with open("resources/top_names.csv", 'r') as fd_names:
        for n in fd_names:
            names.append(n.split(',')[1].strip())

    total = len(names)
    counts = dict()

    with open("patterns.txt", 'r') as fd_patterns:
        for p in fd_patterns:
            if p.startswith("##"):
                continue
            pattern = p.strip()
            counts[pattern] = 0
            for n in names:
                if re.match(pattern, n):
                    counts[pattern] += 1
            #with open(bp + pattern, 'w') as fw:
            #    for n in names:
            #        if re.match(pattern, n):
            #            fw.write(n + "\n")


    with open("resources/verified_patterns.csv",'w') as fd:
        cumsum = 0
        for p, c in sorted(counts.items(),key=lambda x: x[1],reverse=True):
            cumsum += c
            fd.write(f"{c},{c/total*100:.2f},{cumsum/total*100:.2f},{p}\n")

    print("Verified patterns output to 'resources/verified_patterns.csv'")


if __name__ == "__main__":
    exit(main())
