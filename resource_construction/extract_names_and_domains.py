# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Get a list of all cookie names and domains, sorted by occurrences.
This produces the top-names and top-domains rankings used by the feature extraction of the CookieBlock classifier
"""
import sqlite3
import re
import sys
import os

import logging
import traceback

from typing import Dict

name_query = """
    SELECT count(name) as c, name
    FROM (
        SELECT DISTINCT visit_id, name, host, path
        FROM javascript_cookies j
        WHERE j.record_type <> "deleted"
    )
    GROUP BY name
    ORDER BY c DESC;
"""

domain_query = """
    SELECT count(host) as c, host
    FROM (
        SELECT DISTINCT visit_id, name, host, path
        FROM javascript_cookies j
        WHERE j.record_type <> "deleted"
    )
    GROUP BY host
    ORDER BY c DESC;
"""

logger = logging.getLogger("main")

def main() -> int:
    """
    Extract top domains and top names rankings.
    @return:
    """
    if len(sys.argv) < 2:
        print("Error: Need path to database as first argument", file=sys.stderr)
        return 1

    database_path = sys.argv[1]
    if not os.path.exists(database_path):
        print("Error: specified path does not exist", file=sys.stderr)
        return 1

    # enable dictionary access by column name
    conn = sqlite3.connect(database_path)
    conn.row_factory = sqlite3.Row

    name_dict: Dict[str,int] = dict()
    domain_dict: Dict[str,int] = dict()

    try:
        with conn:
            cur = conn.cursor()

            cur.execute(name_query)
            for row in cur:
                count = int(row["c"])
                name = row["name"].strip()
                name_dict[name] = count

            cur.execute(domain_query)
            for row in cur:
                count = int(row["c"])
                domain = row["host"].strip()
                domain = re.sub("^http(s)?://", "", domain)
                domain = re.sub("^www", "", domain)
                domain = re.sub("^\\.", "", domain)
                if domain in domain_dict:
                    domain_dict[domain] += count
                else:
                    domain_dict[domain] = count

    except (sqlite3.OperationalError, sqlite3.IntegrityError):
        logger.error("A database error occurred:")
        logger.error(traceback.format_exc())
        return -1

    outpath = "./resources/"
    os.makedirs(outpath,exist_ok=True)

    n_path = os.path.join(outpath, "top_names.csv")
    with open(n_path, 'w') as fd:
        for d, c in sorted(name_dict.items(), key=(lambda x: x[1]), reverse=True):
            fd.write(f"{c},{d}\n")

    print(f"Domains written to '{n_path}'")

    d_path = os.path.join(outpath, "top_domains.csv")
    with open(d_path, 'w') as fd:
        for d, c in sorted(domain_dict.items(), key=(lambda x: x[1]), reverse=True):
            fd.write(f"{c},{d}\n")

    print(f"Domains written to '{d_path}'")

    return 0

if __name__ == "__main__":
    exit(main())
