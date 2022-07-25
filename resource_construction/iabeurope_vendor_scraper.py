# Copyright (C) 2021-2022 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Scrapes the IAB Vendor list for domains. These domains are then used for the feature extraction.
"""

import logging
import re
import requests
import requests.exceptions as rexcepts
import traceback

from bs4 import BeautifulSoup
from typing import Optional, Set, Tuple

# vendor pages
vendor_url = "https://iabeurope.eu/vendor-list/?fwp_paged="
stop_page = "Sorry, we couldn't find any posts. Please try a different search."

# timeout after which to kill a subprocess
parse_timeout = 120

# logger name
logger = logging.getLogger("iabeurope-vendor-crawler")


def simple_get(url) -> Optional[requests.Response]:
    """
    Perform a simple GET requests using the python requests library and handle errors.
    @param url: url to send the GET request to
    @return: Response from the webserver, or None if failed
    """
    try:
        # fake chrome user agent, required or else Cookiepedia won't answer at all
        headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"}
        r = requests.get(url, timeout=(40, 40), headers=headers)
        return r
    except (rexcepts.TooManyRedirects, rexcepts.SSLError,
            rexcepts.URLRequired, rexcepts.MissingSchema) as ex:
        logger.error(f"'{type(ex)}' exception occurred while trying to access: '{url}")
        return None
    except (rexcepts.ConnectionError, rexcepts.Timeout) as ex:
        logger.error(f"'{type(ex)}' exception occurred while trying to access: '{url}'")
        return None
    except Exception as ex:
        logger.error(f"Unexpected '{type(ex)}' exception occurred while trying to access: '{url}'")
        logger.error(traceback.format_exc())
        return None


def crawl_iabeurope_page(target_url) -> Tuple[Set[str], int]:
    """
    Given a iabeurope URL, attempts to extract listed vendor domains from the page.
    @param target_url: iabeurope path
    @return: (set of domains, return status)
             status == -2  --> Could not connect
             status == -1  --> Page not found
             status == 0  --> Retrieved domains
             status == 1  --> 0 domains extracted
    """
    extracted_domains = set()

    r = simple_get(target_url)
    if r is None:
        return extracted_domains, -2

    soup = BeautifulSoup(r.text, 'html.parser')

    h2_tags = soup.find_all("p")
    for h2 in h2_tags:
        inner_text = h2.get_text()
        if re.match(stop_page, inner_text):
            return extracted_domains, -1

    found_count: int = 0
    a_tags = soup.find_all('a')
    for a in a_tags:
        linktext = a.get('href')
        if linktext is None:
            continue
        else:
            if not linktext.startswith("http") or re.search("iabeurope\.eu", linktext):
                continue

            extracted_domains.add(linktext)
            found_count += 1

    if len(extracted_domains) == 0:
        return extracted_domains, 1

    return extracted_domains, 0


def setup_logger() -> None:
    """ Set up the logger instance. INFO to stderr """
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


def main() -> int:
    """ Run the extraction using a parallel process pool. """
    setup_logger()

    # Storage for all domains, using a set to prune duplicates
    extracted_domains = set()

    # some counters etc.
    curr_page = 0
    failed_pages = []
    try:
        while True:
            result_set, status = crawl_iabeurope_page(vendor_url + str(curr_page))

            if status == -2:
                logger.warning(f"Connection error when trying to access page {curr_page}")
                failed_pages.append(curr_page)
            elif status == -1:
                logger.warning(f"End of list has been reached at {curr_page}")
                break
            elif status == 1:
                logger.warning(f"Connection established for page {curr_page}, but no domains could be extracted!")
            else:
                logger.info(f"Crawled page: {curr_page}")
                extracted_domains.update(result_set)
            curr_page += 1
        logger.info(f"Search ended on page {curr_page}")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected -- outputting current domains")

    logger.info(f"Number of unique domains found: {len(extracted_domains)}")
    logger.info(f"pages that could not be crawl: {failed_pages}")

    outfile = "resources/iabeurope_vendors.txt"
    with open(outfile, "w") as fd:
        for d in sorted(extracted_domains):
            fd.write(d + "\n")

    logger.info(f"Extracted domains written to '{outfile}'")

    return 0


if __name__ == "__main__":
    exit(main())
