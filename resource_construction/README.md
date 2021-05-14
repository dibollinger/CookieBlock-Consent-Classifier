# Scripts to construct the Feature Extraction Resources

The scripts in this folder were used to construct the resources utilized for the 
feature extraction of the CookieBlock Feature Extractor.

The input format can vary. For some script a CookieBlock consent crawl database 
is expected, in others the training data in the format of a JSON file is required.

## Folder Contents
* `example_input/`: Contains an example input JSON file to use with the scripts.
* `resources/`: Target folder for the outputs
* `content_features.py`: Extract a ranking of the most common terms inside cookie contents.
* `extract_names_and_domains.py`: Get a ranking of the top N names and domains found in a database of cookies.
* `find_pattern_types.py`: Try to extract potential pattern names out of cookies.
* `iabeurope_vendor_scraper.py`: Scrape IAB Europe Vendor domains, for use as a feature.
* `name_features.py`: Extract a ranking of the most common terms inside cookie names.
* `verify_patterns.py`: Verify if name patterns work, and extract ranking by number of occurences.