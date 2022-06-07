# CookieBlock Consent Classifier
* [Introduction](#introduction)
* [Requirements](#requirements)
  * [Input Format](#input-format)
* [Repository Contents](#repository-contents)
  * [Additional Details](#additional-details)
* [Feature Extraction](#feature-extraction)
  * [Usage](#usage)
* [Classifier Usage](#classifier-usage)
  * [Training the CookieBlock Model](#training-the-cookieblock-model)
* [Repository Contents](#repository-contents)
* [Credits](#credits)
* [License](#license)

## Introduction

This repository contains the Feature Extractor as well as the classifier approaches used for CookieBlock.
Note that this repository contains the Python variant of the feature extraction, which differs from 
the Javascript version. Its outputs should not be used with CookieBlock!

The feature extractor takes as input a json document of cookie data, and outputs a sparse matrix
of numerical features, where each row is a training sample, and the columns represent the features.

The resulting sparse matrix can be used as input to the classifier training. Currently implemented
are XGBoost, LightGBM, Catboost and a Recurrent Neural Network (the last not being greatly expanded
upon). 

## Requirements

The required libraries are listed inside `requirements.txt` placed in the base folder. 
No special setup or install is needed, only the libraries need to be installed.

In order to perform the feature extraction, some resource files are used. Default files
are provided within the folder `resources/`, but these can be recomputed as desired 
through the scripts located in the folder `resource_construction/`. 

For more information on the contents of this folder, see [the README](resource_construction/README.md).

### Input Format

Each cookie is expected to be stored as a JSON object, with the name, domain and path forming the 
key. Each entry is structured as follows:
```json
{
 "cookie_id": {
   "name": "<name>",
   "domain": "<domain>",
   "path": "/path",
   "first_party_domain": "http://first-party-domain",
   "label": 0,
   "cmp_origin": 0,
    "variable_data": [
     {
     "value": "<cookie content>",
     "expiry": "<expiration in seconds>",
     "session": "<true/false>",
     "http_only": true,
     "host_only": true,
     "secure": true,
     "same_site": "<no restriction/lax/strict>"
     }
   ]
 }
}
```

Each object uniquely identifies a cookie. The `first_party_domain` field is optional and may be left empty.
The `variable_data` attribute contains a list of cookie properties that may change with each update. The 
attribute `label` is only required if the data is to be used for training a classifier. For prediction 
purposes it is not needed.

Scripts to gather labelled training data, as well as to generate the above JSON format from collected cookies 
can be found in the Consent Crawler repository:

https://github.com/dibollinger/CookieBlock-Consent-Crawler

## Feature Extraction

The feature extraction takes as input the cookie information in JSON format, and computes from it a sparse
matrix of numerical data. A large selection of feature extraction steps is supported. For the full list of
them, refer to the [features.json](feature_extraction/features.json).

From each cookie, we extract three distinct categories of features:

* __Features extracted once per cookie:__ These features only need to be computed once for each cookie,
  as they are based on information that cannot be altered through any changes to that cookie. This 
  includes name, domain, path and other properties like the host_only flag.
* __Features extracted once per update:__ These features are based on variable cookie data, such as
  the payload of the cookie itself. Since these values may change, they are hence extracted once for
  each observed "update" to a cookie, i.e. an instance where a previously existing cookie is set again.
* __Features stemming from update differences:__ These features are computed when at least 2 updates are
  present. Examples for this are the difference in expiration date, or the edit distance between the
  values of two cookie updates.

The extracted features are entirely numerical, i.e. no text or categorical data remains. However, the
feature vector may contain boolean, ordinal or even missing data. Missing data is hereby represented as
zero entries in the per-update or per-diff features.

The [features.json](feature_extraction/features.json) holds not only a description of each feature, it
also acts as input to the extractor and defines which features exist, what functions are used to
extract them, what arguments are to be provided, if they require any resources to be loaded in advance
and whether they are enabled or disabled for the next feature extraction run. Note that the "vector_size"
key indicates how many entries a single feature produces.

Each run of the feature extraction will also produce statistics on how long the extraction took.

### Usage
To extract data with labels for the purpose of training a classifier, run the script 
`prepare_training_data.py` with the desired inputs. The resulting data matrix will be 
stored in the subfolder `processed_features/`.

    prepare_training_data.py <tr_data>... [--format <FORMAT>] [--out <OFPATH>]
    Options:
        -f --format <FORMAT>   Output format. Options: {libsvm, sparse, debug. xgb} [default: sparse]
        -o --out <OFPATH>      Filename for the output. If not specified, will reuse input filename.

The results can be output as either libsvm text format, as a pickled sparse matrix,
or in the form of an XGB data matrix. In either case, the script also produces a list of labels, weights
and feature names. For the XGB output these are already integrated into the binary, while for the other formats
they are output as separate files.

## Classifier Usage

    Usage:
      train_xgb.py <tr_data> <mode>
      train_lightgbm.py <tr_data> <mode>
      train_catboost.py <tr_data> <mode>

Inside the subfolder `classifiers` one will find a number of scripts that will train the
corresponding type of classifier. Generally, these scripts support the following modes:

* `simple_train`: Trains the classifier without producing any validation information.
This mode is useful to prepare a model trained on the full data to compute predictions for
previously unseen cookies -- for instance, to use it inside a browser extension.
* `split`: Perform a simple 80/20% train/test split, training the classifier on 80% of the data
while validating on the remaining 20%. In addition to producing validation statistics during
training, this will also output a confusion matrix based on the 20% of validation data, and
an accuracy score based on a simple policy of taking the maximum probability class as the prediction.
There are also some additional statistics outputs not mentioned here that can be enabled with this mode.
* `cross_validate`: Performs 5-fold stratified cross-validation on the training data. Unlike
the train-test split, this will not output a confusion matrix, and it will not output a model,
but it does provide added guarantees on the performance of a classifier, such as mean and standard
deviation over 5 folds, that the 80/20 test split cannot.
* `grid_search`: Performs hyperparameter search using the gridsearch approach.
Allows for the discovery of optimal parameter combination.
* `random_search`: Performs hyperparameter search using random combination of parameters.
Much more efficient but less thorough than grid search. Useful for assessing the overall
impact of altering individual parameters.
  
For XGBoost, the `classifiers/` directory contains a number of additional scripts to produce
feature importance and other similar statistics. Please refer to the documentation in the 
individual script files for more information.

Finally, there is the scripts `predict_class.py` and `feature_matrix_statistics.py` in the base
folder. The former is used as a simple wrapper to compute predictions with given cookie data,
while the latter can be used to output the most common and least commonly extracted features in
the cookie data.

### Training the CookieBlock Model

To produce the model used for the CookieBlock extension, one needs to train an XGBoost model
using the script `classifiers/xgboost/train_xgb.py`, with features extracted from the
separate JavaScript feature extraction implementation, provided at:

https://github.com/dibollinger/CookieBlock

This feature extractor works in the same way as the Python variant, with minor differences.

Then, execute the script `classifiers/xgboost/create_small_dump.py` and provide it as input the
boosted XGB model, in order to obtain four model files named `forest_classX.json`. 
This contains a compressed representation of the tree model that can be read by the browser 
extension and used to make decision. Each file corresponds to a specific class and should not 
be renamed. 


## Repository Contents

* `./classifiers/`: Contains python scripts to train the various classifier types.
* `./feature_extraction/`: Contains all python code needed for feature extraction.
* `./feature_extraction/features.json`: Configuration where one can define, configure, enable or disable individual features.
* `./resource_construction/`: Contains the Python scripts that were used to construct the feature extraction resources.
* `./processed_features/`: Directory where the extracted feature matrices are stored.
* `./resources/`: Contains external resources used for the feature extraction.
* `./training_data/`: Contain some examples for the JSON-formatted training data.
* `./feature_matrix_statistics.py`: Computes the most and least commonly used features, sorted by occurrence.
* `./predict_class.py`: Using a previously constructed classifier model, and given JSON cookie data as input, predicts labels for each cookie.
* `./prepare_training_data.py`: Script to transform input cookie data (in JSON format) into 
                                a sparse feature matrix. The feature selection and parameters
                                are controlled by `features_extraction/features.json`.
  
## Credits

This repository was created as part of the master thesis __"Analyzing Cookies Compliance with the GDPR"__, 
which can be found at:

https://www.research-collection.ethz.ch/handle/20.500.11850/477333

as well as the paper __"Automating Cookie Consent and GDPR Violation Detection"__, which can be found at:

https://karelkubicek.github.io/post/cookieblock.html

__Thesis supervision and co-authors:__
* Karel Kubicek
* Dr. Carlos Cotrini
* Prof. Dr. David Basin
* Information Security Group at ETH Zürich

---
See also the following repositories for other components that were developed as part of the thesis:

* [CookieBlock Browser Extension](https://github.com/dibollinger/CookieBlock)
* [OpenWPM-based Consent Crawler](https://github.com/dibollinger/CookieBlock-Consent-Crawler)
* [Violation Detection](https://github.com/dibollinger/CookieBlock-Other-Scripts)
* [Prototype Consent Crawler](https://github.com/dibollinger/CookieBlock-Crawler-Prototype)
* [Collected Data](https://doi.org/10.5281/zenodo.5838646)

---
This repository uses the XGBoost, LightGBM and CatBoost algorithms, as well as Tensorflow.

They can be found at:
* __XGBoost:__ https://github.com/dmlc/xgboost/
* __LightGBM:__ https://github.com/microsoft/LightGBM
* __CatBoost:__ https://github.com/catboost
* __Tensorflow:__ https://www.tensorflow.org/

## License

__Copyright © 2021 Dino Bollinger, Department of Computer Science at ETH Zürich, Information Security Group__

MIT License, see included LICENSE file
