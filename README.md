# Cookie Consent Classifier
This repository contains a collection of scripts to train machine learning classifiers on cookie data. 
It forms the basis of the classifier included 

Implemented are XGBoost, LightGBM, CatBoost and a simple Recurrent Neural Network.

Also includes the feature extraction scripts to create the sparse input matrix.

# Training a Classifier

## Required Inputs

A JSON document containing the cookie data with associated labels is required. 

The cookie data can be obtained by using the following crawler: https://github.com/dibollinger/CookieBlock-Consent-Classifier

Training data can be extracted from the resulting SQLite database through the cookie extraction
script in the following repository: https://github.com/dibollinger/CookieBlock-Other-Scripts/blob/main/database_scripts/extract_cookie_data.py

## Feature Extraction

All components relevant for the feature extraction are stored in the subfolder 
`feature_extraction/`.

To extract data with labels for the purpose of training a classifier, run the script 
`prepare_training_data.py`. The resulting data matrix will be stored in the subfolder 
`processed_features/`.

### Usage
    prepare_training_data.py <tr_data>... [--format <FORMAT>] [--out <OFPATH>]
    Options:
        -f --format <FORMAT>   Output format. Options: {libsvm, sparse, debug. xgb} [default: sparse]
        -o --out <OFPATH>      Filename for the output. If not specified, will reuse input filename.
        -h --help              Show this help message.

Results can be output as either libsvm text format, as a pickled sparse matrix,
or in the form of an XGB data matrix. In either case, the script also produces a list of labels, weights
and feature names. For the XGB output these are already integrated into the binary, while for the other formats
they are output as separate files.


### Expected JSON Format

The data for each cookie is expected to be stored as a JSON object, with the name, domain and path forming
the key `cookie_id`. Each entry is structured as follows:
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
The `variable_data` attribute contains a list of cookie properties that may change with each update.

The attribute `label` is only required if the data is to be used for training a classifier.
For prediction purposes it is not needed.


## Training the model

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

## Extracting the Model

To produce the model used for the CookieBlock extension, one needs to train an XGBoost model
using the script `classifiers/xgboost/train_xgb.py`, ideally with features extracted from the
separate JavaScript implementation, provided at:

https://github.com/dibollinger/CookieBlock

Then, execute the script `classifiers/xgboost/create_small_dump.py` and provide it as input the
boosted XGB model, in order to obtain four model files named `forest_classX.json`. 
This contains a compressed representation of the tree model that can be read by the browser 
extension and used to make decision. Each file corresponds to a specific class and should not 
be renamed. 

#Repository Contents
* `./classifiers/`: Contains python scripts to train the various classifier types.
    - `./classifiers/catboost/`: Scripts that use the "CatBoost" approach.
    - `./classifiers/lightgbm/`: Scripts that use the "LightGBM" approach.
    - `./classifiers/neural_networks/`: Tensorflow neural network scripts.
    - `./classifiers/xgboost/`: XGBoost-specific scripts. Includes training, feature importance and model extraction.
* `./feature_extraction/`: Contains all python code needed for feature extraction.
* `./feature_extraction/features.json`: Configuration where one can define, configure, enable or disable individual features.
* `./processed_features/`: Directory where the extracted feature matrices are stored.
* `./resources/`: Contains external resources used for the feature extraction.
* `./training_data/`: Contain some examples for the JSON-formatted training data.
* `./predict_class.py`: Using a previously constructed classifier model, and given JSON cookie data as input, predicts labels for each cookie.
* `./prepare_training_data.py`: Script to transform input cookie data (in JSON format) into 
                                a sparse feature matrix. The feature selection and parameters
                                are controlled by `features_extraction/features.json`.
----
# Credits and License

Copyright © 2021 Dino Bollinger

MIT License, see included LICENSE file

This repository uses the XGBoost, LightGBM and CatBoost algorithms, as well as Tensorflow.

They can be found at:
* __XGBoost:__ https://github.com/dmlc/xgboost/
* __LightGBM:__ https://github.com/microsoft/LightGBM
* __CatBoost:__ https://github.com/catboost
* __Tensorflow:__ https://www.tensorflow.org/

The scripts in this repository were created as part of a master thesis on GDPR Compliance, 
and is part of a series of repositories for the __CookieBlock__ browser extension:

https://github.com/dibollinger/CookieBlock

Thanks go to:
* Karel Kubicek
* Dr. Carlos Cotrini
* Prof. Dr. David Basin
* The Institute of Information Security at ETH Zürich
