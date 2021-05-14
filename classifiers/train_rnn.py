# Copyright (C) 2021 Dino Bollinger, ETH Zürich, Information Security Group
# Released under the MIT License
"""
Train a recurrent network using the Keras library, for the purpose of classifying cookies.
There are 4 categories: ["necessary", "functional", "analytics", "advertising"]
Supported modes: train, split
---------------------------------
Usage:
    recurrent_network.py <tr_data> <mode>
"""
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

import os
import logging

from datetime import datetime
from docopt import docopt

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from collections import Counter

from utils import load_data, log_validation_statistics, setupLogger, save_validation

from typing import List, Optional

# logger
logger = logging.getLogger("classifier.neural_networks")


def construct_recurrent_model(num_features: int) -> keras.Sequential:
    """
    Construct a recurrent neural network multi-class classifier. (BiLSTM-GRU)
    :param num_features: Number of input features.
    :return: A Keras Sequential Model
    """
    input_shape = (num_features,)
    model = keras.Sequential()
    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=True, input_shape=input_shape, activation="tanh",
                                                           dropout=0.1, recurrent_activation="sigmoid", use_bias=True)))
    model.add(keras.layers.Dense(units=4, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.CategoricalAccuracy(),
                           keras.metrics.CategoricalCrossentropy()])


    model.add(keras.layers.Dropout(rate=0.1))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=256, input_shape=input_shape, activation="tanh")))
    model.add(keras.layers.Dense(units=4, activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=[keras.metrics.CategoricalAccuracy(),
                           keras.metrics.CategoricalCrossentropy()])


    return model


def save_model(train_history, model: keras.Sequential, ) -> None:
    """
    Save the provided keras Sequential model as a h5 file.
    This contains the architecture and the trained weights.
    :param model: Trained keras model to save
    :param train_history: Training history of the model, as output by the fit function.
    """
    model_folder: str = "./models/"
    os.makedirs(model_folder, exist_ok=True)

    now_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_fp: str = os.path.join(model_folder, f"recmodel_{now_str}.h5")
    model.save(model_fp, include_optimizer=True, save_format="h5", overwrite=True)

    logger.info(f"Computed model written to {model_fp}")


def output_validation_statistics(model: keras.Sequential, dtest: np.ndarray,
                                 y_test: List[int], class_names: List[str]) -> None:
    """
    Output scores such as accuracy, precision and recall, the confusion matrix and
    more to a subfolder.
    :param model: Keras model
    :param dtest: validation data in the form of a sparse matrix
    :param y_test: Labels as a list, corresponding to the validation data
    :param class_names: names corresponding to the classes
    """
    # Further evaluation data
    eval_path = "./nn_validation_stats/"
    os.makedirs(eval_path, exist_ok=True)
    timestamp_str: str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Get actual predictions for the validation data
    predicted_probabilities: np.ndarray = model.predict_proba(dtest)
    true_labels: np.ndarray = np.array(y_test).astype(int)
    assert len(predicted_probabilities) == len(true_labels), "Number of labels did not match number of predictions"

    # output statistics such as confidence in predictions, confusion matrix and accuracy
    log_validation_statistics(predicted_probabilities, true_labels, class_names, eval_path, timestamp_str)

    # Save the validation data
    save_validation(dtest, y_test, eval_path, timestamp_str)


def simple_train(X: csr_matrix, y: List[int], weights: Optional[List[float]]) -> None:
    """
    Train on full provided dataset and labels, with weights, and output the
    resulting model file to disk.
    :param X: Dataset to train on.
    :param y: Labels for each data point.
    :param weights: Weights for each data point.
    """
    class_names = ["necessary", "functional", "analytics", "advertising"]
    num_classes = len(class_names)

    num_epochs = 64
    model = construct_recurrent_model(num_features=X.shape[1])

    # class weights to counter imbalance
    class_weights = {i: weights[y.index(i)] for i in range(num_classes)}
    h = model.fit(X, y, epochs=num_epochs, verbose=2, class_weight=class_weights)
    save_model(h, model)


def split_train(X: csr_matrix, y: List[int], weights: Optional[List[float]], train_fraction=0.8) -> None:
    """
    Train using a train/test split, and output additional statistics.
    :param X: Training features
    :param y: labels per instance
    :param weights: weights per training instance
    :param train_fraction: fraction of data to train on, default 0.8
    """
    assert weights is not None and len(y) == len(weights), "size of weights array doesn't match label array length"

    batch_size = 64
    shuf_buf = 128
    nfeats = X.shape[1]

    class_names = ["necessary", "functional", "analytics", "advertising"]
    num_classes = len(class_names)
    assert num_classes == len(Counter(y).keys()), "number of classes in y does not match expected number of classes"

    # Split the data into train and test set
    (X_train, X_test,
     y_train, y_test,
     w_train, w_test) = train_test_split(X.todense(), y, weights, train_size=train_fraction, shuffle=True)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # One-hot encode the labels because reasons
    ohe = OneHotEncoder(sparse=False)
    y_train_arrs = ohe.fit_transform(np.array(y_train).reshape(-1, 1))
    y_test_arrs = ohe.fit_transform(np.array(y_test).reshape(-1, 1))

    # Set up the training and test datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_arrs))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test_arrs))

    # Shuffle the training set, batch it
    train_dataset = train_dataset.shuffle(shuf_buf).batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    # Finally, perform training
    num_epochs = 32
    model: keras.Sequential = construct_recurrent_model(num_features=nfeats)

    # class weights to counter imbalance
    class_weights = {i: weights[y.index(i)] for i in range(num_classes)}
    h = model.fit(train_dataset, validation_data=test_dataset, epochs=num_epochs, verbose=2, shuffle=True, class_weight=class_weights)

    output_validation_statistics(model, X_test, y_test, class_names)

    save_model(h, model)


def main() -> int:
    """ Perform training of an Recurrent Neural Network model """
    argv = None

    cargs = docopt(__doc__, argv=argv)
    setupLogger(f"./train_rnn{''}.log")

    np.random.seed(0)
    tf.random.set_seed(0)

    # check for errors in input parameters
    tr_dat_path: str = cargs["<tr_data>"]
    if not os.path.exists(tr_dat_path):
        logger.error("Specified training data file does not exist.")
        return 2

    # load features, labels and weights
    X, y, W = load_data(tr_dat_path)

    if y is None:
        logger.error("Could not load labels -- labels required for training.")
        return 3

    # Mode of training
    mode: str = cargs["<mode>"]

    # Switch on training mode
    if mode == "train":
        logger.info("Simple training on full data.")
        simple_train(X, y, W)
    elif mode == "split":
        logger.info("Training using train-test split.")
        split_train(X, y, W)
    else:
        logger.error("Unrecognized mode. Available modes are: {train, split}")
        return 100


if __name__ == "__main__":
    exit(main())
