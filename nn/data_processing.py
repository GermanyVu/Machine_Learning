from sklearn.datasets import fetch_openml
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def flatten_data(training_inputs):  # should be numpy array
    # flattens each sample of data (useful for images) and changes the shape to a column vector
    training_inputs = [
        training_inputs[i].flatten()[:, np.newaxis] for i in range(len(training_inputs))
    ]
    return training_inputs


def vectorize_labels_mnist(y_train):
    training_labels = []
    for label in y_train:
        a = np.zeros((10, 1))
        a[label] = 1
        training_labels.append(a)
    return training_labels


def make_data_pairs(training_inputs, training_labels):
    training_data = [
        (training_input, training_label)
        for training_input, training_label in zip(training_inputs, training_labels)
    ]
    return training_data


def fetch_data(data_name):
    if data_name == "mnist_784":
        mnist = fetch_openml(data_name, version=1)
        X, y = mnist["data"], mnist["target"]
        y = y.astype(np.uint8)  # change the strings to int
        X = X / 255.0  # make values between 0 1 this helps grad descent go faster
        X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

    elif data_name == "ising":
        pathtofile = "training_data/"
        df = pd.read_csv(pathtofile + "trainingData.txt", sep=" ", header=None)
        df2 = pd.read_csv(pathtofile + "trainingLabels.txt", sep=" ", header=None)
        df3 = pd.read_csv(pathtofile + "testData.txt", sep=" ", header=None)
        df4 = pd.read_csv(pathtofile + "testLabels.txt", sep=" ", header=None)

        X_train = np.array(df)
        y_train = np.array(df2)
        X_test = np.array(df3)
        y_test = np.array(df4)

    training_inputs = flatten_data(X_train)
    training_labels = vectorize_labels_mnist(y_train)
    training_data = make_data_pairs(training_inputs, training_labels)
    test_inputs = flatten_data(X_test)
    test_data = make_data_pairs(test_inputs, y_test)
    return training_data, test_data


def show_mnist_number(paired_data, idx):
    """will display the image of the handwritten number.
    the index tells you what pair you want
    """
    # print(X_test[0])
    print(paired_data[idx][1])
    some_digit = paired_data[idx][0].reshape(28, 28)
    plt.imshow(some_digit, cmap=mpl.cm.binary, interpolation="nearest")
    plt.show()
