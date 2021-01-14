#!/usr/bin/env python3
"""
Shuffles the data points in two matrices the same way
"""
import numpy as np


def shuffle_data(X, Y):
    """
    a function that shuffles the data points in two matrices the same way
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation, :], Y[permutation, :]
© 2021 GitHub, Inc.
