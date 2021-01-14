#!/usr/bin/env python3
"""
Calculates the normalization
"""
import numpy as np


def normalization_constants(X):
    """
    a function that calculates the normalization (standardization) constants
    of a matrix
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
