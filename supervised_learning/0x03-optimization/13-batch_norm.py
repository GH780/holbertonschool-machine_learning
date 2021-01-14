#!/usr/bin/env python3
"""
Normalizes an unactivated output of a neural network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    a function that uses batch normalization
    """
    mean = np.mean(Z, axis=0)
    var = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(var + epsilon)
    z_n = gamma * normalized + beta
    return z_n
