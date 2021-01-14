#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    a function that optimizes using Adam
    """
    vdw = beta1 * v + (1 - beta1) * grad
    sdw = beta2 * s + (1 - beta2) * grad ** 2
    # Corrected
    vdw_corrected = vdw / (1 - beta1 ** t)
    sdw_corrected = sdw / (1 - beta2 ** t)
    # Updating value (var)
    var = var - alpha * (vdw_corrected / (np.sqrt(sdw_corrected) + epsilon))
    return var, vdw, sdw
