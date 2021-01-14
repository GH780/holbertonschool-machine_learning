#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    a function that uses momentum optimization algorithm
    """
    momentum = beta1 * v + (1 - beta1) * grad
    updated = var - alpha * momentum
    return updated, momentum
