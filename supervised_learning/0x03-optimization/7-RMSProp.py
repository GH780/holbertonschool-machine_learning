#!/usr/bin/env python3
"""
Updates a variable using the RMSprop optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    a function that updates a variable using RMSprop
    """
    new_moment = beta2 * s + (1 - beta2) * grad ** 2
    updated = var - alpha * grad / (new_moment ** (1 / 2) + epsilon)
    return updated, new_moment
