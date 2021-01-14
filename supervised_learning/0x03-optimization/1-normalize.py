#!/usr/bin/env python3
"""
Normalizes (standardizes) a matrix
"""


def normalize(X, m, s):
    """
    a function that normalizes (standardizes) a matrix
    """
    return (X - m) / s
