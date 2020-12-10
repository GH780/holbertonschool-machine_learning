#!/usr/bin/env python3
"""
module
"""


def add_matrices2D(mat1, mat2):
    """
    function two lists
    """

    result = []

    if len(arr1) != len(arr2):
        return None

    for i in range(len(arr1)):
        for j in range(len arr1[i]):
            result.append(arr1[i][j] + arr2[i][j])
    return result
