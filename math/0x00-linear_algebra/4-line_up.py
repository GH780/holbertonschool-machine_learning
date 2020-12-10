#!/usr/bin/env python3
"""
module
"""


def add_arrays(arr1, arr2):
    """
    function two lists
    """

    result = []

    if len(arr1) != len(arr2):
        return None

    for i in range(len(arr1)):
        result.append(arr1[i] + arr2[i])
    return result
