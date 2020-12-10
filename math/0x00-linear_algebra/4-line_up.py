#!/usr/bin/env python3
"""function"""


def add_arrays(arr1, arr2):
    """add matrix"""
    arr = [0 for i in range(len(arr1))] 
    if (len(arr1) == len(arr2)):
        for i in range(len arr1):
            arr[i] = arr1[i] + arr2[i]
        return arr
    else:
        return None
