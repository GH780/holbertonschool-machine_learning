#!/usr/bin/env python3
"""
module for trans matrix
"""


def matrix_transpose(matrix):
    """
    function
    """
    matrix_result = [[0 for i in matrix] for j in matrix[0]]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_result[j][i] = matrix[i][j]
    return matrix_result
