#!/usr/bin/env python3
"""
module
"""


def mat_mul(mat1, mat2):
    """
    function
    """

    if len(mat1) != len(mat2[0]) and len(mat1[0]) != len(mat2):
        return None

    if len(mat1) == len(mat2[0]):
        mati = mat2
        matj = mat1
    else:
        mati = mat1
        matj = mat2

    mat3 = [[0 for i in matj[0]] for j in mati]

    for i in range(len(mati)):
        for j in range(len(matj[0])):
            for k in range(len(matj)):
                mat3[i][j] += mat1[i][k]*mat2[k][j]
    return mat3
