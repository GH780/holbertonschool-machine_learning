#!/usr/bin/env python3
"""function"""


def poly_derivative(poly):
    """poly"""
    list = poly.copy()
    if (len(poly) == 0):
        return None
    elif (len(poly) == 1):
        return 0
    else:
        for i in range(len(poly)):
            list [i] = i * poly[i]
        return list[1:]
