#!/usr/bin/env python3
"""function"""


def poly_derivative(poly):
    """poly"""
    list = []
    if (len(poly) == 0):
        return None
    elif (len(poly) == 1):
        return 0
    else:
        for i in range(1, len(poly)):
            list.append(i * poly(i))
        return list
