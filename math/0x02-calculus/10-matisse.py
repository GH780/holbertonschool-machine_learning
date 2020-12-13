#!/usr/bin/env python3
"""function"""


def poly_derivative(poly):
    """poly"""
    if type(poly) is not list or poly == []:
        return None
    elif len(poly) < 2:
        return [0]
    else:
        exponent = 1
        derivative = poly.copy()
        derivative.pop(0)
        for i in range(len(derivative)):
            if type(derivative[i]) is int or type(derivative[i]) is float:
                derivative[i] = derivative[i] * exponent
                exponent += 1
            else:
                return None
        return derivative
