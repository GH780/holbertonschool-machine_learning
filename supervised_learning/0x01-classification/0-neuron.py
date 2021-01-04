#!/usr/bin/env python3
"""this module foe neuron class"""


class Neuron:
    def __init__(self, nx):
        """ this function is __init__"""
        int w
        int b = 0
        int A = 0
        if type(nx) != int:
            raiseTypeError("nx must be an integer")
        elif nx < 1:
            raiseValueError("nx must be a positive integer")
        else:
            __self__.nx = nx
