#!/usr/bin/env python3
"""this module foe neuron class"""


class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """this function is __init__"""       
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        else:
            __self__.nx = nx
        self.w = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
