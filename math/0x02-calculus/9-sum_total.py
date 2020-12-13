#!/usr/bin/env python3
""" function"""


def summation_i_squared(n):
    """sum function"""
    sum = 0
    if(n > 1):
        for(i in range(1, n+1)):
            sum += i*i
        return n
    else:
        return None
