#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    a function that performs a same convolution
    """
    c_images = images.shape[0]
    f_height = kernel.shape[0]
    f_width = kernel.shape[1]

    padding_h, padding_w = padding

    c_height = images.shape[1] + 2 * padding_h - f_height + 1
    c_width = images.shape[2] + 2 * padding_w - f_width + 1
    # np.pad works with a before_N and after_N parameter defined in a tuple
    # that will add the selected pad at each dimension
    pad_images = np.pad(images, ((0, 0), (padding_h, padding_h), (padding_w,
                                                                  padding_w)))
    convolved = np.zeros((c_images, c_height, c_width))
    for row in range(c_height):
        for col in range(c_width):
            mul_ele = pad_images[:, row:row + f_height, col:col + f_width] * \
                      kernel
            sum_ele = np.sum(mul_ele, axis=(1, 2))
            convolved[:, row, col] = sum_ele
    return convolved
