#!/usr/bin/env python3
"""
Performs a strided convolution on images with multiple kernels/filters
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    a function that performs a convolution with multiple kernel/filters
    """
    c_images, images_h, images_w, _ = images.shape
    f_height = kernels.shape[0]
    f_width = kernels.shape[1]
    nc = kernels.shape[3]
    stride_h, stride_w = stride

    if padding == "same":
        padding_h = ((images_h - 1) * stride_h + f_height - images_h) // 2 + 1
        padding_w = ((images_w - 1) * stride_w + f_width - images_w) // 2 + 1
    elif padding == "valid":
        padding_h, padding_w = (0, 0)
    else:
        padding_h, padding_w = padding

    c_height = (images.shape[1] + 2 * padding_h - f_height) // stride_h + 1
    c_width = (images.shape[2] + 2 * padding_w - f_width) // stride_w + 1
    # np.pad works with a before_N and after_N parameter defined in a tuple
    # that will add the selected pad at each dimension
    pad_img = np.pad(images, ((0, 0), (padding_h, padding_h), (padding_w,
                                                               padding_w),
                              (0, 0)))

    convolved = np.zeros((c_images, c_height, c_width, nc))
    for row in range(c_height):
        for col in range(c_width):
            for ker in range(nc):
                pad_ele = pad_img[:, row * stride_h:row * stride_h + f_height,
                                  col * stride_w:col * stride_w + f_width]
                sum_mul_ele = np.sum(pad_ele * kernels[:, :, :, ker],
                                     axis=(1, 2, 3))
                convolved[:, row, col, ker] = sum_mul_ele
    return convolved
