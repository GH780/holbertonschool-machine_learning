#!/usr/bin/env python3
"""
Conducts forward propagation using Dropout:
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    a function that conducts forward propagation using dropout
    """
    cache = {'A0': X}
    # Hidden and output layer
    for i in range(L):
        # create keys to access weights(W), biases(b) and store in cache
        key_w = 'W' + str(i + 1)
        key_b = 'b' + str(i + 1)
        key_cache = 'A' + str(i + 1)
        key_cache_last = 'A' + str(i)
        # store activation in cache
        output_Z = np.matmul(weights[key_w], cache[
            key_cache_last]) + weights[key_b]
        if i == L - 1:
            # Softmax
            t = np.exp(output_Z)
            output_A = np.exp(output_Z) / np.sum(t, axis=0,
                                                 keepdims=True)
        else:
            output_A = ((np.exp(output_Z) - np.exp(-output_Z)) / (
                np.exp(output_Z) + np.exp(-output_Z)))
            # dropout value is 0 if random number less than keep_prop
            rand = np.random.rand(output_A.shape[0], output_A.shape[1])
            dropout = (rand < keep_prob).astype(int)
            # dropout of the neuron if multiply by 0
            output_A = np.multiply(output_A, dropout)
            output_A = output_A / keep_prob
            cache['D{}'.format(i + 1)] = dropout
        cache[key_cache] = output_A
    return cache
