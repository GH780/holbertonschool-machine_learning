#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm with tensorflow
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    a function that optimizes using Adam with tensorflow
    """
    return tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon).minimize(loss)
