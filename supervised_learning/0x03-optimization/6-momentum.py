#!/usr/bin/env python3
"""
Updates a variable using the gradient descent with momentum with tensorflow
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    a function that uses momentum optimization algorithm with tensorflow
    """
    # the minimize method of MomentumOptimizer simply the compute_gradients()
    # and apply_gradients() calls
    return tf.train.MomentumOptimizer(alpha, momentum=beta1).minimize(loss)
