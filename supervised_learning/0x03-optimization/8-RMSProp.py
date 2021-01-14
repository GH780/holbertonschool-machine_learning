#!/usr/bin/env python3
"""
Updates a variable using the RMSprop optimization algorithm with tensorflow
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    a function that optimizes using RMSprop with tensorflow
    """
    rms = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return rms.minimize(loss)
