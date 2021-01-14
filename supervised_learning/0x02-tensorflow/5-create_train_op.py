#!/usr/bin/env python3
"""
Module to create the training operation
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    a function creates the training operation for the network:
    """
    return tf.train.GradientDescentOptimizer(alpha).minimize(loss)
