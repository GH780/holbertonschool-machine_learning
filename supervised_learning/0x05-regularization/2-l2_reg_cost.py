#!/usr/bin/env python3
"""
Calculates the cost of a neural network with L2 regularization using tensorflow
"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    a function that calculates the cost of a NN with L2 regularization with
    tensorflow
    """
    return cost + tf.losses.get_regularization_losses()
