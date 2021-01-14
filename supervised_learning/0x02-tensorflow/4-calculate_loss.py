#!/usr/bin/env python3
"""
Module to calculate the loss of a prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    a function that calculates the loss of a prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
