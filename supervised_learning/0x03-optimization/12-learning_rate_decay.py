#!/usr/bin/env python3
"""
Updates the learning rate using inverse time decay with tensorflow
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    a function that updates the learning rate with tensorflow
    """
    # staircase= True will make discrete output as floor(global_step /
    # decay_step in the calculation)
    return tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                       decay_rate, staircase=True)
