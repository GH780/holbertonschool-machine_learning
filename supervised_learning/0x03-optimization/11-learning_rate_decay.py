#!/usr/bin/env python3
"""
Updates the learning rate using inverse time decay in numpy
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    a function that updates the learning rate
    """
    epoch = global_step // decay_step
    new_alpha = alpha / (1 + decay_rate * epoch)
    return new_alpha
