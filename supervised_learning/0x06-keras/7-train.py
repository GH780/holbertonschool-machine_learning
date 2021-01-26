  
#!/usr/bin/env python3
"""
Updates the function train model with learning decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    a function that trains a model using mini-batch gradient descent
    """

    def l_r_decay(epoch):
        """
        a function that performs step decay
        :param epoch:
        :return:
        """
        return alpha / (1 + decay_rate * epoch)

    callbacks = []
    if early_stopping is True and validation_data is not None:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience,
                                                   monitor="val_loss"))
    if learning_rate_decay and validation_data is not None:
        callbacks.append(K.callbacks.LearningRateScheduler(l_r_decay,
                                                           verbose=1))

    history = network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                          verbose=verbose, validation_data=validation_data,
                          shuffle=shuffle, callbacks=callbacks)
    return history
