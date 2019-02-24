# -*- coding: utf-8 -*-
"""
Custom Callbacks

You can find other examples and the inherited class in the Github Repo [1].

# References
- [1][Keras Callback](https://github.com/keras-team/keras/blob/master/keras/callbacks.py#L275)

@author: Christopher Masch
"""

import keras
import numpy as np
from sklearn import metrics

class Metrics_Callback(keras.callbacks.Callback):
    """
    This is an example to evaluate a model on several metrics after each epoch.
    The values of self.model and self.validation_data refers to inherited class of `keras.callbacks.Callback`.
    """
    
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        print('Starting evaluating')
        y_pred = self.model.predict(self.validation_data[0])
        
        # Converting targets from e.g. [[0,1,0], [1,0,0]] to [1,0]
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = np.argmax(self.validation_data[1], axis=-1)
        
        print('Classification report:\n', metrics.classification_report(y_pred=y_pred, y_true=y_true))
        print('Confusion matrix:\n',      metrics.confusion_matrix(y_pred=y_pred, y_true=y_true))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return