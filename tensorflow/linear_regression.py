# -*- coding: utf-8 -*-
"""
Example for linear regression in TensorFlow 2.x

@author: Christopher Masch
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
nb_epochs = 10

# Generating linear data
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([i for i in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])


# Building linear layer
class LinearLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(LinearLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            initializer=tf.keras.initializers.zeros(),  # Dont use ones!
            trainable=True,
            name='weight'
        )

    @tf.function
    def call(self, inputs):
        return inputs * self.w


model = tf.keras.Sequential([
    LinearLayer(input_shape=(1,)),
    # keras.layers.Dense(10), # you can also add Dense layer
    # keras.layers.Dense(1),
])

optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

model.compile(
    loss='mse',
    optimizer=optimizer
)

history = model.fit(
    X_train, y_train,
    batch_size=len(X_train),
    epochs=nb_epochs,
    verbose=1,
)

# Plot results
plt.scatter(X_train, y_train, marker='o', label='Training data')
plt.plot(X_train, model.predict(X_train), 'r', label='Regression model')
plt.legend()
plt.show()

plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

