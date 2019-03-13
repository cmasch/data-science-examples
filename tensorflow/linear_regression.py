# -*- coding: utf-8 -*-
"""
Example for linear regression in TensorFlow 2.x
This demo includes two different approaches (Keras & Gradient Tape)

@author: Christopher Masch
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Parameter
learning_rate = 0.001
nb_epochs     = 10
batch_size    = 5
gen_samples   = 50
use_keras     = True # If False, Gradient Tape is used instead of Keras Sequential API

# Generating linear data
X_train = np.arange(gen_samples).reshape((gen_samples,1)).astype(np.float32)
y_train = np.array([i for i in range(gen_samples)]).astype(np.float32)
train   = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=batch_size).batch(batch_size)

# Building linear layer (x*w+b)
class LinearLayer(tf.keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super(LinearLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w = self.add_variable(
            shape=[1,1],
            initializer=tf.initializers.zeros(), # Dont use ones!
            trainable=True,
            name='weight'
        )
        
        self.b = self.add_variable(
            shape=[1],
            initializer=tf.initializers.zeros(),
            trainable=True,
            name='bias'
        )
  
    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Defining learning algo and loss function
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
loss_min  = tf.losses.MSE

# Metrics
losses = []
preds  = []

if use_keras:
    # Using Keras Sequential API
    model = tf.keras.Sequential([LinearLayer()])
    model.compile(
        loss=loss_min,
        optimizer=optimizer
    )
    history = model.fit(
        train,
        epochs=nb_epochs,
        verbose=1,
    )
    preds  = model.predict(X_train)
    losses = history.history['loss']
else:
    # Using automatic differentiation and gradient tape
    linear_model = LinearLayer()
    for epoch in range(nb_epochs):
        print("Epoch {}/{}".format(epoch+1,nb_epochs))
        for (x, y) in train:
            with tf.GradientTape() as tape:
                logits = linear_model(x)
                logits = tf.squeeze(logits, axis=1)
                loss   = loss_min(y, logits)
            grads = tape.gradient(loss, linear_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, linear_model.trainable_variables))
        print("loss:", loss.numpy())
        losses.append(loss.numpy())
    preds = linear_model(X_train)

# Plot results
plt.scatter(X_train, y_train, marker='o', label='Training data')
plt.plot(X_train, preds, 'r', label='Regression model')
plt.legend()
plt.show()

plt.plot(range(1, len(losses)+1), losses)
plt.tight_layout()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
