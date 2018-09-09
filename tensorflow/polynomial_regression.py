# -*- coding: utf-8 -*-
"""
Example for a polynomial regression in TensorFlow using a quadratic function.

@author: Christopher Masch
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.0005
nb_epochs = 10

# Generating data
X_train = np.arange(10).reshape((10,1))
y_train = np.array([i * i for i in[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

# Placeholder
X = tf.placeholder(tf.float32, name="X")
y = tf.placeholder(tf.float32, name="y")

# Weights
W = tf.Variable(0.0, name="weight_1")
U = tf.Variable(0.0, name="weight_2")
b = tf.Variable(0.0, name="bias")

# Function
function    = X * X * U + X * W + b
y_predicted = tf.squeeze(function)

sqr_error = tf.square(y - y_predicted)
mean_loss = tf.reduce_mean(sqr_error, name='mean_loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(mean_loss)

def predict_reg(sess, X_data):
    """Predict y of X"""
    return sess.run(y_predicted, feed_dict={X:X_data})

with tf.Session() as sess:
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    
    # Displaying graph in TensorBoard
    tf.summary.FileWriter(graph=sess.graph, logdir='tensorboard_dir')
    
    # Run training
    training_loss = []
    for i in range(nb_epochs):
        _, loss = sess.run([optimizer, mean_loss],
                           feed_dict={X:X_train, y:y_train})
        training_loss.append(loss)
        print('Epoch: {:2} ; Loss {:0.5f}'.format(i+1, loss))
    
    
    # Plot results
    plt.scatter(X_train, y_train, marker='o', s=80, label='Training data')
    plt.plot(X_train, predict_reg(sess, X_train), 'r', label='Regression model')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    plt.plot(range(1, len(training_loss)+1), training_loss)
    plt.tight_layout()
    plt.xlabel('Epoch')
    plt.ylabel('Training Cost')
    plt.show()