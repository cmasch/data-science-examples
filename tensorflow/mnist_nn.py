# -*- coding: utf-8 -*-
"""
Simple Neural Network with two hidden layers for MNIST-Classification.

This model achieves an accuracy over ~96% after 3 epochs. Just play with the hyperparameters to find out how the results
change. You can find them in them in the `parse_args` method.
Because of the simplicity I left the validation data out. Feel free to add it in the `fit` method.

# Requirements
- Download the MNIST data from [Yann LeCun](http://yann.lecun.com/exdb/mnist/) and use the
  flag `input_dir` to load the raw data of the given directory.

@author: Christopher Masch
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sys

FLAGS = None

class Model:
    
    def __init__(self):
        self.input_features = 784
        self.mnist = input_data.read_data_sets(FLAGS.input_dir)
        self.sess = tf.Session()
        self.build_graph()
        
    def build_graph(self):
        """Building graph"""
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        
        with tf.variable_scope('input_data'):
            self.X   = tf.placeholder(tf.float32, [None, self.input_features], name='X')
            self.y   = tf.placeholder(tf.int64, [None], name='y')
                
        with tf.variable_scope('dense_1'):
            w = tf.get_variable(shape=[self.input_features, FLAGS.hidden_units], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                name='weights')
            b = tf.get_variable(shape=[FLAGS.hidden_units], initializer=tf.constant_initializer(0.1),
                                name='biases')
            self.dense1 = tf.nn.relu(tf.matmul(self.X, w) + b, name='relu')
            
        with tf.variable_scope('dense_2'):
            w = tf.get_variable(shape=[FLAGS.hidden_units, FLAGS.hidden_units], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                name='weights')
            b = tf.get_variable(shape=[FLAGS.hidden_units], initializer=tf.constant_initializer(0.1),
                                name='biases')
            self.dense2 = tf.nn.relu(tf.matmul(self.dense1, w) + b, name='relu')
            
        with tf.variable_scope('classifier'):
            w = tf.get_variable(shape=[FLAGS.hidden_units, FLAGS.nb_classes], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                name='weights')
            b = tf.get_variable(shape=[FLAGS.nb_classes], initializer=tf.constant_initializer(0.1),
                                name='biases')
            self.logits = tf.matmul(self.dense2, w) + b

        with tf.variable_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
 
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss, global_step=self.global_step)
                 
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        with tf.variable_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.summary = tf.summary.merge_all()
            
    def fit(self):
        """Running computation on graph"""
        
        self.sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(FLAGS.log_dir, self.sess.graph)
        total_batches = int(self.mnist.train.num_examples / FLAGS.batch_size)
        
        for index in range(FLAGS.nb_epochs):
            print('###############\nEpoch:', index+1)
            for _ in range(total_batches):
                X_batch, y_batch = self.mnist.train.next_batch(FLAGS.batch_size)
                _, summary, loss, acc, current_step = self.sess.run([self.optimizer, self.summary, self.loss,
                                                                     self.accuracy, self.global_step],
                                                                    feed_dict={self.X:X_batch, self.y:y_batch})
                # Saving summary for displaying loss / accuracy in TensorBoard for each step
                writer.add_summary(summary, global_step=current_step)
                
                # Displaying every `display_step` the current loss / accuracy
                if(current_step%FLAGS.display_step == 0 or current_step%total_batches == 0):
                    print('Step: {} -- Loss: {:0.4f} -- Accuracy: {:0.2f}'.format(current_step, loss, acc))
    
    def evaluate(self):
        """Evaluate the trained model with test data"""
        
        print('\n* Starting evaluation with test data *')
        test_batch_size = 100
        n_batches = int(self.mnist.test.num_examples/test_batch_size)
        total_correct = 0
        total_losses = 0
        for i in range(n_batches):
            X_test, y_test = self.mnist.test.next_batch(test_batch_size)
            _, loss_batch, logits_batch = self.sess.run([self.optimizer, self.loss, self.logits],
                                                        feed_dict={self.X: X_test, self.y:y_test}) 
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds,1), y_test)
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct += self.sess.run(accuracy)
            total_losses += loss_batch
        print("Loss: {:0.4f} -- Accuracy: {:0.4f}".format(total_losses/n_batches, total_correct/self.mnist.test.num_examples))

def main(_):
    model = Model()
    model.fit()
    model.evaluate()

def parse_args(parser):
    """Method for parsing args"""
    parser.add_argument('--input_dir', default='data', type=str, help='Directory with MNIST files')
    parser.add_argument('--log_dir', default='log_dir', type=str, help='Directory for saving summaries and graph')
    parser.add_argument('--display_step', default=100, type=int, help='Steps after displaying loss and accuracy (default: 100)')
    parser.add_argument('--nb_classes', default=10, type=int, help='Number of classes e.g. 10 for MNIST (default: 10)')
    # Hyperparameter
    parser.add_argument('--learning_rate', default=0.0003, type=float, help='Learning rate (default: 0.0003)')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size (default: 100)')   
    parser.add_argument('--nb_epochs', default=4, type=int, help='Number of epochs (default: 4)')
    parser.add_argument('--hidden_units', default=1024, type=int, help='Units per hidden layer (default: 1024)')
    return parser

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)