# -*- coding: utf-8 -*-
"""
Convolutional Neural Network (CNN) for MNIST-Classification.

This model achieves an accuracy over ~98% after 2-3 epochs. Just play with the hyperparameters to find out how the results
change. You can find them in them in the `parse_args` method.
Because of the simplicity I left the validation data out. Feel free to add it in the `fit` method.

Architecture:
1. Conv + ReLU
2. Maxpooling
3. Conv + ReLU
4. Maxpooling
5. Dense
6. Dropout
7. Classifier

You get a better overview of the architecture/graph in TensorBoard (tensorboard --logdir log_dir/)

# Requirements
- Download the MNIST data from [Yann LeCun](http://yann.lecun.com/exdb/mnist/) and use the
  flag `input_dir` to load the raw data of the given directory.

# References
- [1] [A Guide to TF Layers: Building a Convolutional Neural Network](https://www.tensorflow.org/tutorials/layers)

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

class CNN:
    
    def __init__(self):
        self.mnist = input_data.read_data_sets(FLAGS.input_dir)
        self.sess = tf.Session()
        self.build_graph()
        
    def build_graph(self):
        """Building graph"""
        
        self.global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
        
        with tf.variable_scope('input_data'):
            self.X   = tf.placeholder(tf.float32, [None, 784], name='X')
            self.img = tf.reshape(self.X, [-1, 28, 28, 1], name='X_reshape')
            self.y   = tf.placeholder(tf.int64, [None], name='y')
            
        with tf.variable_scope('conv_1'):
            w = tf.get_variable(shape=[5,5,1,32], initializer=tf.truncated_normal_initializer(stddev=0.1), name='weights')
            b = tf.get_variable(shape=[32], initializer=tf.constant_initializer(0.1), name='biases')
            conv1 = tf.nn.conv2d(self.img, w, strides=[1, 1, 1, 1], padding='SAME', name='conv2d') 
            self.relu1 = tf.nn.relu(conv1 + b, name='relu')
            
        with tf.variable_scope('maxpool_1'):
            self.pool1 = tf.nn.max_pool(self.relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            
        with tf.variable_scope('conv_2'):
            w = tf.get_variable(shape=[5,5,32,64], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                name='weights')
            b = tf.get_variable(shape=[64], initializer=tf.constant_initializer(0.1),
                                name='biases')
            conv2 = tf.nn.conv2d(self.pool1, w, strides=[1,1,1,1], padding='SAME', name='conv2d')
            self.relu2 = tf.nn.relu(conv2 + b, name='relu')
            
        with tf.variable_scope('maxpool_2'):
            self.pool2 = tf.nn.max_pool(self.relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
                
        with tf.variable_scope('dense'):
            input_features = 7*7*64
            w = tf.get_variable(shape=[input_features, FLAGS.hidden_units], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                name='weights')
            b = tf.get_variable(shape=[FLAGS.hidden_units], initializer=tf.constant_initializer(0.1),
                                name='biases')
            pool2 = tf.reshape(self.pool2, [-1, input_features])
            self.dense = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')
        
        with tf.variable_scope('dropout'):
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            self.keep_prob = tf.nn.dropout(self.dense, self.dropout_rate)   
            
        with tf.variable_scope('classifier'):
            w = tf.get_variable(shape=[FLAGS.hidden_units, FLAGS.nb_classes], initializer=tf.truncated_normal_initializer(stddev=0.1),
                                name='weights')
            b = tf.get_variable(shape=[FLAGS.nb_classes], initializer=tf.constant_initializer(0.1),
                                name='biases')
            self.logits = tf.matmul(self.keep_prob, w) + b

        with tf.variable_scope('loss'):
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy, name='loss')
            
        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.loss, global_step=self.global_step)
         
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y)
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
                                                                    feed_dict={self.X:X_batch, self.y:y_batch,
                                                                               self.dropout_rate:float(FLAGS.dropout_rate)})
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
                                                        feed_dict={self.X: X_test, self.y:y_test, self.dropout_rate: 1.0}) 
            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds,1), y_test)
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct += self.sess.run(accuracy)
            total_losses += loss_batch
        print("Loss: {:0.4f} -- Accuracy: {:0.4f}".format(total_losses/n_batches, total_correct/self.mnist.test.num_examples))

def main(_):
    model = CNN()
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
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='Droput rate (default: 0.4)')
    parser.add_argument('--nb_epochs', default=4, type=int, help='Number of epochs (default: 4)')
    parser.add_argument('--hidden_units', default=1024, type=int, help='Units per hidden layer (default: 1024)')
    return parser

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    parser = argparse.ArgumentParser()
    parser = parse_args(parser)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)