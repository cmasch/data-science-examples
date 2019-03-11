# -*- coding: utf-8 -*-
"""
Attention Layer

If you need an overview about attention mechanism, please check out this blog:
https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/

# References
- [1] [Hierarchical Attention Networks for Document Classification](http://www.aclweb.org/anthology/N16-1174)
- [2] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

@author: Christopher Masch
"""

from keras import initializers
from keras import backend as K
from keras.engine import Layer

class RecurrentAttention(Layer):
    """
    Attention layer for recurrent layer e.g. LSTM, GRU.
    This model is really straightforward. If you search for other implementations, they will differ in some way.
    
    Scores:
        IMDB (test): 0.2434 loss / 0.9047 acc 
    
    Usage:
        from attention_layer import RecurrentAttention
        x = Bidirectional(LSTM(128, recurrent_dropout=0.4, dropout=0.5, return_sequences=True))(x)
        x = RecurrentAttention()(x)
    """

    def __init__(self, kernel_initializer='glorot_uniform', **kwargs):
        self.kernel_initializer = initializers.get(kernel_initializer)
        super(RecurrentAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        
        if len(input_shape) != 3:
            raise Exception('The input should have 3 dimensions. Maybe you forgot: `return_sequences = True` for RNN layer.')
        
        self.W = self.add_weight(
            shape = (input_shape[-1], input_shape[-1],),
            initializer = self.kernel_initializer,
            name = '{}_W'.format(self.name),
        )
        
        self.b = self.add_weight(
            shape = (input_shape[-1],),
            initializer = 'zero',
            name = '{}_bias'.format(self.name)
        )
        
        self.u = self.add_weight(
            shape = (input_shape[-1],),
            initializer = self.kernel_initializer,
            name = '{}_u'.format(self.name),
        )
        
        self.trainable_weights = [self.W, self.b, self.u]
        super(RecurrentAttention, self).build(input_shape)

    def call(self, x):
        a = K.dot(x, self.W)
        a += self.b
        a = K.tanh(a)
        a = K.dot(a, K.expand_dims(self.u))
        a = K.squeeze(a, axis = -1)
        a = K.softmax(a)
        a = x * K.expand_dims(a)
        return K.sum(a, axis = 1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]
    
    def get_config(self):
        config = {
            'name' : self.name,
        }
        base_config = super(RecurrentAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))