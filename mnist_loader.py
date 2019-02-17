# -*- coding: utf-8 -*-
"""
If you are using MNIST or Fashion-MNIST you can use the method `load_data()` to load the raw files.
This will create numpy arrays which can be used for training and evaluating models.

# References
- [1] [The MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [2] [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

@author: Christopher Masch
"""

import gzip
import numpy as np

img_size = 28


def load_data(image_path, label_path):
    # Loading images
    with gzip.open(image_path, 'r') as images_gzip:
        images_gzip.read(16)
        buf = images_gzip.read()
        images = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = images.reshape(int(len(buf) / img_size ** 2), img_size, img_size, 1)

    # Loading labels
    with gzip.open(label_path, 'r') as labels_gzip:
        labels_gzip.read(8)
        buf = labels_gzip.read()
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        labels = labels.reshape(len(buf), 1)

    return images, labels


X_train, y_train = load_data('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
X_test, y_test = load_data('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')