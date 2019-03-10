# -*- coding: utf-8 -*-
"""
@author: Christopher Masch
"""

import numpy as np

def random_linear_data(deviation=50):
    """
    Generate random data for linear regression.
    
    Arguments:
        deviation : Defines the deviation of the data.
                    Smaller value generates larger deviation.         
    Return:
        x, y      : Generated random data for x, y
    """
    x = np.random.uniform(low=0, high=10, size=100)
    y = []
    for t in x:
        r = np.random.normal(loc=0.0, scale=(0.2 + t * t / deviation))
        y.append(r)
    return x, 1.316 * x + 0.678 + np.array(y)