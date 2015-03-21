# coding utf-8

import numpy as np

def my_kernel(x, y, first_dim):
    """define custom kernel between vectors x and y.
    
    :param x data
    :param y data
    :param first_dim 
    :return: result linear kernel between x and y for first_dim first dimensions, 
    intersection kernel on others."""
    
    hist = np.sum(np.absolute(np.minimum(x[first_dim:],y[first_dim:])))
    lin = np.dot(x[:first_dim], y[:first_dim].T)
    return hist + lin