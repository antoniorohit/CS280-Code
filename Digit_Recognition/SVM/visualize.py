# coding utf-8

import matplotlib.pyplot as plt

def plot_error_rate(clf, label=""):
    """Plot error rates vs training size, and export figure.
    
    :param training_sizes : list of training sizes
    :param errorRate_array : list of error rates with custom kernel
    :param linear_errorRate_array : list of error rates with linear kernel
    :param label : specify label for exported images"""
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Error Rate Vs Training Size')
    ax.set_ylabel('Training Size')
    ax.set_xlabel('Error Rate')
    
    training_sizes = list()
    errorRate_array = list()
    linear_errorRate_array = list()
    for cust_model in clf.cust_models:
        training_sizes.append(cust_model.training_size)
        errorRate_array.append(cust_model.error_rate)
    
    for lin_model in clf.lin_models:
        linear_errorRate_array.append(lin_model.error_rate)
        
    intersect, = ax.plot(training_sizes, errorRate_array, label='Intersection kernel')
    linear, = ax.plot(training_sizes, linear_errorRate_array, label='Linear kernel')
    ax.legend([intersect, linear], ['Intersection kernel', 'Linear kernel'])
    
    for xy in zip(training_sizes, errorRate_array):                                                # <--
        ax.annotate('%s' % int(xy[1]) + "%", xy=xy, fontsize = 'small') # <--
    
    plt.grid()
    plt.savefig("./Results/ErrorRate" + label + ".png")
    plt.close()
    
    return