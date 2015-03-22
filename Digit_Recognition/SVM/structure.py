# coding utf-8

from build_data import buildData
from train import train_svm
from visualize import plot_error_rate
from cross_validation import crossValidation

class my_clf:
    
    def __init__(self, DEBUG = False, visualization = False, n_bins = 0, training_sizes = list(), testing_size = 0):
        
        self.DEBUG = DEBUG
        self.visualization = visualization
        
        self.n_bins = n_bins
        self.training_sizes = training_sizes
        self.testing_size = testing_size
        
    def build(self):
        
        buildData(self)
        
    def train(self):
        
        train_svm(self)
        
    def plot(self):
        
        plot_error_rate(self)
        
    def cross_validate(self):
        
        crossValidation(self)
        
        
        
        
    