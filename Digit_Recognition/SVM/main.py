# coding utf-8

from structure import my_clf

DEBUG = False
visualization = False

# define filter to extract features
gauss_bool = False

# define number of bins in histograms
n_bins = 9

# define different training size to test
training_sizes = [100, 200, 500, 1000, 2000, 5000, 10000]
# training_sizes = [100, 200, 500]

# define testing size
testing_size = 10000

# create instance of my_clf class
clf = my_clf(DEBUG, visualization, n_bins, training_sizes, testing_size)

# open data from files  
clf.build()

# train svm model and compute accuracy
clf.train()

# plot results
clf.plot()

# use cross-validation to compute best regularization parameter
clf.cross_validate()