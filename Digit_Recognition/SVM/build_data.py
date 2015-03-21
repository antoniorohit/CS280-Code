from scipy import signal
import numpy as np
import random
import cPickle as pickle
import os
import scipy.ndimage.filters as filters
from scipy import io

def getDataFromFiles():
    
    testFileMNIST = "../digit-dataset/test.mat"
    trainFileMNIST = "../digit-dataset/train.mat"
    trainMatrix = io.loadmat(trainFileMNIST)                 # Dictionary
    testMatrix = io.loadmat(testFileMNIST)                   # Dictionary
    
    testData = np.array(testMatrix['test_images'])
    testData = np.rollaxis(testData, 2, 0)                # move the index axis to be the first 
    testData_flat = []
    for elem in testData:
        testData_flat.append(elem.flatten())
    imageData = np.array(trainMatrix['train_images'])
    imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
    imageLabels = np.array(trainMatrix['train_labels'])
    
    return imageData, imageLabels

def getDataFromPickle():
    # Arrays to hold the shuffled data and labels
    shuffledData = list()
    shuffledLabels = list()

    if(os.path.isfile("./Results/shuffledData.p")):
        print('opening data with pickle...')
        shuffledData = pickle.load(open("./Results/shuffledData.p", 'rb'))
        print('data successfully opened')
        print('opening labels with pickle..')
        shuffledLabels = pickle.load(open("./Results/shuffledLabels.p", 'rb'))
        print('labels successfully opened')
        
    return shuffledData, shuffledLabels

def buildData(clf):
    """get data, compute features and shuffle evrything.
    
    :param gauss_bool (boolean) use a gaussian filter if true, gradient one otherwise
    :param clf.imageData contains all data, labels
    :return: shuffledData, shuffledLabels, imageComplete"""

    imageData, imageLabels = getDataFromFiles()

    ############# 
    # Ink Normalization
    ############# 
    for i in range(len(imageData)):
        aux_norm = np.linalg.norm(imageData[i])
        if aux_norm != 0:
            imageData[i] /= aux_norm
    
    clf.imageComplete = zip(imageData, imageLabels)
    
    if clf.DEBUG:
        print 50*'-'
        print ("Shapes of image data and labels: ", clf.imageData.shape, 
                                        clf.imageLabels.shape, len(clf.imageComplete))
                
    ############# SET ASIDE VALIDATION DATA (10,000) ############# 
    # SHUFFLE THE IMAGES
    random.shuffle(clf.imageComplete)
    
    clf.shuffledData, clf.shuffledLabels = getDataFromPickle()
    
    if len(clf.shuffledLabels)>0:
        return clf.shuffledData, clf.shuffledLabels, clf.imageComplete

    for ind in range(len(clf.imageComplete)):
        if ind % 100 ==0:
            print 'feature extraction :' + str(ind*100./len(clf.imageComplete))+ ' % over'
        
        if clf.gauss_bool:
            gaussFirst_x = filters.gaussian_filter1d(clf.imageComplete[i][0], 1, order = 1, axis = 0)
            gaussFirst_y = filters.gaussian_filter1d(clf.imageComplete[i][0], 1, order = 1, axis = 1)
            ori = np.array(np.arctan2(gaussFirst_y, gaussFirst_x))

        else:
            grad_filter = np.array([[-1, 0, 1]])
            gradx = signal.convolve2d(clf.imageComplete[i][0], grad_filter, 'same')
            grady = signal.convolve2d(clf.imageComplete[i][0], np.transpose(grad_filter), 'same')
            ori = np.array(np.arctan2(grady, gradx))
        
        ori_4_hist = list()
        ori_7_hist = list()
                     
        ori_4_1 = blockshaped(ori, 4, 4)
        ori_4_2 = blockshaped(ori[2:-2, 2:-2], 4, 4)
        
        for (elem1, elem2) in zip(ori_4_1, ori_4_2):
            ori_4_hist.append(np.histogram(elem1.flatten(), clf.n_bins, (-np.pi, np.pi))[0])
            ori_4_hist.append(np.histogram(elem2.flatten(), clf.n_bins, (-np.pi, np.pi))[0])
    
        ori_7_1 = (blockshaped(ori, 7, 7))
        ori_7_2 = (blockshaped(ori[3:-4, 3:-4], 7, 7))
        
        for elem1, elem2 in zip(ori_7_1, ori_7_2):
            ori_4_hist.append(np.histogram(elem1.flatten(), clf.n_bins, (-np.pi, np.pi))[0])
            ori_4_hist.append(np.histogram(elem2.flatten(), clf.n_bins, (-np.pi, np.pi))[0])
        
        ori_4_hist = np.float64(ori_4_hist)/(np.linalg.norm(ori_4_hist))
        ori_7_hist = np.float64(ori_7_hist)/(np.linalg.norm(ori_7_hist))
        
        clf.shuffledData.append(np.append(ori_4_hist, ori_7_hist))
        clf.shuffledLabels.append((clf.imageComplete[i][1][0]))
                
    pickle.dump(clf.shuffledData, open("./Results/shuffledData.p", 'wb'))
    pickle.dump(clf.shuffledLabels, open("./Results/shuffledLabels.p", 'wb'))
    
    return
        
def blockshaped(arr, nrows, ncols):
    """If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    :param arr
    :param nrows
    :param ncols
    :return: arr (array) array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))